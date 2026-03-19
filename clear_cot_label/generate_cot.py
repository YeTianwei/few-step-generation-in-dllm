"""
CoT生成模块
===========

利用动作原语和视频生成完整的思维链(Chain of Thought)标注。

流程:
1. 读取预计算的动作原语JSON
2. 提取视频帧
3. 构建Prompt
4. 调用VLM生成CoT
5. 解析并保存结果
"""

import os
import io
import json
import argparse
import random
import re
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import h5py
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont
import httpx

# 尝试导入API客户端
try:
    from volcenginesdkarkruntime import Ark
    HAS_ARK = True
except ImportError:
    HAS_ARK = False
    print("Warning: volcenginesdkarkruntime not installed, API calls will not work")


# ================================================================================
# LLM 客户端配置
# ================================================================================

def get_llm_client():
    """获取LLM客户端"""
    if not HAS_ARK:
        return None, None
    
    llm_client = Ark(
        base_url="https://ark-cn-beijing.bytedance.net/api/v3",
        api_key="5cf86bc0-724c-4814-adf2-f9738a55052b",
    )
    model_name = "ep-20251218134827-bmzx2"
    return llm_client, model_name


# ================================================================================
# 动作原语处理
# ================================================================================

def load_primitives_from_json(json_path: str) -> Optional[Dict]:
    """
    从JSON文件中读取动作原语
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        dict: 动作原语数据
    """
    if not os.path.exists(json_path):
        print(f"Warning: Primitives JSON not found: {json_path}")
        return None
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def format_primitives_for_prompt(
    primitives_data: Dict,
    fps: int = 15,
    sample_interval: int = 1
) -> str:
    """
    将动作原语格式化为Prompt中可用的文本
    
    Args:
        primitives_data: 动作原语数据
        fps: 帧率
        sample_interval: 采样间隔
    
    Returns:
        str: 格式化的文本
    """
    num_steps = primitives_data["num_steps"]
    left_primitives = primitives_data["left_primitives"]
    right_primitives = primitives_data["right_primitives"]
    
    lines = []
    lines.append("Frame | Time(s) | Left Arm | Right Arm")
    lines.append("-" * 80)
    
    for i in range(0, num_steps, sample_interval):
        time_sec = i / fps
        
        left_prim = left_primitives[i]["primitive"] if i < len(left_primitives) else "N/A"
        right_prim = right_primitives[i]["primitive"] if i < len(right_primitives) else "N/A"
        
        lines.append(f"{i:5d} | {time_sec:7.2f} | {left_prim:30s} | {right_prim:30s}")
    
    # 确保最后一帧被包含
    if (num_steps - 1) % sample_interval != 0:
        i = num_steps - 1
        time_sec = i / fps
        left_prim = left_primitives[i]["primitive"] if i < len(left_primitives) else "N/A"
        right_prim = right_primitives[i]["primitive"] if i < len(right_primitives) else "N/A"
        lines.append(f"{i:5d} | {time_sec:7.2f} | {left_prim:30s} | {right_prim:30s}")
    
    return "\n".join(lines)


# ================================================================================
# 视频处理
# ================================================================================

def extract_frames_multi_view(
    hdf5_path: str,
    cameras: List[str] = ["head_camera", "front_camera", "left_camera", "right_camera"]
) -> List[Image.Image]:
    """提取多视角视频帧并组合成2x2网格"""
    camera_frames = {}
    
    with h5py.File(hdf5_path, "r") as f:
        for camera in cameras:
            if camera in f["observation"]:
                rgb = f["observation"][camera]["rgb"][:]
                frames = []
                for b in rgb:
                    data = b.tobytes() if hasattr(b, "tobytes") else b
                    img = Image.open(io.BytesIO(data))
                    img_rgb = np.array(img.convert("RGB"))
                    img_array_rgb = img_rgb[:, :, ::-1]
                    img = Image.fromarray(img_array_rgb)
                    frames.append(img)
                camera_frames[camera] = frames
    
    if not camera_frames:
        return []
    
    num_frames = min(len(frames) for frames in camera_frames.values())
    
    combined_frames = []
    for i in range(num_frames):
        head_frame = camera_frames.get("head_camera", [None] * num_frames)[i]
        front_frame = camera_frames.get("front_camera", [None] * num_frames)[i]
        left_frame = camera_frames.get("left_camera", [None] * num_frames)[i]
        right_frame = camera_frames.get("right_camera", [None] * num_frames)[i]
        
        sample_frame = head_frame or front_frame or left_frame or right_frame
        if sample_frame is None:
            continue
        
        w, h = sample_frame.size
        grid = Image.new('RGB', (w * 2, h * 2), (0, 0, 0))
        
        if head_frame:
            grid.paste(head_frame, (0, 0))
        if front_frame:
            grid.paste(front_frame, (w, 0))
        if left_frame:
            grid.paste(left_frame, (0, h))
        if right_frame:
            grid.paste(right_frame, (w, h))
        
        draw = ImageDraw.Draw(grid)
        font = ImageFont.load_default()
        draw.text((10, 10), "Head", fill=(255, 255, 0), font=font)
        draw.text((w + 10, 10), "Front", fill=(255, 255, 0), font=font)
        draw.text((10, h + 10), "Left", fill=(255, 255, 0), font=font)
        draw.text((w + 10, h + 10), "Right", fill=(255, 255, 0), font=font)
        
        combined_frames.append(grid)
    
    return combined_frames


def save_video_from_frames(frames: List[Image.Image], output_path: str, fps: int = 15) -> str:
    """保存帧为视频"""
    if not frames:
        raise RuntimeError("No frames to save")
    arrs = [np.asarray(img.convert("RGB")) for img in frames]
    iio.imwrite(output_path, arrs, fps=float(fps), codec="libx264", macro_block_size=1)
    return output_path


# ================================================================================
# 文件上传
# ================================================================================

def upload_to_tos(file_path: str, prefix: str = "embodied-cot-clear", overwrite: bool = True, expires: int = 2592000) -> str:
    """上传文件到TOS并返回预签名URL"""
    import sys
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    from scripts.generate_embodied_data.simpleUploadTos import CustomTosClient
    
    tos_client = CustomTosClient(
        bucket="mimo-image",
        access_key="GQ75PH27G69SE36EQOIE",
        service_name="toutiao.tos.tosapi",
        idc="hl",
    )
    tos_client.upload_file(file_path, prefix=prefix, overwrite=overwrite)
    actual_object_key = prefix + "/" + os.path.basename(file_path)
    response = tos_client.client.presigned(actual_object_key, expires)
    presigned_url = response.headers.get("X-Tos-Presigned-Url")
    return presigned_url


# ================================================================================
# LLM调用
# ================================================================================

def call_vlm(video_url: str, prompt: str, llm_client, model_name: str) -> Tuple[str, str]:
    """
    调用VLM生成CoT标注
    
    Args:
        video_url: 视频URL
        prompt: Prompt文本
        llm_client: LLM客户端
        model_name: 模型名称
    
    Returns:
        (cot, text): 推理过程和输出文本
    """
    response = llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            }
        ],
        temperature=0.0,
        reasoning_effort="medium",
    )
    cot = response.choices[0].message.reasoning_content
    text = response.choices[0].message.content
    return cot, text


# ================================================================================
# 结果解析
# ================================================================================

def parse_reasoning_dict(text: str) -> Optional[Dict]:
    """
    解析LLM返回的JSON格式reasoning
    
    Args:
        text: LLM输出文本
    
    Returns:
        dict: 解析后的reasoning字典
    """
    try:
        s = text.strip()
        
        # 移除FINISHED标记
        if s.endswith("FINISHED"):
            s = s[:-8].strip()
        elif "FINISHED" in s:
            s = s.replace("FINISHED", "").strip()
        
        # 移除代码块标记
        if s.startswith("```python"):
            s = s[9:]
        elif s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        
        if s.endswith("```"):
            s = s[:-3]
        
        s = s.strip()
        
        d = json.loads(s)
        return d if isinstance(d, dict) else None
    except Exception as e:
        print(f"Warning: JSON parse failed: {e}")
        print(f"Text preview: {s[:200] if len(s) > 200 else s}...")
        try:
            d = eval(s)
            return d if isinstance(d, dict) else None
        except Exception as e2:
            print(f"Warning: eval also failed: {e2}")
            return None


# ================================================================================
# 主处理流程
# ================================================================================

def load_prompt_template(prompt_path: str) -> str:
    """加载Prompt模板"""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(
    template: str,
    task_description: str,
    num_frames: int,
    fps: int,
    primitives_text: str
) -> str:
    """
    构建完整的Prompt
    
    Args:
        template: Prompt模板
        task_description: 任务描述
        num_frames: 总帧数
        fps: 帧率
        primitives_text: 格式化的动作原语文本
    
    Returns:
        str: 完整的Prompt
    """
    duration = num_frames / fps
    
    prompt = template.replace("{task_description}", task_description)
    prompt = prompt.replace("{total_frames}", str(num_frames))
    prompt = prompt.replace("{fps}", str(fps))
    prompt = prompt.replace("{duration:.2f}", f"{duration:.2f}")
    prompt = prompt.replace("{duration}", str(duration))
    prompt = prompt.replace("{primitive_movements}", primitives_text)
    
    return prompt


def process_single_episode(
    primitives_json_path: str,
    output_dir: str,
    prompt_template: str,
    base_dir: str,
    fps: int = 15,
    sample_interval: int = 1,
    llm_client=None,
    model_name: str = None
) -> Optional[Dict]:
    """
    处理单个episode
    
    Args:
        primitives_json_path: 动作原语JSON路径
        output_dir: 输出目录
        prompt_template: Prompt模板
        base_dir: 数据集根目录
        fps: 帧率
        sample_interval: 原语采样间隔
        llm_client: LLM客户端
        model_name: 模型名称
    
    Returns:
        dict: 处理结果
    """
    # 加载动作原语
    primitives_data = load_primitives_from_json(primitives_json_path)
    if primitives_data is None:
        return None
    
    task_name = primitives_data["task_name"]
    episode_name = primitives_data["episode_name"]
    hdf5_path = primitives_data["hdf5_path"]
    file_name = f"{task_name}_{episode_name}"
    
    # 检查是否已处理
    json_path = os.path.join(output_dir, "json", f"{file_name}.json")
    if os.path.exists(json_path):
        print(f"⊙ Skipping {file_name} (already exists)")
        return None
    
    print(f"Processing {file_name}...")
    
    # 检查HDF5文件
    if not os.path.exists(hdf5_path):
        hdf5_path = os.path.join(base_dir, task_name, "aloha-agilex_clean_50", "data", f"{episode_name}.hdf5")
    
    if not os.path.exists(hdf5_path):
        print(f"Error: HDF5 file not found: {hdf5_path}")
        return None
    
    # 格式化动作原语
    primitives_text = format_primitives_for_prompt(primitives_data, fps=fps, sample_interval=sample_interval)
    
    # 读取任务描述
    instruction_path = os.path.join(base_dir, task_name, "aloha-agilex_clean_50", "instructions", f"{episode_name}.json")
    try:
        with open(instruction_path, "r", encoding="utf-8") as f:
            task_description = random.choice(json.load(f)['seen'])
    except Exception as e:
        print(f"Warning: Could not read instruction: {e}")
        task_description = task_name.replace("_", " ")
    
    # 提取视频帧
    frames = extract_frames_multi_view(hdf5_path)
    num_frames = len(frames)
    
    # 构建Prompt
    prompt = build_prompt(
        prompt_template,
        task_description,
        num_frames,
        fps,
        primitives_text
    )
    
    # 保存原始视频并上传
    os.makedirs(os.path.join(output_dir, "origin"), exist_ok=True)
    origin_video_path = os.path.join(output_dir, "origin", f"{file_name}.mp4")
    save_video_from_frames(frames, origin_video_path, fps=fps)
    
    # 如果没有LLM客户端，只保存视频和prompt
    if llm_client is None:
        print(f"  No LLM client, saving prompt only")
        os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
        prompt_save_path = os.path.join(output_dir, "prompts", f"{file_name}.txt")
        with open(prompt_save_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        return {
            "task_name": task_name,
            "episode_name": episode_name,
            "file_name": file_name,
            "prompt_path": prompt_save_path,
            "video_path": origin_video_path
        }
    
    # 上传视频
    video_url = upload_to_tos(origin_video_path)
    
    # 调用VLM
    print(f"  Calling VLM for {file_name}...")
    cot, text = call_vlm(video_url, prompt, llm_client, model_name)
    print(f"  VLM response received")
    
    # 解析结果
    reasoning = parse_reasoning_dict(text)
    
    # 保存结果
    os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
    result = {
        "task_name": task_name,
        "episode_name": episode_name,
        "hdf5_path": hdf5_path,
        "task_description": task_description,
        "num_frames": num_frames,
        "fps": fps,
        "origin_video": origin_video_path,
        "video_url": video_url,
        "cot": cot,
        "text": text,
        "reasoning": reasoning,
        "primitives": {
            "left": [p["primitive"] for p in primitives_data["left_primitives"]],
            "right": [p["primitive"] for p in primitives_data["right_primitives"]]
        }
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved JSON to {json_path}")
    
    return {
        "task_name": task_name,
        "episode_name": episode_name,
        "file_name": file_name,
        "json_path": json_path
    }


def main():
    parser = argparse.ArgumentParser(description="利用动作原语生成CoT标注")
    
    # 数据路径
    parser.add_argument("--base_dir", type=str,
                        default="/mnt/hdfs/_BYTE_DATA_SEED_/hl_lq/wangwenqian.wq/datasets/robotwin2_0816_raw",
                        help="数据集根目录")
    parser.add_argument("--primitives_dir", type=str,
                        default="/mnt/bn/ic-vlm/zhufangqi/embodied-CoT/results/clear_cot_label/primitives/json",
                        help="动作原语JSON目录")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/bn/ic-vlm/zhufangqi/embodied-CoT/results/clear_cot_label/cot",
                        help="输出目录")
    parser.add_argument("--prompt_file", type=str,
                        default="",
                        help="Prompt模板文件路径")
    
    # 处理参数
    parser.add_argument("--fps", type=int, default=15, help="视频帧率")
    parser.add_argument("--sample_interval", type=int, default=1, help="原语采样间隔")
    parser.add_argument("--max_workers", type=int, default=4, help="并行处理线程数")
    parser.add_argument("--task_name", type=str, default="", help="只处理指定task")
    parser.add_argument("--no_api", action="store_true", help="不调用API，只生成prompt")
    
    args = parser.parse_args()
    
    # 处理备用路径
    if not os.path.exists(args.base_dir):
        args.base_dir = "/mnt/hdfs/_BYTE_DATA_SEED_/hl_lq/iccv/user/wangwenqian.wq/datasets/robotwin2_0816_raw"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载Prompt模板
    if args.prompt_file:
        prompt_path = args.prompt_file
    else:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "bimanual_cot.txt")
    
    prompt_template = load_prompt_template(prompt_path)
    
    # 获取LLM客户端
    llm_client, model_name = None, None
    if not args.no_api:
        llm_client, model_name = get_llm_client()
        if llm_client is None:
            print("Warning: LLM client not available, running in no-api mode")
    
    # 获取所有动作原语JSON文件
    json_files = []
    for fn in os.listdir(args.primitives_dir):
        if fn.endswith('.json'):
            if args.task_name:
                if fn.startswith(args.task_name + "_"):
                    json_files.append(os.path.join(args.primitives_dir, fn))
            else:
                json_files.append(os.path.join(args.primitives_dir, fn))
    
    print(f"Found {len(json_files)} primitive files")
    
    results = []
    
    # 处理每个文件
    for json_path in json_files:
        try:
            result = process_single_episode(
                json_path,
                args.output_dir,
                prompt_template,
                args.base_dir,
                fps=args.fps,
                sample_interval=args.sample_interval,
                llm_client=llm_client,
                model_name=model_name
            )
            if result:
                results.append(result)
                print(f"✓ Completed: {result['file_name']}")
        except Exception as e:
            print(f"✗ Error processing {json_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存汇总
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成! 共处理 {len(results)} 个episodes")
    print(f"结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
