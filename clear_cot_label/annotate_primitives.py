"""
动作原语标注模块
================

从HDF5文件中提取双臂机器人的动作原语，并生成可视化视频。

输入:
- HDF5文件（包含endpose, gripper等数据）

输出:
- JSON格式的动作原语标注
- 可视化视频（原语叠加在视频上）
"""

import os
import io
import json
import argparse
import numpy as np
import h5py
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .primitives import extract_primitives_from_trajectory, quat_to_euler


def extract_arm_data_from_hdf5(hdf5_path: str) -> Dict:
    """
    从HDF5文件中提取双臂数据
    
    Args:
        hdf5_path: HDF5文件路径
    
    Returns:
        dict: 包含左右臂的位置、四元数、gripper数据
    """
    with h5py.File(hdf5_path, "r") as f:
        # 左臂数据
        left_endpose = f["endpose"]["left_endpose"][:]  # (N, 7): xyz + quat(wxyz)
        left_gripper = f["endpose"]["left_gripper"][:]  # (N,)
        
        # 右臂数据
        right_endpose = f["endpose"]["right_endpose"][:]  # (N, 7)
        right_gripper = f["endpose"]["right_gripper"][:]  # (N,)
        
        num_steps = len(left_endpose)
        
        # 解析左臂: endpose格式为 [x, y, z, qw, qx, qy, qz] 或 [x, y, z, qx, qy, qz, qw]
        # 需要确认四元数格式，假设是 [x, y, z, qw, qx, qy, qz]
        left_positions = left_endpose[:, :3]
        left_orientations = left_endpose[:, 3:7]  # 四元数 wxyz
        
        # 解析右臂
        right_positions = right_endpose[:, :3]
        right_orientations = right_endpose[:, 3:7]
        
    return {
        "left": {
            "positions": left_positions,
            "orientations": left_orientations,
            "grippers": left_gripper
        },
        "right": {
            "positions": right_positions,
            "orientations": right_orientations,
            "grippers": right_gripper
        },
        "num_steps": num_steps
    }


def extract_primitives_from_hdf5(
    hdf5_path: str,
    window_size: int = 4,
    pos_threshold: float = 0.02,
    rot_threshold: float = 0.05,
    gripper_threshold: float = 0.05
) -> Dict:
    """
    从HDF5文件中提取左右臂的动作原语
    
    Args:
        hdf5_path: HDF5文件路径
        window_size: 滑动窗口大小
        pos_threshold: 位置阈值
        rot_threshold: 旋转阈值
        gripper_threshold: gripper阈值
    
    Returns:
        dict: 包含左右臂原语和步数
    """
    arm_data = extract_arm_data_from_hdf5(hdf5_path)
    
    # 提取左臂原语
    left_primitives = extract_primitives_from_trajectory(
        positions=arm_data["left"]["positions"],
        orientations=arm_data["left"]["orientations"],
        grippers=arm_data["left"]["grippers"],
        window_size=window_size,
        pos_threshold=pos_threshold,
        rot_threshold=rot_threshold,
        gripper_threshold=gripper_threshold
    )
    
    # 提取右臂原语
    right_primitives = extract_primitives_from_trajectory(
        positions=arm_data["right"]["positions"],
        orientations=arm_data["right"]["orientations"],
        grippers=arm_data["right"]["grippers"],
        window_size=window_size,
        pos_threshold=pos_threshold,
        rot_threshold=rot_threshold,
        gripper_threshold=gripper_threshold
    )
    
    return {
        "left": left_primitives,
        "right": right_primitives,
        "num_steps": arm_data["num_steps"]
    }


def extract_frames_multi_view(
    hdf5_path: str,
    cameras: List[str] = ["head_camera", "front_camera", "left_camera", "right_camera"]
) -> List[Image.Image]:
    """
    提取四个摄像头的帧并组合成2x2网格
    
    Args:
        hdf5_path: HDF5文件路径
        cameras: 摄像头列表
    
    Returns:
        List[Image.Image]: 组合后的帧列表
    """
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
                    # BGR -> RGB
                    img_array_rgb = img_rgb[:, :, ::-1]
                    img = Image.fromarray(img_array_rgb)
                    frames.append(img)
                camera_frames[camera] = frames
    
    if not camera_frames:
        return []
    
    # 确保所有摄像头都有相同数量的帧
    num_frames = min(len(frames) for frames in camera_frames.values())
    
    # 组合成2x2网格
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
        
        # 添加摄像头标签
        draw = ImageDraw.Draw(grid)
        font = ImageFont.load_default()
        draw.text((10, 10), "Head", fill=(255, 255, 0), font=font)
        draw.text((w + 10, 10), "Front", fill=(255, 255, 0), font=font)
        draw.text((10, h + 10), "Left", fill=(255, 255, 0), font=font)
        draw.text((w + 10, h + 10), "Right", fill=(255, 255, 0), font=font)
        
        combined_frames.append(grid)
    
    return combined_frames


def ensure_fonts() -> Tuple:
    """加载字体"""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    candidates = [
        os.path.join(repo_root, "NotoSansCJKsc-Regular.otf"),
        os.path.join(repo_root, "assets", "fonts", "NotoSansCJKsc-Regular.otf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return (
                    ImageFont.truetype(path, 20),
                    ImageFont.truetype(path, 16),
                    ImageFont.truetype(path, 12),
                )
            except Exception:
                pass
    return (ImageFont.load_default(), ImageFont.load_default(), ImageFont.load_default())


def wrap_text(text: str, max_width: int, font, draw) -> List[str]:
    """文本换行"""
    if not text:
        return []
    if ' ' in text:
        words = text.split(' ')
        lines = []
        current = []
        for w in words:
            test = ' '.join(current + [w])
            bbox = draw.textbbox((0, 0), test, font=font)
            width = bbox[2] - bbox[0]
            if width <= max_width:
                current.append(w)
            else:
                if current:
                    lines.append(' '.join(current))
                    current = [w]
                else:
                    lines.append(w)
        if current:
            lines.append(' '.join(current))
        return lines
    # 无空格，按字符换行
    lines = []
    cur = ''
    for ch in text:
        test = cur + ch
        bbox = draw.textbbox((0, 0), test, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
                cur = ch
            else:
                lines.append(ch)
                cur = ''
    if cur:
        lines.append(cur)
    return lines


def visualize_primitives(
    frames: List[Image.Image],
    primitives_data: Dict,
    output_path: str,
    fps: int = 15,
    target_img_w: int = 640,
    target_img_h: int = 560,
    text_panel_w: int = 500
) -> str:
    """
    可视化动作原语到视频
    
    Args:
        frames: 视频帧列表
        primitives_data: 动作原语数据
        output_path: 输出视频路径
        fps: 帧率
        target_img_w: 目标图像宽度
        target_img_h: 目标图像高度
        text_panel_w: 文本面板宽度
    
    Returns:
        str: 输出视频路径
    """
    font_title, font_label, font_text = ensure_fonts()
    canvas_w = target_img_w + text_panel_w
    canvas_h = target_img_h
    
    temp_canvas = Image.new('RGB', (text_panel_w, 2000), (255, 255, 255))
    temp_draw = ImageDraw.Draw(temp_canvas)
    text_width = text_panel_w - 20
    line_h = 18
    
    left_primitives = primitives_data["left"]
    right_primitives = primitives_data["right"]
    
    out_frames = []
    
    for idx, frm in enumerate(frames):
        # 缩放图像
        img_w, img_h = frm.size
        img_aspect = img_w / img_h
        target_aspect = target_img_w / target_img_h
        if img_aspect > target_aspect:
            new_w = target_img_w
            new_h = int(target_img_w / img_aspect)
        else:
            new_h = target_img_h
            new_w = int(target_img_h * img_aspect)
        pil_img = frm.resize((new_w, new_h), Image.Resampling.BILINEAR).convert('RGB')
        
        # 创建画布
        canvas = Image.new('RGB', (canvas_w, canvas_h), (30, 30, 30))
        img_x = (target_img_w - new_w) // 2
        img_y = (target_img_h - new_h) // 2
        canvas.paste(pil_img, (img_x, img_y))
        
        # 在视频上绘制帧号
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"Frame {idx} | Time: {idx/fps:.2f}s", fill=(255, 255, 0), font=font_title)
        
        # 绘制文本面板
        panel = Image.new('RGB', (text_panel_w, canvas_h), (245, 245, 245))
        panel_draw = ImageDraw.Draw(panel)
        
        y = 10
        
        # 标题
        panel_draw.text((10, y), "Primitive Movements", fill=(0, 0, 128), font=font_title)
        y += 28
        
        # 分隔线
        panel_draw.line([(10, y), (text_panel_w - 10, y)], fill=(180, 180, 180), width=1)
        y += 10
        
        # 左臂原语
        panel_draw.text((10, y), "Left Arm:", fill=(0, 100, 0), font=font_label)
        y += line_h + 2
        
        if idx < len(left_primitives):
            left_prim = left_primitives[idx]
            prim_text = left_prim["primitive"]
            for line in wrap_text(prim_text, text_width - 10, font_text, temp_draw):
                if y + line_h > canvas_h - 50:
                    break
                panel_draw.text((15, y), line, fill=(0, 0, 0), font=font_text)
                y += line_h
            
            # 显示位置信息
            y += 5
            pos = left_prim.get("position", [0, 0, 0])
            pos_text = f"Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
            panel_draw.text((15, y), pos_text, fill=(100, 100, 100), font=font_text)
            y += line_h
            
            # 显示欧拉角（弧度）
            orient = left_prim.get("orientation", [1, 0, 0, 0])  # wxyz
            roll, pitch, yaw = quat_to_euler(orient[0], orient[1], orient[2], orient[3])
            euler_text = f"Euler: [R:{roll:.3f}, P:{pitch:.3f}, Y:{yaw:.3f}] rad"
            panel_draw.text((15, y), euler_text, fill=(100, 100, 100), font=font_text)
            y += line_h
            
            # 显示运动向量中的旋转变化
            move_vec = left_prim.get("move_vec", [0, 0, 0, 0, 0, 0, 0])
            rot_names = ["roll", "pitch", "yaw"]
            rot_changes = []
            for i, name in enumerate(rot_names):
                val = move_vec[3 + i]
                if val == 1:
                    rot_changes.append(f"{name}↑")
                elif val == -1:
                    rot_changes.append(f"{name}↓")
            if rot_changes:
                rot_text = f"RotΔ: {', '.join(rot_changes)}"
                panel_draw.text((15, y), rot_text, fill=(150, 80, 0), font=font_text)
                y += line_h
        
        y += 15
        
        # 分隔线
        panel_draw.line([(10, y), (text_panel_w - 10, y)], fill=(180, 180, 180), width=1)
        y += 10
        
        # 右臂原语
        panel_draw.text((10, y), "Right Arm:", fill=(100, 0, 0), font=font_label)
        y += line_h + 2
        
        if idx < len(right_primitives):
            right_prim = right_primitives[idx]
            prim_text = right_prim["primitive"]
            for line in wrap_text(prim_text, text_width - 10, font_text, temp_draw):
                if y + line_h > canvas_h - 50:
                    break
                panel_draw.text((15, y), line, fill=(0, 0, 0), font=font_text)
                y += line_h
            
            # 显示位置信息
            y += 5
            pos = right_prim.get("position", [0, 0, 0])
            pos_text = f"Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
            panel_draw.text((15, y), pos_text, fill=(100, 100, 100), font=font_text)
            y += line_h
            
            # 显示欧拉角（弧度）
            orient = right_prim.get("orientation", [1, 0, 0, 0])  # wxyz
            roll, pitch, yaw = quat_to_euler(orient[0], orient[1], orient[2], orient[3])
            euler_text = f"Euler: [R:{roll:.3f}, P:{pitch:.3f}, Y:{yaw:.3f}] rad"
            panel_draw.text((15, y), euler_text, fill=(100, 100, 100), font=font_text)
            y += line_h
            
            # 显示运动向量中的旋转变化
            move_vec = right_prim.get("move_vec", [0, 0, 0, 0, 0, 0, 0])
            rot_names = ["roll", "pitch", "yaw"]
            rot_changes = []
            for i, name in enumerate(rot_names):
                val = move_vec[3 + i]
                if val == 1:
                    rot_changes.append(f"{name}↑")
                elif val == -1:
                    rot_changes.append(f"{name}↓")
            if rot_changes:
                rot_text = f"RotΔ: {', '.join(rot_changes)}"
                panel_draw.text((15, y), rot_text, fill=(150, 80, 0), font=font_text)
                y += line_h
        
        # 粘贴面板
        canvas.paste(panel, (target_img_w, 0))
        out_frames.append(np.asarray(canvas))
    
    # 保存视频
    iio.imwrite(output_path, out_frames, fps=float(fps), codec="libx264", macro_block_size=1)
    return output_path


def group_hdf5_by_task(base_dir: str) -> Dict[str, List[str]]:
    """按task分组hdf5文件
    
    基于已知的目录结构直接查找：
    {base_dir}/{task_name}/aloha-agilex_clean_50/data/episode{N}.hdf5
    """
    task_map = {}
    
    print(f"Scanning directory: {base_dir}", flush=True)
    
    # 列出顶层目录获取所有 task
    try:
        task_dirs = os.listdir(base_dir)
    except Exception as e:
        print(f"  Error listing base directory: {e}", flush=True)
        return task_map
    
    for task_name in task_dirs:
        # 跳过非目录项和特殊目录
        task_path = os.path.join(base_dir, task_name)
        if not os.path.isdir(task_path):
            continue
        if task_name.startswith('.') or task_name in ['dataset', 'aloha-agilex']:
            continue
        
        # 构建 data 目录路径：{task}/aloha-agilex_clean_50/data/
        data_dir = os.path.join(task_path, "aloha-agilex_clean_50", "data")
        
        if not os.path.exists(data_dir):
            continue
        
        # 列出 data 目录中的 hdf5 文件
        try:
            files = os.listdir(data_dir)
            hdf5_files = [
                os.path.join(data_dir, fn)
                for fn in files
                if fn.endswith('.hdf5')
            ]
            if hdf5_files:
                task_map[task_name] = sorted(hdf5_files)
        except Exception as e:
            print(f"  Warning: Error listing {data_dir}: {e}", flush=True)
    
    print(f"  Found {len(task_map)} tasks", flush=True)
    return task_map


def process_episode(
    hdf5_path: str,
    output_dir: str,
    fps: int = 15,
    window_size: int = 4,
    pos_threshold: float = 0.02,
    rot_threshold: float = 0.05,
    gripper_threshold: float = 0.05,
    save_video: bool = True
) -> Optional[Dict]:
    """
    处理单个episode
    
    Args:
        hdf5_path: HDF5文件路径
        output_dir: 输出目录
        fps: 帧率
        window_size: 滑动窗口大小
        pos_threshold: 位置阈值
        rot_threshold: 旋转阈值
        gripper_threshold: gripper阈值
        save_video: 是否保存可视化视频
    
    Returns:
        dict: 处理结果
    """
    # 提取task名称和episode名称
    parts = hdf5_path.split('/')
    task_name = parts[-4]
    episode_name = os.path.splitext(parts[-1])[0]
    file_name = f"{task_name}_{episode_name}"
    
    # 检查是否已经处理过
    json_path = os.path.join(output_dir, "json", f"{file_name}.json")
    video_path = os.path.join(output_dir, "videos", f"{file_name}.mp4")
    
    if os.path.exists(json_path):
        if not save_video or os.path.exists(video_path):
            print(f"⊙ Skipping {file_name} (already exists)")
            with open(json_path, 'r') as f:
                data = json.load(f)
            return {
                "task_name": task_name,
                "episode_name": episode_name,
                "hdf5_path": hdf5_path,
                "json_path": json_path,
                "video_path": video_path if save_video else None,
                "num_steps": data["num_steps"]
            }
    
    print(f"Processing {file_name}...")
    
    try:
        # 提取动作原语
        primitives_data = extract_primitives_from_hdf5(
            hdf5_path,
            window_size=window_size,
            pos_threshold=pos_threshold,
            rot_threshold=rot_threshold,
            gripper_threshold=gripper_threshold
        )
        
        # 保存JSON
        json_dir = os.path.join(output_dir, "json")
        os.makedirs(json_dir, exist_ok=True)
        
        # 简化JSON输出（只保留primitive字符串）
        json_output = {
            "hdf5_path": hdf5_path,
            "task_name": task_name,
            "episode_name": episode_name,
            "num_steps": primitives_data["num_steps"],
            "left_primitives": primitives_data["left"],
            "right_primitives": primitives_data["right"],
            "params": {
                "window_size": window_size,
                "pos_threshold": pos_threshold,
                "rot_threshold": rot_threshold,
                "gripper_threshold": gripper_threshold
            }
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        
        print(f"  Saved JSON to {json_path}")
        
        # 可视化
        if save_video:
            frames = extract_frames_multi_view(hdf5_path)
            
            video_dir = os.path.join(output_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            
            visualize_primitives(frames, primitives_data, video_path, fps=fps)
            print(f"  Saved video to {video_path}")
        
        return {
            "task_name": task_name,
            "episode_name": episode_name,
            "hdf5_path": hdf5_path,
            "json_path": json_path,
            "video_path": video_path if save_video else None,
            "num_steps": primitives_data["num_steps"]
        }
        
    except Exception as e:
        print(f"  Error processing {file_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="标注双臂机器人的动作原语")
    
    # 数据路径
    parser.add_argument("--base_dir", type=str,
                        default="/mnt/hdfs/_BYTE_DATA_SEED_/hl_lq/wangwenqian.wq/datasets/robotwin2_0816_raw",
                        help="数据集根目录")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/bn/ic-vlm/zhufangqi/embodied-CoT/results/clear_cot_label/primitives",
                        help="输出目录")
    
    # 处理参数
    parser.add_argument("--fps", type=int, default=15, help="视频帧率")
    parser.add_argument("--window_size", type=int, default=4, help="滑动窗口大小")
    parser.add_argument("--pos_threshold", type=float, default=0.02, help="位置阈值（米）")
    parser.add_argument("--rot_threshold", type=float, default=0.05, help="旋转阈值（弧度）")
    parser.add_argument("--gripper_threshold", type=float, default=0.05, help="gripper阈值")
    
    # 控制选项
    parser.add_argument("--first_episode_only", action="store_true", default=True,
                        help="只处理每个task的第一个episode")
    parser.add_argument("--max_workers", type=int, default=16, help="并行处理的线程数")
    parser.add_argument("--no_video", action="store_true", help="不生成可视化视频")
    parser.add_argument("--task_name", type=str, default="", help="只处理指定的task")
    
    args = parser.parse_args()
    
    # 处理备用路径
    if not os.path.exists(args.base_dir):
        args.base_dir = "/mnt/hdfs/_BYTE_DATA_SEED_/hl_lq/iccv/user/wangwenqian.wq/datasets/robotwin2_0816_raw"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 按task分组
    task_map = group_hdf5_by_task(args.base_dir)
    print(f"Found {len(task_map)} tasks")
    
    # 收集要处理的hdf5文件
    hdf5_paths = []
    for task_name, hdf5_files in task_map.items():
        if args.task_name and task_name != args.task_name:
            continue
        if args.first_episode_only:
            hdf5_paths.append(hdf5_files[0])
        else:
            hdf5_paths.extend(hdf5_files)
    
    print(f"Processing {len(hdf5_paths)} episodes with {args.max_workers} workers")
    
    results = []
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_path = {
            executor.submit(
                process_episode,
                hdf5_path,
                args.output_dir,
                fps=args.fps,
                window_size=args.window_size,
                pos_threshold=args.pos_threshold,
                rot_threshold=args.rot_threshold,
                gripper_threshold=args.gripper_threshold,
                save_video=not args.no_video
            ): hdf5_path
            for hdf5_path in hdf5_paths
        }
        
        for future in as_completed(future_to_path):
            hdf5_path = future_to_path[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"✓ Completed {len(results)}/{len(hdf5_paths)}: {result['task_name']}")
            except Exception as e:
                print(f"✗ Error processing {hdf5_path}: {e}")
    
    # 保存汇总结果
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成! 共处理 {len(results)} 个episodes")
    print(f"结果保存在: {args.output_dir}")
    print(f"汇总文件: {summary_path}")


if __name__ == "__main__":
    main()
