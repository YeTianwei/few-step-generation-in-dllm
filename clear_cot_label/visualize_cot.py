"""
CoT可视化模块
=============

将CoT标注结果可视化到视频上。
"""

import os
import io
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont


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
                    ImageFont.truetype(path, 18),
                    ImageFont.truetype(path, 14),
                    ImageFont.truetype(path, 11),
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


def extract_frames_multi_view(
    hdf5_path: str,
    cameras: List[str] = ["head_camera", "front_camera", "left_camera", "right_camera"]
) -> List[Image.Image]:
    """提取多视角视频帧"""
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


def visualize_cot(
    frames: List[Image.Image],
    reasoning_dict: Optional[Dict],
    primitives_data: Dict,
    output_path: str,
    fps: int = 15,
    target_img_w: int = 900,
    text_panel_w: int = 900
) -> str:
    """
    可视化CoT标注到视频
    
    Args:
        frames: 视频帧列表
        reasoning_dict: CoT推理字典
        primitives_data: 动作原语数据
        output_path: 输出视频路径
        fps: 帧率
        target_img_w: 目标图像宽度
        text_panel_w: 文本面板宽度
    
    Returns:
        str: 输出视频路径
    """
    font_title, font_label, font_text = ensure_fonts()
    
    # 计算画布尺寸
    if frames:
        first_frame = frames[0]
        orig_w, orig_h = first_frame.size
        target_img_h = int(target_img_w * orig_h / orig_w)
    else:
        target_img_h = 675
    
    canvas_h = max(target_img_h, 800)
    canvas_w = target_img_w + text_panel_w
    
    temp_canvas = Image.new('RGB', (text_panel_w, 3000), (255, 255, 255))
    temp_draw = ImageDraw.Draw(temp_canvas)
    text_width = text_panel_w - 30
    line_h = 14
    
    left_primitives = primitives_data.get("left", [])
    right_primitives = primitives_data.get("right", [])
    
    # 预处理reasoning_dict的keys
    frame_to_reasoning = {}
    if reasoning_dict:
        # 直接遍历原始keys，保留原始字符串格式
        original_keys = list(reasoning_dict.keys())
        sorted_keys = sorted(original_keys, key=lambda x: float(x))
        for str_key in sorted_keys:
            k = float(str_key)
            frame_idx = int(round(k * fps))
            frame_to_reasoning[frame_idx] = reasoning_dict[str_key]
    
    def pick_reasoning(idx: int) -> Optional[Dict]:
        """找到当前帧对应的reasoning"""
        if not frame_to_reasoning:
            return None
        keys = sorted(frame_to_reasoning.keys())
        chosen = None
        for k in keys:
            if k <= idx:
                chosen = k
            else:
                break
        return frame_to_reasoning.get(chosen) if chosen is not None else (
            frame_to_reasoning.get(keys[0]) if keys else None
        )
    
    def get_current_move_index(idx: int, moves: List[Dict]) -> int:
        """找到当前帧对应的move索引"""
        if not moves:
            return -1
        current_idx = -1
        for i, move in enumerate(moves):
            move_time = move.get('time', 0)
            move_frame = int(round(move_time * fps))
            if move_frame <= idx:
                current_idx = i
        return current_idx
    
    out_frames = []
    
    def render_panel(idx: int, active_data: Optional[Dict]) -> Image.Image:
        panel = Image.new('RGB', (text_panel_w, canvas_h), (245, 245, 245))
        d = ImageDraw.Draw(panel)
        y = 8
        
        # 帧信息
        d.text((10, y), f"Frame {idx} | Time: {idx/fps:.2f}s", fill=(0, 0, 128), font=font_title)
        y += 22
        
        # 动作原语
        left_prim = left_primitives[idx] if idx < len(left_primitives) else "N/A"
        right_prim = right_primitives[idx] if idx < len(right_primitives) else "N/A"
        
        if isinstance(left_prim, dict):
            left_prim = left_prim.get("primitive", "N/A")
        if isinstance(right_prim, dict):
            right_prim = right_prim.get("primitive", "N/A")
        
        d.text((10, y), f"L: {left_prim}", fill=(0, 100, 0), font=font_text)
        y += line_h
        d.text((10, y), f"R: {right_prim}", fill=(100, 0, 0), font=font_text)
        y += line_h + 5
        
        # 分隔线
        d.line([(10, y), (text_panel_w - 10, y)], fill=(180, 180, 180), width=1)
        y += 8
        
        if active_data:
            # Task
            d.text((10, y), "Task:", fill=(0, 80, 0), font=font_label)
            y += line_h + 2
            for l in wrap_text(active_data.get('task', ''), text_width, font_text, temp_draw):
                d.text((15, y), l, fill=(0, 0, 0), font=font_text)
                y += line_h
            y += 4
            
            # Plan
            plan = active_data.get('plan', [])
            if plan:
                d.text((10, y), "Plan:", fill=(0, 80, 0), font=font_label)
                y += line_h + 2
                for i, step in enumerate(plan):
                    step_text = f"{i+1}. {step}"
                    for l in wrap_text(step_text, text_width - 10, font_text, temp_draw):
                        d.text((20, y), l, fill=(60, 60, 60), font=font_text)
                        y += line_h
                y += 4
            
            # Subtask
            d.text((10, y), "Subtask:", fill=(0, 80, 0), font=font_label)
            y += line_h + 2
            for l in wrap_text(active_data.get('subtask', ''), text_width, font_text, temp_draw):
                d.text((15, y), l, fill=(0, 0, 0), font=font_text)
                y += line_h
            y += 4
            
            # Subtask Reason
            d.text((10, y), "Subtask Reason:", fill=(0, 80, 0), font=font_label)
            y += line_h + 2
            for l in wrap_text(active_data.get('subtask_reason', ''), text_width, font_text, temp_draw):
                d.text((15, y), l, fill=(50, 50, 50), font=font_text)
                y += line_h
            y += 6
            
            # 分隔线
            d.line([(10, y), (text_panel_w - 10, y)], fill=(180, 180, 180), width=1)
            y += 6
            
            # Moves
            moves = active_data.get('moves', [])
            current_move_idx = get_current_move_index(idx, moves)
            
            if moves:
                d.text((10, y), f"Moves ({len(moves)}):", fill=(150, 0, 0), font=font_label)
                y += line_h + 2
                
                for i, move in enumerate(moves):
                    is_current = (i == current_move_idx)
                    
                    # 高亮当前move
                    if is_current:
                        move_text = move.get('move', '')
                        move_reason = move.get('move_reason', '')
                        arm = move.get('arm', '')
                        move_lines = wrap_text(f"[{arm}] {move_text}", text_width - 20, font_text, temp_draw)
                        reason_lines = wrap_text(f"  {move_reason}", text_width - 20, font_text, temp_draw) if move_reason else []
                        total_height = (len(move_lines) + len(reason_lines) + 1) * line_h + 6
                        
                        d.rectangle([(5, y - 2), (text_panel_w - 5, y + total_height - 2)],
                                   fill=(255, 255, 200), outline=(200, 180, 0))
                    
                    # Move内容
                    move_time = move.get('time', 0)
                    move_text = move.get('move', '')
                    arm = move.get('arm', '')
                    prefix = "→ " if is_current else "  "
                    time_color = (180, 0, 0) if is_current else (100, 100, 100)
                    text_color = (0, 0, 0) if is_current else (60, 60, 60)
                    
                    d.text((15, y), f"{prefix}[{move_time:.2f}s] [{arm}]", fill=time_color, font=font_text)
                    y += line_h
                    
                    for l in wrap_text(move_text, text_width - 30, font_text, temp_draw):
                        d.text((25, y), l, fill=text_color, font=font_text)
                        y += line_h
                    
                    # Move Reason
                    move_reason = move.get('move_reason', '')
                    if move_reason:
                        reason_color = (80, 80, 80) if is_current else (120, 120, 120)
                        for l in wrap_text(f"Reason: {move_reason}", text_width - 30, font_text, temp_draw):
                            d.text((25, y), l, fill=reason_color, font=font_text)
                            y += line_h
                    
                    y += 4
        
        return panel
    
    for idx, frm in enumerate(frames):
        # 缩放图像
        pil_img = frm.resize((target_img_w, target_img_h), Image.Resampling.BILINEAR).convert('RGB')
        
        canvas = Image.new('RGB', (canvas_w, canvas_h), (40, 40, 40))
        canvas.paste(pil_img, (0, 0))
        
        # 在视频上绘制信息
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([(0, target_img_h - 28), (target_img_w, target_img_h)], fill=(0, 0, 0))
        draw.text((10, target_img_h - 24), f"Frame {idx} | Time: {idx/fps:.2f}s", fill=(255, 255, 0), font=font_title)
        
        active_data = pick_reasoning(idx)
        panel_img = render_panel(idx, active_data)
        canvas.paste(panel_img, (target_img_w, 0))
        
        out_frames.append(np.asarray(canvas))
    
    iio.imwrite(output_path, out_frames, fps=float(fps), codec="libx264", macro_block_size=1)
    return output_path


def visualize_from_json(
    cot_json_path: str,
    output_path: str,
    fps: int = 15
) -> str:
    """
    从CoT JSON文件创建可视化视频
    
    Args:
        cot_json_path: CoT JSON文件路径
        output_path: 输出视频路径
        fps: 帧率
    
    Returns:
        str: 输出视频路径
    """
    with open(cot_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    hdf5_path = data["hdf5_path"]
    reasoning = data.get("reasoning", {})
    primitives = data.get("primitives", {"left": [], "right": []})
    
    frames = extract_frames_multi_view(hdf5_path)
    
    return visualize_cot(
        frames,
        reasoning,
        primitives,
        output_path,
        fps=fps
    )


def main():
    parser = argparse.ArgumentParser(description="可视化CoT标注结果")
    
    parser.add_argument("--cot_dir", type=str,
                        default="/mnt/bn/ic-vlm/zhufangqi/embodied-CoT/results/clear_cot_label/cot/json",
                        help="CoT JSON文件目录")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/bn/ic-vlm/zhufangqi/embodied-CoT/results/clear_cot_label/cot/annotated",
                        help="输出视频目录")
    parser.add_argument("--fps", type=int, default=15, help="帧率")
    parser.add_argument("--task_name", type=str, default="", help="只处理指定task")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = []
    for fn in os.listdir(args.cot_dir):
        if fn.endswith('.json'):
            if args.task_name:
                if fn.startswith(args.task_name + "_"):
                    json_files.append(os.path.join(args.cot_dir, fn))
            else:
                json_files.append(os.path.join(args.cot_dir, fn))
    
    print(f"Found {len(json_files)} CoT JSON files")
    
    for json_path in json_files:
        try:
            file_name = os.path.splitext(os.path.basename(json_path))[0]
            output_path = os.path.join(args.output_dir, f"{file_name}.mp4")
            
            if os.path.exists(output_path):
                print(f"⊙ Skipping {file_name} (already exists)")
                continue
            
            print(f"Processing {file_name}...")
            visualize_from_json(json_path, output_path, fps=args.fps)
            print(f"  Saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n可视化完成!")


if __name__ == "__main__":
    main()
