#!/usr/bin/env python3
"""
检查 JSON 文件中 reasoning 字段的 key 是否能被解析为浮点数。
如果不能解析，则删除该 JSON 文件以及对应的视频文件。
"""

import os
import json
import glob
from pathlib import Path

# 路径配置
JSON_DIR = "/mnt/bn/ic-vlm/zhufangqi/embodied-CoT/results/clear_cot_label/cot/json"
VIDEO_DIR = "/mnt/bn/ic-vlm/zhufangqi/embodied-CoT/results/clear_cot_label/cot/annotated"


def check_reasoning_keys(json_path: str) -> tuple[bool, list]:
    """
    检查 JSON 文件中 reasoning 字段的 key 是否都能被解析为浮点数。
    
    Returns:
        (is_valid, invalid_keys): 是否有效，无效的 key 列表
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"JSON解析错误: {e}"]
    except Exception as e:
        return False, [f"文件读取错误: {e}"]
    
    reasoning = data.get("reasoning", {})
    if not reasoning:
        # 没有 reasoning 字段或为空，认为是有效的
        return True, []
    
    invalid_keys = []
    for key in reasoning.keys():
        try:
            float(key)
        except ValueError:
            invalid_keys.append(key)
    
    return len(invalid_keys) == 0, invalid_keys


def find_corresponding_video(json_path: str) -> str:
    """
    根据 JSON 文件路径找到对应的视频文件路径。
    假设视频文件名与 JSON 文件名相同，只是扩展名不同。
    """
    json_name = Path(json_path).stem  # 获取不带扩展名的文件名
    
    # 尝试常见的视频扩展名
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    for ext in video_extensions:
        video_path = os.path.join(VIDEO_DIR, json_name + ext)
        if os.path.exists(video_path):
            return video_path
    
    return None


def main():
    # 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    invalid_files = []
    
    # 检查每个 JSON 文件
    for json_path in json_files:
        is_valid, invalid_keys = check_reasoning_keys(json_path)
        if not is_valid:
            invalid_files.append({
                'json_path': json_path,
                'invalid_keys': invalid_keys
            })
    
    print(f"\n发现 {len(invalid_files)} 个无效的 JSON 文件")
    
    if not invalid_files:
        print("所有文件都有效，无需删除。")
        return
    
    # 显示无效文件详情
    print("\n无效文件列表:")
    print("=" * 80)
    for item in invalid_files:
        print(f"\n文件: {item['json_path']}")
        print(f"无效的 key: {item['invalid_keys']}")
    
    # 确认删除
    print("\n" + "=" * 80)
    user_input = input(f"是否删除这 {len(invalid_files)} 个无效的 JSON 文件及其对应的视频? (yes/no): ")
    
    if user_input.lower() != 'yes':
        print("取消删除操作。")
        return
    
    # 执行删除
    deleted_json = 0
    deleted_video = 0
    
    for item in invalid_files:
        json_path = item['json_path']
        
        # 删除 JSON 文件
        try:
            os.remove(json_path)
            deleted_json += 1
            print(f"已删除 JSON: {json_path}")
        except Exception as e:
            print(f"删除 JSON 失败: {json_path}, 错误: {e}")
        
        # 删除对应的视频文件
        video_path = find_corresponding_video(json_path)
        if video_path:
            try:
                os.remove(video_path)
                deleted_video += 1
                print(f"已删除视频: {video_path}")
            except Exception as e:
                print(f"删除视频失败: {video_path}, 错误: {e}")
        else:
            print(f"未找到对应的视频文件: {Path(json_path).stem}")
    
    print(f"\n删除完成: {deleted_json} 个 JSON 文件, {deleted_video} 个视频文件")


if __name__ == "__main__":
    main()
