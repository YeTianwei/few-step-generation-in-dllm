"""
Clear CoT Label 主流程
======================

两阶段流程：
1. 动作原语标注：HDF5 → primitives.json + primitives_video.mp4
2. CoT生成：primitives.json + video → cot.json + annotated_video.mp4

使用方法:
    # 只标注动作原语
    python main.py --stage primitives --base_dir ... --output_dir ...
    
    # 只生成CoT（需要已有动作原语）
    python main.py --stage cot --primitives_dir ... --output_dir ...
    
    # 完整流程
    python main.py --stage all --base_dir ... --output_dir ...
"""

import os
import argparse
import json
from typing import Optional


def run_primitives_stage(args) -> bool:
    """运行动作原语标注阶段"""
    from .annotate_primitives import (
        group_hdf5_by_task,
        process_episode
    )
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print("=" * 60)
    print("Stage 1: Annotating Primitive Movements")
    print("=" * 60)
    
    # 处理备用路径
    base_dir = args.base_dir
    if not os.path.exists(base_dir):
        base_dir = "/mnt/hdfs/_BYTE_DATA_SEED_/hl_lq/iccv/user/wangwenqian.wq/datasets/robotwin2_0816_raw"
    
    output_dir = os.path.join(args.output_dir, "primitives")
    os.makedirs(output_dir, exist_ok=True)
    
    # 按task分组
    task_map = group_hdf5_by_task(base_dir)
    print(f"Found {len(task_map)} tasks")
    
    # 收集要处理的文件
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
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_path = {
            executor.submit(
                process_episode,
                hdf5_path,
                output_dir,
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
    
    # 保存汇总
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nPrimitives stage completed! Processed {len(results)} episodes")
    print(f"Results saved to: {output_dir}")
    
    return len(results) > 0


def run_cot_stage(args) -> bool:
    """运行CoT生成阶段"""
    from .generate_cot import (
        load_prompt_template,
        load_primitives_from_json,
        process_single_episode,
        get_llm_client
    )
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print("=" * 60)
    print("Stage 2: Generating Chain of Thought")
    print("=" * 60)
    
    # 处理备用路径
    base_dir = args.base_dir
    if not os.path.exists(base_dir):
        base_dir = "/mnt/hdfs/_BYTE_DATA_SEED_/hl_lq/iccv/user/wangwenqian.wq/datasets/robotwin2_0816_raw"
    
    # 确定primitives目录
    if args.primitives_dir:
        primitives_dir = args.primitives_dir
    else:
        primitives_dir = os.path.join(args.output_dir, "primitives", "json")
    
    if not os.path.exists(primitives_dir):
        print(f"Error: Primitives directory not found: {primitives_dir}")
        print("Please run primitives stage first or specify --primitives_dir")
        return False
    
    output_dir = os.path.join(args.output_dir, "cot")
    os.makedirs(output_dir, exist_ok=True)
    
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
    for fn in os.listdir(primitives_dir):
        if fn.endswith('.json') and fn != 'summary.json':
            if args.task_name:
                if fn.startswith(args.task_name + "_"):
                    json_files.append(os.path.join(primitives_dir, fn))
            else:
                json_files.append(os.path.join(primitives_dir, fn))
    
    print(f"Found {len(json_files)} primitive files")
    print(f"Processing with {args.cot_workers} workers")
    
    results = []
    
    with ThreadPoolExecutor(max_workers=args.cot_workers) as executor:
        future_to_path = {
            executor.submit(
                process_single_episode,
                json_path,
                output_dir,
                prompt_template,
                base_dir,
                fps=args.fps,
                sample_interval=args.sample_interval,
                llm_client=llm_client,
                model_name=model_name
            ): json_path
            for json_path in json_files
        }
        
        for future in as_completed(future_to_path):
            json_path = future_to_path[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"✓ Completed {len(results)}/{len(json_files)}: {result['file_name']}")
            except Exception as e:
                print(f"✗ Error processing {json_path}: {e}")
                import traceback
                traceback.print_exc()
    
    # 保存汇总
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nCoT stage completed! Processed {len(results)} episodes")
    print(f"Results saved to: {output_dir}")
    
    return len(results) > 0


def run_visualize_stage(args) -> bool:
    """运行可视化阶段"""
    from .visualize_cot import visualize_from_json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print("=" * 60)
    print("Stage 3: Visualizing CoT Annotations")
    print("=" * 60)
    
    cot_dir = os.path.join(args.output_dir, "cot", "json")
    output_dir = os.path.join(args.output_dir, "cot", "annotated")
    
    if not os.path.exists(cot_dir):
        print(f"Error: CoT directory not found: {cot_dir}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = []
    for fn in os.listdir(cot_dir):
        if fn.endswith('.json') and fn != 'summary.json':
            if args.task_name:
                if fn.startswith(args.task_name + "_"):
                    json_files.append(os.path.join(cot_dir, fn))
            else:
                json_files.append(os.path.join(cot_dir, fn))
    
    # 过滤掉已存在的文件
    tasks_to_process = []
    for json_path in json_files:
        file_name = os.path.splitext(os.path.basename(json_path))[0]
        output_path = os.path.join(output_dir, f"{file_name}.mp4")
        if os.path.exists(output_path):
            print(f"⊙ Skipping {file_name} (already exists)")
        else:
            tasks_to_process.append((json_path, output_path, file_name))
    
    print(f"Found {len(json_files)} CoT JSON files, {len(tasks_to_process)} to process")
    print(f"Processing with {args.visualize_workers} workers")
    
    if not tasks_to_process:
        print("No new files to process")
        return True
    
    def process_visualization(task_info):
        json_path, output_path, file_name = task_info
        visualize_from_json(json_path, output_path, fps=args.fps)
        return file_name
    
    count = 0
    with ThreadPoolExecutor(max_workers=args.visualize_workers) as executor:
        future_to_task = {
            executor.submit(process_visualization, task): task
            for task in tasks_to_process
        }
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            json_path, output_path, file_name = task
            try:
                result = future.result()
                count += 1
                print(f"✓ Completed {count}/{len(tasks_to_process)}: {result}")
            except Exception as e:
                print(f"✗ Error processing {file_name}: {e}")
    
    print(f"\nVisualization completed! Created {count} videos")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Clear CoT Label - 双臂机器人动作原语和CoT标注工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    # 只标注动作原语
    python -m scripts.generate_embodied_data.clear_cot_label.main --stage primitives
    
    # 只生成CoT（需要已有动作原语）
    python -m scripts.generate_embodied_data.clear_cot_label.main --stage cot --no_api
    
    # 完整流程
    python -m scripts.generate_embodied_data.clear_cot_label.main --stage all
    
    # 只处理特定任务
    python -m scripts.generate_embodied_data.clear_cot_label.main --stage all --task_name adjust_bottle
        """
    )
    
    # 阶段选择
    parser.add_argument("--stage", type=str, default="cot",
                        choices=["primitives", "cot", "visualize", "all"],
                        help="运行阶段: primitives, cot, visualize, all")
    
    # 数据路径
    parser.add_argument("--base_dir", type=str,
                        default="/mnt/hdfs/_BYTE_DATA_SEED_/hl_lq/wangwenqian.wq/datasets/robotwin2_0816_raw",
                        help="数据集根目录")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/bn/ic-vlm/zhufangqi/embodied-CoT/results/clear_cot_label",
                        help="输出目录")
    parser.add_argument("--primitives_dir", type=str, default="",
                        help="动作原语JSON目录（CoT阶段使用）")
    parser.add_argument("--prompt_file", type=str, default="",
                        help="Prompt模板文件路径")
    
    # 动作原语参数
    parser.add_argument("--window_size", type=int, default=4, help="滑动窗口大小")
    parser.add_argument("--pos_threshold", type=float, default=0.02, help="位置阈值（米）")
    parser.add_argument("--rot_threshold", type=float, default=0.15, help="旋转阈值（弧度）")
    parser.add_argument("--gripper_threshold", type=float, default=0.05, help="gripper阈值")
    
    # 视频参数
    parser.add_argument("--fps", type=int, default=15, help="视频帧率")
    parser.add_argument("--sample_interval", type=int, default=1, help="原语采样间隔")
    
    # 控制选项
    parser.add_argument("--first_episode_only", default=False,
                        help="只处理每个task的第一个episode")
    parser.add_argument("--max_workers", type=int, default=8, help="primitives阶段并行处理线程数")
    parser.add_argument("--cot_workers", type=int, default=8, help="CoT阶段VLM调用并行线程数")
    parser.add_argument("--visualize_workers", type=int, default=8, help="可视化阶段并行线程数")
    parser.add_argument("--no_video", action="store_true", help="不生成可视化视频")
    parser.add_argument("--no_api", action="store_true", help="不调用API")
    parser.add_argument("--task_name", type=str, default="", help="只处理指定task")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Clear CoT Label - Bimanual Robot Annotation Tool")
    print("=" * 60)
    print(f"Stage: {args.stage}")
    print(f"Output dir: {args.output_dir}")
    print(f"Task filter: {args.task_name if args.task_name else 'All tasks'}")
    print("=" * 60)
    
    success = True
    
    if args.stage in ["primitives", "all"]:
        success = run_primitives_stage(args) and success
    
    if args.stage in ["cot", "all"]:
        success = run_cot_stage(args) and success
    
    if args.stage in ["visualize", "all"]:
        # 只有在cot阶段有结果时才运行可视化
        if not args.no_api or args.stage == "visualize":
            success = run_visualize_stage(args) and success
    
    print("\n" + "=" * 60)
    if success:
        print("All stages completed successfully!")
    else:
        print("Some stages had errors. Please check the output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
