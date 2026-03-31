"""
Shared helpers for CALVIN joint infill experiments.

Usage examples:
  cd /data/ytw/VLA_baseline/dllm
  /home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_probe.py
  CUDA_VISIBLE_DEVICES=0 /home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_eval.py --experiment_name baseline_eval_formal
  CUDA_VISIBLE_DEVICES=0 /home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_fit.py --experiment_name coord_train_eval_formal
"""

from __future__ import annotations

import json
import random
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional plotting dependency
    plt = None


def _require_matplotlib() -> None:
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is required for CALVIN joint infill plotting helpers"
        )


def _can_plot() -> bool:
    return plt is not None

from dllm.core.samplers.coord_proxy import build_text_action_region_masks

DATASET_PATH = Path(
    "/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl"
)
OUTPUT_ROOT = Path("/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill")
DEFAULT_COORD_MODULE_PATH = Path(
    "/data/ytw/VLA_baseline/dllm/.models/a2d/proxy-coordination-nl"
)

TEXT_START_MARKER = "Assistant response:"
TEXT_END_MARKER = "Action sequence:"
ACTION_START_MARKER = "Action sequence:"
ACTION_END_MARKER = "End of plan."
ACTION_REPRESENTATIONS = ("float_4dp", "float_2dp", "bucketed_int")


@dataclass
class CalvinJointInfillExample:
    sample_id: str
    task_name: str
    instruction: str
    think: str
    actions: list[list[float]]
    action_steps: int
    duration_sec: float
    sequence_start: int
    sequence_end: int


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def today_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_action_representation(action_representation: str) -> str:
    normalized = action_representation.strip().lower()
    if normalized not in ACTION_REPRESENTATIONS:
        raise ValueError(
            "action_representation must be one of "
            f"{ACTION_REPRESENTATIONS}, got {action_representation!r}"
        )
    return normalized


def _bucket_action_value(value: float, bucket_count: int) -> int:
    if bucket_count <= 1:
        raise ValueError("action_bucket_count must be greater than 1")
    clipped = max(-1.0, min(1.0, float(value)))
    scaled = (clipped + 1.0) / 2.0 * (bucket_count - 1)
    return int(round(scaled))


def action_representation_title(
    action_representation: str,
    *,
    round_digits: int,
    action_bucket_count: int,
) -> str:
    normalized = normalize_action_representation(action_representation)
    if normalized == "float_4dp":
        return f"float_4dp（{round_digits} 位小数）"
    if normalized == "float_2dp":
        return "float_2dp（2 位小数）"
    return f"bucketed_int（{action_bucket_count} 桶）"


def action_representation_description(
    action_representation: str,
    *,
    round_digits: int,
    action_bucket_count: int,
) -> str:
    normalized = normalize_action_representation(action_representation)
    if normalized == "float_4dp":
        return f"四舍五入到 {round_digits} 位小数"
    if normalized == "float_2dp":
        return "四舍五入到 2 位小数"
    return f"将每维动作裁剪到 [-1, 1] 后离散成 {action_bucket_count} 个整数桶"


def serialize_actions_with_representation(
    actions: list[list[float]],
    *,
    action_representation: str = "float_4dp",
    round_digits: int = 4,
    action_bucket_count: int = 8,
) -> str:
    normalized = normalize_action_representation(action_representation)
    if normalized in {"float_4dp", "float_2dp"}:
        digits = round_digits if normalized == "float_4dp" else 2
        fmt = f"{{:.{digits}f}}"
        steps = []
        for step in actions:
            values = ",".join(fmt.format(float(value)) for value in step)
            steps.append(f"[{values}]")
        return "; ".join(steps)

    steps = []
    for step in actions:
        values = ",".join(
            str(_bucket_action_value(value, action_bucket_count)) for value in step
        )
        steps.append(f"[{values}]")
    return "; ".join(steps)


def serialize_actions(actions: list[list[float]], round_digits: int = 4) -> str:
    return serialize_actions_with_representation(
        actions,
        action_representation="float_4dp",
        round_digits=round_digits,
    )


def make_joint_target(
    instruction: str,
    think: str,
    serialized_actions: str,
    text_start_marker: str = TEXT_START_MARKER,
    text_end_marker: str = TEXT_END_MARKER,
    action_start_marker: str = ACTION_START_MARKER,
    action_end_marker: str = ACTION_END_MARKER,
) -> str:
    return (
        f"Instruction: {instruction}\n"
        f"{text_start_marker}{think} {text_end_marker}\n"
        f"{action_start_marker}{serialized_actions} {action_end_marker}"
    )


def load_examples(
    jsonl_path: Path = DATASET_PATH,
    *,
    max_samples: int | None = None,
) -> list[CalvinJointInfillExample]:
    examples: list[CalvinJointInfillExample] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if max_samples is not None and line_idx >= max_samples:
                break
            row = json.loads(line)
            actions = row["actions"]
            examples.append(
                CalvinJointInfillExample(
                    sample_id=row["sample_id"],
                    task_name=row["task_name"],
                    instruction=row["instruction"],
                    think=row["think"],
                    actions=actions,
                    action_steps=len(actions),
                    duration_sec=float(row.get("duration_sec", 0.0)),
                    sequence_start=int(row.get("sequence_start", 0)),
                    sequence_end=int(row.get("sequence_end", 0)),
                )
            )
    return examples


def shuffled_examples(
    examples: list[CalvinJointInfillExample],
    *,
    seed: int,
) -> list[CalvinJointInfillExample]:
    rng = random.Random(seed)
    copied = list(examples)
    rng.shuffle(copied)
    return copied


def split_examples(
    examples: list[CalvinJointInfillExample],
    *,
    train_size: int,
    holdout_size: int,
    seed: int,
) -> tuple[list[CalvinJointInfillExample], list[CalvinJointInfillExample]]:
    ordered = shuffled_examples(examples, seed=seed)
    train = ordered[:train_size]
    holdout = ordered[train_size : train_size + holdout_size]
    return train, holdout


def split_examples_by_ratio(
    examples: list[CalvinJointInfillExample],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[CalvinJointInfillExample], list[CalvinJointInfillExample]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    ordered = shuffled_examples(examples, seed=seed)
    train_size = int(len(ordered) * train_ratio)
    train_size = max(1, min(train_size, len(ordered) - 1))
    return ordered[:train_size], ordered[train_size:]


def export_split_manifest(
    *,
    output_path: Path,
    train_examples: list[CalvinJointInfillExample],
    test_examples: list[CalvinJointInfillExample],
    seed: int,
    train_ratio: float,
) -> dict[str, Any]:
    manifest = {
        "split_date": today_date(),
        "split_timestamp": iso_now(),
        "seed": seed,
        "train_ratio": train_ratio,
        "train_count": len(train_examples),
        "test_count": len(test_examples),
        "train_sample_ids": [example.sample_id for example in train_examples],
        "test_sample_ids": [example.sample_id for example in test_examples],
    }
    dump_json(output_path, manifest)
    return manifest


def build_case(
    tokenizer,
    example: CalvinJointInfillExample,
    *,
    action_representation: str = "float_4dp",
    round_digits: int = 4,
    action_bucket_count: int = 8,
) -> tuple[list[int], list[int], str]:
    action_text = serialize_actions_with_representation(
        example.actions,
        action_representation=action_representation,
        round_digits=round_digits,
        action_bucket_count=action_bucket_count,
    )
    target = make_joint_target(
        instruction=example.instruction,
        think=example.think,
        serialized_actions=action_text,
    )
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    text_mask, action_mask, _ = build_text_action_region_masks(
        tokenizer,
        [target_ids],
        text_start_marker=TEXT_START_MARKER,
        text_end_marker=TEXT_END_MARKER,
        action_start_marker=ACTION_START_MARKER,
        action_end_marker=ACTION_END_MARKER,
    )
    prompt_ids = list(target_ids)
    region_mask = (text_mask | action_mask)[0].tolist()
    for idx, keep in enumerate(region_mask):
        if keep:
            prompt_ids[idx] = tokenizer.mask_token_id
    return prompt_ids, target_ids, action_text


def compute_region_metrics(
    tokenizer,
    *,
    prediction_ids: list[int],
    target_ids: list[int],
) -> dict[str, float]:
    text_mask, action_mask, _ = build_text_action_region_masks(
        tokenizer,
        [target_ids],
        text_start_marker=TEXT_START_MARKER,
        text_end_marker=TEXT_END_MARKER,
        action_start_marker=ACTION_START_MARKER,
        action_end_marker=ACTION_END_MARKER,
    )
    text_mask = text_mask[0].tolist()
    action_mask = action_mask[0].tolist()
    joint_mask = [t or a for t, a in zip(text_mask, action_mask)]

    def _acc(mask: list[bool]) -> float:
        pairs = [
            (pred, tgt)
            for pred, tgt, keep in zip(prediction_ids, target_ids, mask)
            if keep
        ]
        if not pairs:
            return 0.0
        return sum(int(pred == tgt) for pred, tgt in pairs) / len(pairs)

    def _exact(mask: list[bool]) -> float:
        pairs = [
            (pred, tgt)
            for pred, tgt, keep in zip(prediction_ids, target_ids, mask)
            if keep
        ]
        if not pairs:
            return 0.0
        return float(all(pred == tgt for pred, tgt in pairs))

    return {
        "text_region_token_acc": _acc(text_mask),
        "action_region_token_acc": _acc(action_mask),
        "joint_region_token_acc": _acc(joint_mask),
        "text_region_exact": _exact(text_mask),
        "action_region_exact": _exact(action_mask),
        "joint_region_exact": _exact(joint_mask),
        "text_region_tokens": float(sum(text_mask)),
        "action_region_tokens": float(sum(action_mask)),
        "joint_region_tokens": float(sum(joint_mask)),
    }


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        key: float(statistics.mean(float(row[key]) for row in rows))
        for key in numeric_keys
    }


def aggregate_task_metrics(
    case_rows: list[dict[str, Any]],
    *,
    baseline_key: str = "baseline_metrics",
    coordinated_key: str | None = "coordinated_metrics",
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in case_rows:
        task_name = row["task_name"]
        grouped.setdefault(task_name, {"baseline": [], "coordinated": []})
        grouped[task_name]["baseline"].append(row[baseline_key])
        if coordinated_key is not None and coordinated_key in row:
            grouped[task_name]["coordinated"].append(row[coordinated_key])

    summary: dict[str, Any] = {}
    for task_name, metrics in grouped.items():
        baseline = aggregate_metric_rows(metrics["baseline"])
        row = {"count": len(metrics["baseline"]), "baseline": baseline}
        if coordinated_key is not None and metrics["coordinated"]:
            coordinated = aggregate_metric_rows(metrics["coordinated"])
            row["coordinated"] = coordinated
            row["delta"] = {
                key: coordinated.get(key, 0.0) - baseline.get(key, 0.0)
                for key in coordinated.keys()
            }
        summary[task_name] = row
    return summary


def top_task_deltas(
    task_summary: dict[str, Any],
    *,
    metric_key: str = "joint_region_token_acc",
    limit: int = 15,
) -> dict[str, float]:
    rows = []
    for task_name, payload in task_summary.items():
        if "delta" in payload:
            rows.append((task_name, payload["delta"].get(metric_key, 0.0)))
        else:
            rows.append((task_name, payload["baseline"].get(metric_key, 0.0)))
    rows.sort(key=lambda item: item[1], reverse=True)
    return dict(rows[:limit])


def summarize_lengths(values: list[int | float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}

    def q(p: float) -> float:
        idx = int((len(ordered) - 1) * p)
        return ordered[idx]

    return {
        "min": ordered[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": ordered[-1],
        "mean": float(statistics.mean(ordered)),
    }


def trim_text(text: str, limit: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def save_bar_chart(
    values: dict[str, float],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if not _can_plot():
        return
    ensure_dir(output_path.parent)
    labels = list(values.keys())
    heights = [values[label] for label in labels]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, heights, color="#4C72B0")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_histogram(
    values: list[int | float],
    *,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 20,
) -> None:
    if not _can_plot():
        return
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color="#55A868", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_grouped_metric_chart(
    *,
    baseline: dict[str, float],
    coordinated: dict[str, float],
    metric_keys: list[str],
    output_path: Path,
    title: str,
) -> None:
    if not _can_plot():
        return
    ensure_dir(output_path.parent)
    labels = metric_keys
    baseline_vals = [baseline.get(key, 0.0) for key in labels]
    coordinated_vals = [coordinated.get(key, 0.0) for key in labels]
    x = list(range(len(labels)))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar([v - width / 2 for v in x], baseline_vals, width=width, label="baseline")
    plt.bar(
        [v + width / 2 for v in x],
        coordinated_vals,
        width=width,
        label="coordinated",
    )
    plt.title(title)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_curve(
    ys: list[float],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if not _can_plot():
        return
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(ys) + 1), ys, marker="o", color="#C44E52")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_task_delta_chart(
    *,
    task_summary: dict[str, Any],
    metric_key: str,
    output_path: Path,
    title: str,
) -> None:
    values = top_task_deltas(task_summary, metric_key=metric_key, limit=15)
    if not values:
        return
    save_bar_chart(values, title=title, ylabel=metric_key, output_path=output_path)


def make_experiment_dir(
    *,
    prefix: str,
    experiment_name: str | None = None,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    name = experiment_name or f"{prefix}_{timestamp()}"
    return ensure_dir(output_root / name)


def sample_preview_rows(
    examples: list[CalvinJointInfillExample],
    *,
    count: int = 10,
) -> list[dict[str, Any]]:
    previews = []
    for example in examples[:count]:
        previews.append(
            {
                "sample_id": example.sample_id,
                "task_name": example.task_name,
                "instruction": example.instruction,
                "think_preview": trim_text(example.think, limit=220),
                "action_steps": example.action_steps,
                "duration_sec": example.duration_sec,
            }
        )
    return previews


def build_probe_readme(
    *,
    report_path: Path,
    config: dict[str, Any],
    dataset_stats: dict[str, Any],
    figures: dict[str, Path],
    preview_rows: list[dict[str, Any]],
    notes: list[str],
) -> None:
    top_tasks = dataset_stats["top_tasks"]
    target_stats_by_rep = dataset_stats["target_token_stats_by_representation"]
    action_repr_title = action_representation_title(
        config["action_representation"],
        round_digits=config["round_digits"],
        action_bucket_count=config["action_bucket_count"],
    )
    readme = f"""# CALVIN Joint Infill 数据探查报告

- 实验日期：`{config["experiment_date"]}`
- 运行时间：`{config["experiment_timestamp"]}`

## 1. 实验目的
本次实验用于验证 `/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl` 是否适合直接构造成 joint infill 样本，并检查：

- 原始 JSONL 是否可以稳定读取
- `think + serialized actions` 的样本构造是否稳定
- text/action 两个区域是否能被 marker 正确定位
- 数据长度和任务分布是否适合后续的小规模验证

## 2. 数据来源与样本构造
- 原始数据路径：`{config["jsonl_path"]}`
- 原始数据只读：是
- 本次扫描样本数：`{dataset_stats["sample_count"]}`
- text 区域：`think`
- action 区域：`actions`
- 默认 action 表示：`{action_repr_title}`
- 动作序列格式：`[a1,...,a7]; [a1,...,a7]; ...`
- 本次是否做截断：`{config["token_filter_desc"]}`

## 3. 模型与配置
- tokenizer：`{config["tokenizer_name"]}`
- 随机种子：`{config["seed"]}`
- 输出目录：`{config["experiment_dir"]}`

## 4. 运行命令
```bash
cd /data/ytw/VLA_baseline/dllm
/home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_probe.py --experiment_name "{config["experiment_name"]}"
```

## 5. 指标定义
- `think_char_stats`：`think` 字符长度分布
- `action_step_stats`：动作步数分布
- `target_token_stats_by_representation`：不同动作表示下的联合样本 token 长度分布
- `mask_validation_pass_rate`：marker 能成功切出 text/action 区域的比例

## 6. 结果总表
- 样本总数：`{dataset_stats["sample_count"]}`
- 任务数：`{dataset_stats["task_count"]}`
- mask 验证通过率：`{dataset_stats["mask_validation_pass_rate"]:.4f}`
- think 长度统计：`{json.dumps(dataset_stats["think_char_stats"], ensure_ascii=False)}`
- action 步数统计：`{json.dumps(dataset_stats["action_step_stats"], ensure_ascii=False)}`

### 不同动作表示的 target token 长度统计

| representation | min | p50 | p90 | p95 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {representation} | {stats.get('min', 0.0):.1f} | {stats.get('p50', 0.0):.1f} | {stats.get('p90', 0.0):.1f} | {stats.get('p95', 0.0):.1f} | {stats.get('max', 0.0):.1f} | {stats.get('mean', 0.0):.1f} |"
        for representation, stats in target_stats_by_rep.items()
    ) + f"""

Top-10 任务分布：

| task | count |
|---|---:|
""" + "\n".join(f"| {task} | {count} |" for task, count in top_tasks[:10]) + f"""

## 7. 可视化结果
- 任务分布柱状图：`{figures["task_bar"]}`
- think 长度分布图：`{figures["think_hist"]}`
- action 步数分布图：`{figures["action_hist"]}`
- target token 长度分布图：`{figures["token_hist"]}`

## 8. 典型样本预览
| sample_id | task_name | action_steps | think_preview |
|---|---|---:|---|
""" + "\n".join(
        f'| {row["sample_id"]} | {row["task_name"]} | {row["action_steps"]} | {row["think_preview"]} |'
        for row in preview_rows
    ) + """

## 9. 结论与下一步
""" + "\n".join(f"- {note}" for note in notes)
    report_path.write_text(readme, encoding="utf-8")


def build_sample_readme(
    *,
    report_path: Path,
    config: dict[str, Any],
    aggregate_baseline: dict[str, float],
    aggregate_coordinated: dict[str, float] | None,
    figures: dict[str, Path],
    highlights: list[dict[str, Any]],
    task_summary: dict[str, Any] | None,
    conclusion_lines: list[str],
    baseline_only: bool = False,
) -> None:
    metric_keys = [
        "text_region_token_acc",
        "action_region_token_acc",
        "joint_region_token_acc",
        "joint_region_exact",
        "effective_steps",
        "latency_sec",
    ]
    title = (
        "CALVIN Joint Infill Baseline 测试报告"
        if baseline_only
        else "CALVIN Joint Infill 采样对比报告"
    )
    lines = [
        f"# {title}",
        "",
        f"- 实验日期：`{config['experiment_date']}`",
        f"- 运行时间：`{config['experiment_timestamp']}`",
        "",
        "## 1. 实验目的",
    ]
    if baseline_only:
        lines.append("本次实验在固定 test split 上评估 baseline 的 joint infill 恢复质量。")
    else:
        lines.extend(
            [
                "本次实验用真实 CALVIN 标注数据验证 joint infill 链路是否打通，并比较：",
                "",
                "- `enable_coordination=False`",
                "- `enable_coordination=True`",
                "",
                "关注点是 text/action 两个区域的联合恢复质量，而不是最终任务成功率。",
            ]
        )
    lines.extend(
        [
            "",
            "## 2. 数据来源与样本构造",
            f"- 原始数据路径：`{config['jsonl_path']}`",
            "- 原始数据只读：是",
            f"- 评估样本数：`{config['sample_count']}`",
            "- text 区域：`think`",
            f"- action 表示：`{action_representation_title(config['action_representation'], round_digits=config['round_digits'], action_bucket_count=config['action_bucket_count'])}`",
            f"- token 长度过滤：`{config['token_filter_desc']}`",
            f"- split 说明：`{config['split_desc']}`",
            f"- split 清单：`{config['split_manifest_path']}`",
            "",
            "## 3. 模型与配置",
            f"- 模型：`{config['model_name_or_path']}`",
            f"- coord module：`{config.get('coord_module_path', 'N/A')}`",
            f"- sampler steps：`{config['steps']}`",
            f"- few_step_budget：`{config['few_step_budget']}`",
            f"- coord_tokens：`{config['coord_tokens']}`",
            f"- 输出目录：`{config['experiment_dir']}`",
            "",
            "## 4. 运行命令",
            "```bash",
            "cd /data/ytw/VLA_baseline/dllm",
            f"/home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_eval.py --experiment_name \"{config['experiment_name']}\""
            if baseline_only
            else f"/home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_sample.py --experiment_name \"{config['experiment_name']}\"",
            "```",
            "",
            "## 5. 指标定义",
            "- `text_region_token_acc`：text 区域 token 命中率",
            "- `action_region_token_acc`：action 区域 token 命中率",
            "- `joint_region_token_acc`：text+action 联合区域 token 命中率",
            "- `joint_region_exact`：联合区域完全一致比例",
            "- `effective_steps`：有效去噪步数",
            "- `latency_sec`：单条样本推理时间",
            "",
            "## 6. 结果总表",
            "| metric | baseline | coordinated | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in metric_keys:
        if aggregate_coordinated is None:
            lines.append(f"| {key} | {aggregate_baseline.get(key, 0.0):.4f} | - | - |")
        else:
            baseline_value = aggregate_baseline.get(key, 0.0)
            coord_value = aggregate_coordinated.get(key, 0.0)
            lines.append(
                f"| {key} | {baseline_value:.4f} | {coord_value:.4f} | {coord_value - baseline_value:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 7. 可视化结果",
            f"- 指标对比图：`{figures['metric_bar']}`",
        ]
    )
    if task_summary:
        lines.append(f"- task-level 图：`{figures['task_bar']}`")
    lines.extend(["", "## 8. 典型案例"])
    if baseline_only:
        lines.extend(
            [
                "| sample_id | joint_acc_baseline | baseline_preview |",
                "|---|---:|---|",
            ]
        )
        for row in highlights:
            lines.append(
                f"| {row['sample_id']} | {row['baseline_joint']:.4f} | {row['baseline_preview']} |"
            )
    else:
        lines.extend(
            [
                "| sample_id | joint_acc_baseline | joint_acc_coord | delta | baseline_preview | coordinated_preview |",
                "|---|---:|---:|---:|---|---|",
            ]
        )
        for row in highlights:
            lines.append(
                f"| {row['sample_id']} | {row['baseline_joint']:.4f} | {row['coord_joint']:.4f} | {row['delta_joint']:.4f} | {row['baseline_preview']} | {row['coordinated_preview']} |"
            )
    if task_summary:
        lines.extend(["", "## 9. Task-Level 摘要"])
        if baseline_only:
            lines.extend(["| task | count | baseline_joint |", "|---|---:|---:|"])
            ordered_tasks = sorted(
                task_summary.items(),
                key=lambda item: item[1]["baseline"].get("joint_region_token_acc", 0.0),
                reverse=True,
            )[:10]
            for task_name, payload in ordered_tasks:
                lines.append(
                    f"| {task_name} | {payload['count']} | {payload['baseline'].get('joint_region_token_acc', 0.0):.4f} |"
                )
        else:
            lines.extend(["| task | count | delta_joint |", "|---|---:|---:|"])
            ordered_tasks = sorted(
                task_summary.items(),
                key=lambda item: item[1]["delta"].get("joint_region_token_acc", 0.0),
                reverse=True,
            )[:10]
            for task_name, payload in ordered_tasks:
                lines.append(
                    f"| {task_name} | {payload['count']} | {payload['delta'].get('joint_region_token_acc', 0.0):.4f} |"
                )
    lines.extend(["", "## 10. 结论与下一步"])
    lines.extend(f"- {line}" for line in conclusion_lines)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_fit_readme(
    *,
    report_path: Path,
    config: dict[str, Any],
    training_summary: dict[str, Any],
    aggregate_baseline: dict[str, float],
    aggregate_coordinated: dict[str, float],
    figures: dict[str, Path],
    highlights: list[dict[str, Any]],
    task_summary: dict[str, Any],
    conclusion_lines: list[str],
) -> None:
    lines = [
        "# CALVIN Joint Infill 协调模块训练报告",
        "",
        f"- 实验日期：`{config['experiment_date']}`",
        f"- 训练开始时间：`{config['train_start_timestamp']}`",
        f"- 训练结束时间：`{config['train_end_timestamp']}`",
        f"- 评估时间：`{config['eval_timestamp']}`",
        "",
        "## 1. 实验目的",
        "本次实验在真实 CALVIN joint infill 样本上只训练 coordination module，并在固定 test split 上比较 baseline 与 coordinated 的区域恢复质量。",
        "",
        "## 2. 数据来源与样本构造",
        f"- 原始数据路径：`{config['jsonl_path']}`",
        "- 原始数据只读：是",
        f"- 训练样本数：`{config['train_size']}`",
        f"- test 样本数：`{config['holdout_size']}`",
        "- text 区域：`think`",
        f"- action 表示：`{action_representation_title(config['action_representation'], round_digits=config['round_digits'], action_bucket_count=config['action_bucket_count'])}`",
        f"- token 长度过滤：`{config['token_filter_desc']}`",
        f"- split 说明：`{config['split_desc']}`",
        f"- split 清单：`{config['split_manifest_path']}`",
        "",
        "## 3. 模型与配置",
        f"- 模型：`{config['model_name_or_path']}`",
        f"- 学习率：`{config['learning_rate']}`",
        f"- epoch 数：`{config['num_epochs']}`",
        f"- coord_tokens：`{config['coord_tokens']}`",
        f"- few_step_budget：`{config['few_step_budget']}`",
        f"- 输出目录：`{config['experiment_dir']}`",
        "",
        "## 4. 运行命令",
        "```bash",
        "cd /data/ytw/VLA_baseline/dllm",
        f"/home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_fit.py --experiment_name \"{config['experiment_name']}\"",
        "```",
        "",
        "## 5. 指标定义",
        "- `epoch_loss`：每个 epoch 的平均 masked CE loss",
        "- `*_region_token_acc`：对应区域 token 恢复率",
        "- `joint_region_exact`：联合区域完全恢复比例",
        "",
        "## 6. 训练结果",
        f"- 训练完成 epoch 数：`{training_summary['epochs_completed']}`",
        f"- 最终平均训练损失：`{training_summary['final_train_loss']:.6f}`",
        f"- coord module 保存路径：`{training_summary['coord_module_output']}`",
        f"- 训练设备：`{training_summary['device']}`",
        "",
        "## 7. test 结果总表",
        "| metric | baseline | coordinated | delta |",
        "|---|---:|---:|---:|",
    ]
    for key in [
        "text_region_token_acc",
        "action_region_token_acc",
        "joint_region_token_acc",
        "joint_region_exact",
        "effective_steps",
        "latency_sec",
    ]:
        baseline_value = aggregate_baseline.get(key, 0.0)
        coord_value = aggregate_coordinated.get(key, 0.0)
        lines.append(
            f"| {key} | {baseline_value:.4f} | {coord_value:.4f} | {coord_value - baseline_value:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 8. 可视化结果",
            f"- loss 曲线：`{figures['loss_curve']}`",
            f"- baseline/coordinated 指标对比图：`{figures['metric_bar']}`",
            f"- task-level delta 图：`{figures['task_bar']}`",
            "",
            "## 9. 典型案例",
            "| sample_id | joint_acc_baseline | joint_acc_coord | delta | coordinated_preview |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in highlights:
        lines.append(
            f"| {row['sample_id']} | {row['baseline_joint']:.4f} | {row['coord_joint']:.4f} | {row['delta_joint']:.4f} | {row['coordinated_preview']} |"
        )
    lines.extend(["", "## 10. Task-Level 摘要", "| task | count | delta_joint |", "|---|---:|---:|"])
    ordered_tasks = sorted(
        task_summary.items(),
        key=lambda item: item[1]["delta"].get("joint_region_token_acc", 0.0),
        reverse=True,
    )[:10]
    for task_name, payload in ordered_tasks:
        lines.append(
            f"| {task_name} | {payload['count']} | {payload['delta'].get('joint_region_token_acc', 0.0):.4f} |"
        )
    lines.extend(["", "## 11. 结论与下一步"])
    lines.extend(f"- {line}" for line in conclusion_lines)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_highlights(
    case_rows: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ordered = sorted(
        case_rows,
        key=lambda row: row.get("coord_joint", row.get("baseline_joint", 0.0))
        - row.get("baseline_joint", 0.0),
        reverse=True,
    )
    return ordered[:limit]


def choose_failures(
    case_rows: list[dict[str, Any]],
    *,
    limit: int = 5,
    use_coord: bool = False,
) -> list[dict[str, Any]]:
    metric_key = "coord_joint" if use_coord else "baseline_joint"
    ordered = sorted(case_rows, key=lambda row: row.get(metric_key, 0.0))
    return ordered[:limit]


def save_case_table_plot(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    if not _can_plot():
        return
    ensure_dir(output_path.parent)
    if not rows:
        return
    columns = ["sample_id", "baseline_joint", "coord_joint", "delta_joint"]
    data = [
        [
            row["sample_id"],
            f'{row.get("baseline_joint", 0.0):.3f}',
            f'{row.get("coord_joint", row.get("baseline_joint", 0.0)):.3f}',
            f'{row.get("delta_joint", 0.0):.3f}',
        ]
        for row in rows
    ]
    plt.figure(figsize=(12, 0.8 * (len(rows) + 2)))
    plt.axis("off")
    table = plt.table(cellText=data, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def token_length_stats(
    tokenizer,
    examples: list[CalvinJointInfillExample],
    *,
    action_representation: str,
    round_digits: int,
    action_bucket_count: int,
    action_representations: tuple[str, ...] = ACTION_REPRESENTATIONS,
) -> tuple[dict[str, dict[str, float]], list[int], float]:
    stats_by_representation: dict[str, dict[str, float]] = {}
    selected_lengths: list[int] = []
    for representation in action_representations:
        lengths: list[int] = []
        for example in examples:
            _, target_ids, _ = build_case(
                tokenizer=tokenizer,
                example=example,
                action_representation=representation,
                round_digits=round_digits,
                action_bucket_count=action_bucket_count,
            )
            lengths.append(len(target_ids))
        stats_by_representation[representation] = summarize_lengths(lengths)
        if representation == action_representation:
            selected_lengths = lengths

    pass_count = 0
    for example in examples:
        _, target_ids, _ = build_case(
            tokenizer=tokenizer,
            example=example,
            action_representation=action_representation,
            round_digits=round_digits,
            action_bucket_count=action_bucket_count,
        )
        try:
            build_text_action_region_masks(
                tokenizer,
                [target_ids],
                text_start_marker=TEXT_START_MARKER,
                text_end_marker=TEXT_END_MARKER,
                action_start_marker=ACTION_START_MARKER,
                action_end_marker=ACTION_END_MARKER,
            )
            pass_count += 1
        except Exception:
            pass
    pass_rate = pass_count / max(len(examples), 1)
    return stats_by_representation, selected_lengths, pass_rate


def filter_by_token_length(
    tokenizer,
    examples: list[CalvinJointInfillExample],
    *,
    action_representation: str,
    round_digits: int,
    action_bucket_count: int,
    max_target_tokens: int | None,
) -> tuple[list[CalvinJointInfillExample], dict[str, int]]:
    if max_target_tokens is None:
        return list(examples), {"kept": len(examples), "dropped": 0}
    kept: list[CalvinJointInfillExample] = []
    dropped = 0
    for example in examples:
        _, target_ids, _ = build_case(
            tokenizer=tokenizer,
            example=example,
            action_representation=action_representation,
            round_digits=round_digits,
            action_bucket_count=action_bucket_count,
        )
        if len(target_ids) <= max_target_tokens:
            kept.append(example)
        else:
            dropped += 1
    return kept, {"kept": len(kept), "dropped": dropped}


def default_coord_module_path() -> str | None:
    if DEFAULT_COORD_MODULE_PATH.exists():
        return str(DEFAULT_COORD_MODULE_PATH)
    return None


def config_to_dict(args) -> dict[str, Any]:
    if hasattr(args, "__dataclass_fields__"):
        return asdict(args)
    return {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def with_experiment_metadata(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    updated["experiment_date"] = today_date()
    updated["experiment_timestamp"] = iso_now()
    return updated
