"""
Run:
  cd /data/ytw/VLA_baseline/dllm
  source ~/.zshrc
  conda activate ~/miniconda3/envs/dllm
  python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_diagnose.py --experiment_name failure_diagnosis_formal
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import transformers

OUTPUT_ROOT = Path("/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill")

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]")
PUNCT_RE = re.compile(r"^[^\w\s]+$")
DIGIT_RE = re.compile(r"\d")


@dataclass
class ScriptArguments:
    output_root: str = str(OUTPUT_ROOT)
    experiment_name: str | None = "failure_diagnosis_formal"
    bucket_count: int = 5
    top_k_tasks: int = 10
    include_supporting_runs: bool = True
    repetition_threshold: float = 0.30
    punctuation_threshold: float = 0.35
    unique_ratio_threshold: float = 0.20
    action_failure_threshold: float = 0.02
    text_gain_threshold: float = 0.003
    action_flat_threshold: float = 0.001


parser = transformers.HfArgumentParser((ScriptArguments,))


def today_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def trim_text(text: str, limit: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def make_experiment_dir(
    *,
    prefix: str,
    experiment_name: str | None = None,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    if experiment_name:
        return ensure_dir(output_root / experiment_name)
    return ensure_dir(output_root / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def with_experiment_metadata(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    updated["experiment_date"] = today_date()
    updated["experiment_timestamp"] = iso_now()
    return updated


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_supporting_run(name: str) -> bool:
    return name in {
        "probe_initial",
        "sample_smoke",
        "fit_smoke",
        "baseline_eval_formal",
        "coord_train_eval_formal",
    }


def load_experiment(exp_dir: Path) -> dict[str, Any] | None:
    metrics_path = exp_dir / "metrics.json"
    cases_path = exp_dir / "cases.jsonl"
    if not metrics_path.exists() or not cases_path.exists():
        return None
    try:
        metrics = read_json(metrics_path)
        cases = read_jsonl(cases_path)
    except Exception:
        return None
    return {
        "name": exp_dir.name,
        "path": exp_dir,
        "metrics": metrics,
        "cases": cases,
        "has_coordinated": any("coordinated_metrics" in row for row in cases),
        "has_baseline_predictions": any("baseline_prediction" in row for row in cases),
        "has_target_text": any("target_text" in row for row in cases),
    }


def longest_repeat_run(tokens: list[str]) -> int:
    if not tokens:
        return 0
    best = 1
    run = 1
    previous = tokens[0]
    for token in tokens[1:]:
        if token == previous:
            run += 1
        else:
            best = max(best, run)
            run = 1
            previous = token
    return max(best, run)


def analyze_text(text: str) -> dict[str, float]:
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return {
            "token_count": 0.0,
            "unique_ratio": 0.0,
            "adjacent_repeat_ratio": 0.0,
            "punctuation_ratio": 0.0,
            "digit_ratio": 0.0,
            "top_token_share": 0.0,
            "max_repeat_run": 0.0,
        }

    normalized = [token.lower() for token in tokens]
    counts = Counter(normalized)
    adjacent_repeats = sum(
        1 for left, right in zip(normalized, normalized[1:]) if left == right
    )
    punct_count = sum(1 for token in tokens if PUNCT_RE.match(token))
    digit_count = sum(1 for token in tokens if DIGIT_RE.search(token))
    unique_count = len(counts)
    top_token_share = max(counts.values()) / len(tokens)

    return {
        "token_count": float(len(tokens)),
        "unique_ratio": unique_count / len(tokens),
        "adjacent_repeat_ratio": adjacent_repeats / max(len(tokens) - 1, 1),
        "punctuation_ratio": punct_count / len(tokens),
        "digit_ratio": digit_count / len(tokens),
        "top_token_share": top_token_share,
        "max_repeat_run": float(longest_repeat_run(normalized)),
    }


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set()
    for row in rows:
        keys.update(row.keys())
    return {
        key: float(statistics.mean(row[key] for row in rows if key in row))
        for key in sorted(keys)
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if q <= 0:
        return float(ordered[0])
    if q >= 1:
        return float(ordered[-1])
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return float(ordered[low] * (1 - fraction) + ordered[high] * fraction)


def build_bucket_edges(lengths: list[float], bucket_count: int) -> list[float]:
    if not lengths:
        return [0.0, 1.0]
    if bucket_count < 2:
        bucket_count = 2
    qs = [i / bucket_count for i in range(bucket_count + 1)]
    edges = [percentile(lengths, q) for q in qs]
    if edges[0] == edges[-1]:
        return [float(min(lengths)), float(max(lengths))]
    return edges


def assign_bucket(length: float, edges: list[float]) -> int:
    if len(edges) <= 2:
        return 0
    for index in range(len(edges) - 1):
        left = edges[index]
        right = edges[index + 1]
        if index == len(edges) - 2:
            if left <= length <= right:
                return index
        elif left <= length < right:
            return index
    return max(len(edges) - 2, 0)


def bucket_label(index: int, edges: list[float]) -> str:
    if len(edges) <= 2:
        return f"bucket_{index + 1}"
    left = edges[index]
    right = edges[index + 1]
    if index == len(edges) - 2:
        return f"[{left:.0f}, {right:.0f}]"
    return f"[{left:.0f}, {right:.0f})"


def get_prediction(row: dict[str, Any], side: str) -> str:
    prediction_key = f"{side}_prediction"
    preview_key = f"{side}_preview"
    if prediction_key in row and isinstance(row[prediction_key], str):
        return row[prediction_key]
    if preview_key in row and isinstance(row[preview_key], str):
        return row[preview_key]
    if side == "coordinated" and row.get("coordinated_metrics") is not None:
        return row.get("baseline_prediction", row.get("baseline_preview", ""))
    return row.get("target_text", "")


def get_metrics(row: dict[str, Any], side: str) -> dict[str, float]:
    metrics_key = f"{side}_metrics"
    metrics = row.get(metrics_key, {})
    if not isinstance(metrics, dict):
        return {}
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def diagnose_prediction(
    *,
    metrics: dict[str, float],
    stats: dict[str, float],
    delta_text: float = 0.0,
    delta_action: float = 0.0,
    delta_joint: float = 0.0,
    repetition_threshold: float,
    punctuation_threshold: float,
    unique_ratio_threshold: float,
    action_failure_threshold: float,
    text_gain_threshold: float,
    action_flat_threshold: float,
    has_pair: bool,
) -> list[str]:
    labels: list[str] = []
    if (
        stats["adjacent_repeat_ratio"] >= repetition_threshold
        or stats["unique_ratio"] <= unique_ratio_threshold
        or stats["max_repeat_run"] >= 8.0
    ):
        labels.append("repetition_collapse")
    if (
        stats["punctuation_ratio"] >= punctuation_threshold
        or (
            stats["top_token_share"] >= 0.40
            and stats["punctuation_ratio"] >= 0.20
        )
    ):
        labels.append("punctuation_collapse")
    if has_pair and delta_text >= text_gain_threshold and delta_action <= action_flat_threshold:
        labels.append("text_only_gain")
    if (
        metrics.get("action_region_token_acc", 0.0) <= action_failure_threshold
        or (
            metrics.get("joint_region_token_acc", 0.0) <= 0.02
            and stats["unique_ratio"] <= 0.30
        )
    ):
        labels.append("action_failure")
    if (
        has_pair
        and abs(delta_joint) < 5e-4
        and abs(delta_text) < 5e-4
        and abs(delta_action) < 5e-4
    ):
        labels.append("no_change")
    if not labels:
        labels.append("other")
    return labels


def aggregate_rows(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(statistics.mean(row.get(key, 0.0) for row in rows))
        for key in keys
    }


def summarize_bucket_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for bucket, bucket_rows in sorted(rows, key=lambda item: item[0]):
        bucket_name = bucket_rows[0]["length_bucket"]
        baseline_metric_rows = [row["baseline_metrics"] for row in bucket_rows]
        coord_metric_rows = [
            row["coordinated_metrics"]
            for row in bucket_rows
            if row.get("coordinated_metrics") is not None
        ]
        baseline_signal_rows = [row["baseline_signals"] for row in bucket_rows]
        coord_signal_rows = [
            row["coordinated_signals"]
            for row in bucket_rows
            if row.get("coordinated_signals") is not None
        ]
        summary[bucket_name] = {
            "count": len(bucket_rows),
            "length_min": float(min(row["length"] for row in bucket_rows)),
            "length_max": float(max(row["length"] for row in bucket_rows)),
            "baseline_metrics": mean_dict(baseline_metric_rows),
            "coordinated_metrics": mean_dict(coord_metric_rows) if coord_metric_rows else None,
            "baseline_signals": mean_dict(baseline_signal_rows),
            "coordinated_signals": mean_dict(coord_signal_rows) if coord_signal_rows else None,
        }
    return summary


def summarize_task_ranking(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[row["task_name"]].append(row)

    task_rows = []
    for task_name, rows in grouped.items():
        baseline_metrics = [row["baseline_metrics"] for row in rows]
        coord_metrics = [
            row["coordinated_metrics"]
            for row in rows
            if row.get("coordinated_metrics") is not None
        ]
        baseline = mean_dict(baseline_metrics)
        coordinated = mean_dict(coord_metrics) if coord_metrics else None
        task_row: dict[str, Any] = {
            "task_name": task_name,
            "count": len(rows),
            "baseline": baseline,
        }
        if coordinated is not None:
            task_row["coordinated"] = coordinated
            task_row["delta"] = {
                key: coordinated.get(key, 0.0) - baseline.get(key, 0.0)
                for key in coordinated.keys()
            }
        task_rows.append(task_row)

    improved = sorted(
        [row for row in task_rows if "delta" in row],
        key=lambda item: item["delta"].get("joint_region_token_acc", 0.0),
        reverse=True,
    )
    regressed = sorted(
        [row for row in task_rows if "delta" in row],
        key=lambda item: item["delta"].get("joint_region_token_acc", 0.0),
    )
    return {
        "all_tasks": task_rows,
        "top_improved": improved[:10],
        "top_regressed": regressed[:10],
    }


def format_metric_row(row: dict[str, float] | None, keys: list[str]) -> str:
    if row is None:
        return "-"
    return ", ".join(f"{key}={row.get(key, 0.0):.4f}" for key in keys)


def write_readme(
    *,
    report_path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    lines: list[str] = [
        "# CALVIN Joint Infill 失败模式诊断报告",
        "",
        f"- 生成日期：`{config['experiment_date']}`",
        f"- 生成时间：`{config['experiment_timestamp']}`",
        f"- 输出目录：`{config['experiment_dir']}`",
        f"- 扫描根目录：`{config['output_root']}`",
        "",
        "## 1. 结论摘要",
        f"- 主结论：{summary['headline']}",
        f"- 配对实验数量：`{summary['paired_experiment_count']}`",
        f"- case 总数：`{summary['total_case_count']}`",
        f"- 预测未变化比例：`{summary['same_prediction_rate']:.2%}`",
        f"- `no_change` 标签占比：`{summary['label_rates'].get('no_change', 0.0):.2%}`",
        f"- `repetition_collapse` 标签占比：`{summary['label_rates'].get('repetition_collapse', 0.0):.2%}`",
        f"- `punctuation_collapse` 标签占比：`{summary['label_rates'].get('punctuation_collapse', 0.0):.2%}`",
        f"- `text_only_gain` 标签占比：`{summary['label_rates'].get('text_only_gain', 0.0):.2%}`",
        "",
        "## 2. 扫描到的实验",
    ]
    for exp in summary["experiments"]:
        lines.append(
            f"- `{exp['name']}`: cases={exp['case_count']}, paired={exp['has_pair']}, role={exp['role']}"
        )

    lines.extend(
        [
            "",
            "## 3. 长度分桶",
            f"- 分桶边界：`{summary['bucket_edges']}`",
            "",
            "| bucket | count | baseline joint | coord joint | delta | coord repeat | coord punct | coord unique |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket_name, bucket_row in summary["length_buckets"].items():
        baseline = bucket_row["baseline_metrics"]
        coord = bucket_row["coordinated_metrics"]
        coord_signals = bucket_row["coordinated_signals"]
        lines.append(
            "| {bucket} | {count} | {baseline_joint:.4f} | {coord_joint:.4f} | {delta:.4f} | {repeat:.4f} | {punct:.4f} | {unique:.4f} |".format(
                bucket=bucket_name,
                count=bucket_row["count"],
                baseline_joint=baseline.get("joint_region_token_acc", 0.0),
                coord_joint=coord.get("joint_region_token_acc", 0.0) if coord else baseline.get("joint_region_token_acc", 0.0),
                delta=(coord.get("joint_region_token_acc", 0.0) - baseline.get("joint_region_token_acc", 0.0)) if coord else 0.0,
                repeat=coord_signals.get("adjacent_repeat_ratio", 0.0) if coord_signals else bucket_row["baseline_signals"].get("adjacent_repeat_ratio", 0.0),
                punct=coord_signals.get("punctuation_ratio", 0.0) if coord_signals else bucket_row["baseline_signals"].get("punctuation_ratio", 0.0),
                unique=coord_signals.get("unique_ratio", 0.0) if coord_signals else bucket_row["baseline_signals"].get("unique_ratio", 0.0),
            )
        )

    lines.extend(
        [
            "",
            "## 4. 任务 Delta 排名",
            "",
            "### 提升最多",
            "| task | count | delta joint | delta text | delta action |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["task_rankings"]["top_improved"]:
        lines.append(
            f"| {row['task_name']} | {row['count']} | {row['delta'].get('joint_region_token_acc', 0.0):.4f} | {row['delta'].get('text_region_token_acc', 0.0):.4f} | {row['delta'].get('action_region_token_acc', 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "### 回退最多",
            "| task | count | delta joint | delta text | delta action |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["task_rankings"]["top_regressed"]:
        lines.append(
            f"| {row['task_name']} | {row['count']} | {row['delta'].get('joint_region_token_acc', 0.0):.4f} | {row['delta'].get('text_region_token_acc', 0.0):.4f} | {row['delta'].get('action_region_token_acc', 0.0):.4f} |"
        )

    lines.extend(
        [
            "",
            "## 5. 退化模式统计",
            "| signal | baseline mean | coord mean | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ["adjacent_repeat_ratio", "punctuation_ratio", "digit_ratio", "unique_ratio", "top_token_share"]:
        baseline_value = summary["paired_signal_means"]["baseline"].get(key, 0.0)
        coord_value = summary["paired_signal_means"]["coordinated"].get(key, 0.0)
        lines.append(
            f"| {key} | {baseline_value:.4f} | {coord_value:.4f} | {coord_value - baseline_value:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 6. 自动标签",
            "| label | count | rate |",
            "|---|---:|---:|",
        ]
    )
    for label, count in sorted(summary["label_counts"].items(), key=lambda item: item[0]):
        rate = count / max(summary["total_case_count"], 1)
        lines.append(f"| {label} | {count} | {rate:.2%} |")

    lines.extend(
        [
            "",
            "## 7. 典型案例",
            "| sample_id | task_name | delta_joint | labels | preview |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in summary["highlight_cases"]:
        lines.append(
            f"| {row['sample_id']} | {row['task_name']} | {row['delta_joint']:.4f} | {', '.join(row['labels'])} | {row['preview']} |"
        )

    lines.extend(
        [
            "",
            "## 8. 下一步建议",
        ]
    )
    for item in summary["next_steps"]:
        lines.append(f"- {item}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    script_args, = parser.parse_args_into_dataclasses()
    experiment_dir = make_experiment_dir(
        prefix="diagnose",
        experiment_name=script_args.experiment_name,
        output_root=Path(script_args.output_root),
    )
    ensure_dir(experiment_dir)

    root = Path(script_args.output_root)
    experiment_dirs = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir()
        and not path.name.startswith("failure_diagnosis")
        and not path.name.startswith("diagnose_")
        and (path / "metrics.json").exists()
        and (path / "cases.jsonl").exists()
    ]

    experiments = []
    paired_rows: list[dict[str, Any]] = []
    all_case_rows: list[dict[str, Any]] = []
    for exp_dir in experiment_dirs:
        payload = load_experiment(exp_dir)
        if payload is None:
            continue
        if not script_args.include_supporting_runs and payload["name"] not in {
            "baseline_eval_formal",
            "coord_train_eval_formal",
        }:
            continue
        role = "paired" if payload["has_coordinated"] else "baseline_only"
        if payload["name"] == "probe_initial":
            role = "probe"
        elif payload["name"] in {"sample_smoke", "fit_smoke"}:
            role = "smoke"
        experiments.append(
            {
                "name": payload["name"],
                "path": str(payload["path"]),
                "case_count": len(payload["cases"]),
                "has_pair": payload["has_coordinated"],
                "role": role,
            }
        )
        for row in payload["cases"]:
            row = dict(row)
            row["experiment_name"] = payload["name"]
            row["experiment_path"] = str(payload["path"])
            row["has_pair"] = payload["has_coordinated"]
            row["has_baseline_prediction"] = payload["has_baseline_predictions"]
            row["has_target_text"] = payload["has_target_text"]
            all_case_rows.append(row)
            if payload["has_coordinated"]:
                paired_rows.append(row)

    if not all_case_rows:
        raise RuntimeError(f"no experiment outputs found under {root}")

    analysis_rows = [row for row in all_case_rows if get_metrics(row, "baseline")]

    all_lengths = []
    for row in analysis_rows:
        baseline_metrics = get_metrics(row, "baseline")
        if baseline_metrics:
            all_lengths.append(baseline_metrics.get("joint_region_tokens", 0.0))
        if row.get("coordinated_metrics"):
            all_lengths.append(get_metrics(row, "coordinated").get("joint_region_tokens", 0.0))
    bucket_edges = build_bucket_edges(all_lengths, script_args.bucket_count)
    bucket_map: dict[int, list[dict[str, Any]]] = defaultdict(list)

    case_diagnostics: list[dict[str, Any]] = []
    paired_signal_baseline_rows: list[dict[str, float]] = []
    paired_signal_coord_rows: list[dict[str, float]] = []
    label_counts: Counter[str] = Counter()
    same_prediction_count = 0

    for row in analysis_rows:
        baseline_metrics = get_metrics(row, "baseline")
        coord_metrics = get_metrics(row, "coordinated") if row.get("coordinated_metrics") else None
        baseline_text = get_prediction(row, "baseline")
        coord_text = get_prediction(row, "coordinated") if row.get("coordinated_metrics") else None
        analysis_side = "coordinated" if coord_text is not None and coord_metrics is not None else "baseline"
        analysis_text = coord_text if coord_text is not None else baseline_text
        analysis_metrics = coord_metrics if coord_metrics is not None else baseline_metrics
        analysis_stats = analyze_text(analysis_text)

        baseline_stats = analyze_text(baseline_text)
        coord_stats = analyze_text(coord_text) if coord_text is not None else None

        delta_text = 0.0
        delta_action = 0.0
        delta_joint = 0.0
        if coord_metrics is not None:
            delta_text = coord_metrics.get("text_region_token_acc", 0.0) - baseline_metrics.get("text_region_token_acc", 0.0)
            delta_action = coord_metrics.get("action_region_token_acc", 0.0) - baseline_metrics.get("action_region_token_acc", 0.0)
            delta_joint = coord_metrics.get("joint_region_token_acc", 0.0) - baseline_metrics.get("joint_region_token_acc", 0.0)

        labels = diagnose_prediction(
            metrics=analysis_metrics,
            stats=analysis_stats,
            delta_text=delta_text,
            delta_action=delta_action,
            delta_joint=delta_joint,
            repetition_threshold=script_args.repetition_threshold,
            punctuation_threshold=script_args.punctuation_threshold,
            unique_ratio_threshold=script_args.unique_ratio_threshold,
            action_failure_threshold=script_args.action_failure_threshold,
            text_gain_threshold=script_args.text_gain_threshold,
            action_flat_threshold=script_args.action_flat_threshold,
            has_pair=coord_metrics is not None,
        )
        label_counts.update(labels)
        if coord_text is not None and baseline_text == coord_text:
            same_prediction_count += 1

        row_length = baseline_metrics.get("joint_region_tokens", coord_metrics.get("joint_region_tokens", 0.0) if coord_metrics else 0.0)
        bucket_index = assign_bucket(row_length, bucket_edges)
        bucket = bucket_label(bucket_index, bucket_edges)
        case_row = {
            "experiment_name": row["experiment_name"],
            "sample_id": row.get("sample_id", ""),
            "task_name": row.get("task_name", ""),
            "length_bucket": bucket,
            "length": float(row_length),
            "baseline_joint": baseline_metrics.get("joint_region_token_acc", 0.0),
            "coord_joint": coord_metrics.get("joint_region_token_acc", baseline_metrics.get("joint_region_token_acc", 0.0)) if coord_metrics is not None else baseline_metrics.get("joint_region_token_acc", 0.0),
            "delta_joint": delta_joint,
            "baseline_text": baseline_text,
            "coord_text": coord_text,
            "baseline_metrics": baseline_metrics,
            "coordinated_metrics": coord_metrics,
            "baseline_signals": baseline_stats,
            "coordinated_signals": coord_stats,
            "analysis_side": analysis_side,
            "analysis_signals": analysis_stats,
            "labels": labels,
            "baseline_labels": diagnose_prediction(
                metrics=baseline_metrics,
                stats=baseline_stats,
                repetition_threshold=script_args.repetition_threshold,
                punctuation_threshold=script_args.punctuation_threshold,
                unique_ratio_threshold=script_args.unique_ratio_threshold,
                action_failure_threshold=script_args.action_failure_threshold,
                text_gain_threshold=script_args.text_gain_threshold,
                action_flat_threshold=script_args.action_flat_threshold,
                has_pair=False,
            ),
        }
        if coord_text is not None and coord_metrics is not None:
            case_row["coordinated_labels"] = labels
            paired_signal_baseline_rows.append(baseline_stats)
            if coord_stats is not None:
                paired_signal_coord_rows.append(coord_stats)
        case_row["bucket_index"] = bucket_index
        bucket_map[bucket_index].append(case_row)
        case_diagnostics.append(case_row)

    length_buckets = summarize_bucket_rows(list(bucket_map.items()))
    task_rankings = summarize_task_ranking(paired_rows if paired_rows else all_case_rows)

    paired_case_rows = [row for row in case_diagnostics if row["coordinated_metrics"] is not None]
    paired_case_count = len(paired_case_rows)
    paired_label_counts = Counter()
    for row in paired_case_rows:
        paired_label_counts.update(row["labels"])

    paired_signal_means = {
        "baseline": mean_dict(paired_signal_baseline_rows),
        "coordinated": mean_dict(paired_signal_coord_rows),
    }
    if paired_signal_coord_rows:
        same_prediction_rate = same_prediction_count / max(paired_case_count, 1)
    else:
        same_prediction_rate = 0.0

    label_rates = {
        label: count / max(paired_case_count, 1)
        for label, count in paired_label_counts.items()
    }
    if not paired_label_counts:
        label_rates = {}

    highlight_cases = sorted(
        paired_case_rows,
        key=lambda row: row["delta_joint"],
        reverse=True,
    )[:10]
    for row in highlight_cases:
        row["preview"] = trim_text(row["coord_text"] or row["baseline_text"], limit=180)

    paired_task_ranking = task_rankings["all_tasks"]
    headline = (
        "协调模块对 text 有轻微帮助，但 action 侧没有同步改善，joint 指标几乎没变。"
        if paired_case_count
        and abs(paired_signal_means["coordinated"].get("joint_region_token_acc", 0.0) - paired_signal_means["baseline"].get("joint_region_token_acc", 0.0)) < 0.001
        else "当前实验没有显示出稳定的 joint infill 提升。"
    )

    summary = {
        "headline": headline,
        "experiments": experiments,
        "paired_experiment_count": sum(1 for exp in experiments if exp["has_pair"]),
        "total_case_count": len(case_diagnostics),
        "paired_case_count": paired_case_count,
        "same_prediction_rate": same_prediction_rate,
        "label_counts": dict(paired_label_counts),
        "label_rates": label_rates,
        "bucket_edges": [float(edge) for edge in bucket_edges],
        "length_buckets": length_buckets,
        "paired_signal_means": paired_signal_means,
        "task_rankings": {
            "top_improved": task_rankings["top_improved"][: script_args.top_k_tasks],
            "top_regressed": task_rankings["top_regressed"][: script_args.top_k_tasks],
        },
        "highlight_cases": highlight_cases,
        "next_steps": [
            "优先处理 action 表示压缩，因为长度分桶里长序列通常伴随更高的重复和标点塌缩。",
            "如果希望 coordination 真正起作用，需要先确认 backbone 能在同一 joint target 上给出更稳定的序列结构。",
            "当前最强信号不是 joint 提升，而是 text 端轻微改善但 action 端几乎没动。",
        ],
    }

    config = with_experiment_metadata(
        {
            "output_root": str(root),
            "experiment_dir": str(experiment_dir),
            "bucket_count": script_args.bucket_count,
            "top_k_tasks": script_args.top_k_tasks,
            "include_supporting_runs": script_args.include_supporting_runs,
            "repetition_threshold": script_args.repetition_threshold,
            "punctuation_threshold": script_args.punctuation_threshold,
            "unique_ratio_threshold": script_args.unique_ratio_threshold,
            "action_failure_threshold": script_args.action_failure_threshold,
            "text_gain_threshold": script_args.text_gain_threshold,
            "action_flat_threshold": script_args.action_flat_threshold,
        }
    )

    dump_json(experiment_dir / "config.json", config)
    dump_json(experiment_dir / "metrics.json", summary)
    dump_jsonl(experiment_dir / "cases.jsonl", case_diagnostics)
    write_readme(
        report_path=experiment_dir / "README.md",
        config=config,
        summary=summary,
    )
    print(f"diagnosis report written to {experiment_dir}")


if __name__ == "__main__":
    main()
