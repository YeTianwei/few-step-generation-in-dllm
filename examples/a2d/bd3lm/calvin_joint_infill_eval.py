"""
Run baseline-only evaluation on a fixed 8:2 CALVIN split and write a Chinese README report.

Run:
  cd /data/ytw/VLA_baseline/dllm
  CUDA_VISIBLE_DEVICES=0 /home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_eval.py --experiment_name baseline_eval_formal
  CUDA_VISIBLE_DEVICES=0 /home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_eval.py --experiment_name baseline_eval_float2 --action_representation float_2dp
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import transformers

import dllm

from calvin_joint_infill import (
    DATASET_PATH,
    OUTPUT_ROOT,
    aggregate_metric_rows,
    aggregate_task_metrics,
    build_case,
    build_sample_readme,
    choose_failures,
    choose_highlights,
    compute_region_metrics,
    config_to_dict,
    dump_json,
    dump_jsonl,
    ensure_dir,
    export_split_manifest,
    filter_by_token_length,
    load_examples,
    make_experiment_dir,
    save_case_table_plot,
    save_grouped_metric_chart,
    save_task_delta_chart,
    split_examples_by_ratio,
    trim_text,
    with_experiment_metadata,
    normalize_action_representation,
)


@dataclass
class ScriptArguments:
    model_name_or_path: str = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
    jsonl_path: str = str(DATASET_PATH)
    output_root: str = str(OUTPUT_ROOT)
    experiment_name: str | None = None
    seed: int = 42
    train_ratio: float = 0.8
    max_target_tokens: int = 4096
    round_digits: int = 4
    action_representation: str = "float_4dp"
    action_bucket_count: int = 8

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )
        self.action_representation = normalize_action_representation(
            self.action_representation
        )


@dataclass
class SamplerConfig(dllm.core.samplers.CoordinationProxySamplerConfig):
    steps: int = 24
    few_step_budget: int = 8
    block_size: int | None = None
    temperature: float = 0.0
    coord_tokens: int = 64
    text_transfer_ratio: float = 0.7
    action_transfer_ratio: float = 1.3
    coord_confidence_scale: float = 0.25


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))


def main() -> None:
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)
    experiment_dir = make_experiment_dir(
        prefix="baseline_eval",
        experiment_name=script_args.experiment_name,
        output_root=Path(script_args.output_root),
    )
    figures_dir = ensure_dir(experiment_dir / "figures")

    model = dllm.utils.get_model(model_name_or_path=script_args.model_name_or_path).eval()
    tokenizer = dllm.utils.get_tokenizer(model_name_or_path=script_args.model_name_or_path)
    sampler = dllm.core.samplers.CoordinationProxySampler(model=model, tokenizer=tokenizer)

    examples = load_examples(Path(script_args.jsonl_path))
    examples, filter_stats = filter_by_token_length(
        tokenizer,
        examples,
        action_representation=script_args.action_representation,
        round_digits=script_args.round_digits,
        action_bucket_count=script_args.action_bucket_count,
        max_target_tokens=script_args.max_target_tokens,
    )
    train_examples, test_examples = split_examples_by_ratio(
        examples,
        train_ratio=script_args.train_ratio,
        seed=script_args.seed,
    )
    split_manifest = export_split_manifest(
        output_path=experiment_dir / "split_manifest.json",
        train_examples=train_examples,
        test_examples=test_examples,
        seed=script_args.seed,
        train_ratio=script_args.train_ratio,
    )

    case_rows = []
    eval_start = time.perf_counter()
    for example in test_examples:
        inputs, targets, action_text = build_case(
            tokenizer=tokenizer,
            example=example,
            action_representation=script_args.action_representation,
            round_digits=script_args.round_digits,
            action_bucket_count=script_args.action_bucket_count,
        )
        baseline_config = SamplerConfig(**sampler_config.__dict__)
        baseline_config.enable_coordination = False
        start = time.perf_counter()
        baseline = sampler.infill([inputs], baseline_config, return_dict=True)
        latency = time.perf_counter() - start
        baseline_ids = baseline.sequences[0].tolist()
        baseline_metrics = compute_region_metrics(
            tokenizer, prediction_ids=baseline_ids, target_ids=targets
        )
        baseline_metrics["effective_steps"] = float(baseline.effective_steps)
        baseline_metrics["latency_sec"] = latency
        baseline_text = tokenizer.decode(baseline_ids, skip_special_tokens=False)
        target_text = tokenizer.decode(targets, skip_special_tokens=False)
        case_rows.append(
            {
                "sample_id": example.sample_id,
                "task_name": example.task_name,
                "instruction": example.instruction,
                "action_steps": example.action_steps,
                "serialized_actions": action_text,
                "target_text": target_text,
                "baseline_prediction": baseline_text,
                "baseline_metrics": baseline_metrics,
                "baseline_joint": baseline_metrics["joint_region_token_acc"],
                "baseline_preview": trim_text(baseline_text, limit=150),
            }
        )
    eval_seconds = time.perf_counter() - eval_start

    baseline_agg = aggregate_metric_rows([row["baseline_metrics"] for row in case_rows])
    task_summary = aggregate_task_metrics(
        case_rows, baseline_key="baseline_metrics", coordinated_key=None
    )
    save_grouped_metric_chart(
        baseline=baseline_agg,
        coordinated={key: 0.0 for key in baseline_agg},
        metric_keys=[
            "text_region_token_acc",
            "action_region_token_acc",
            "joint_region_token_acc",
            "joint_region_exact",
        ],
        output_path=figures_dir / "metric_compare.png",
        title="Baseline Metrics on Test Split",
    )
    save_task_delta_chart(
        task_summary=task_summary,
        metric_key="joint_region_token_acc",
        output_path=figures_dir / "task_joint_bar.png",
        title="Top Task Joint Accuracy (Baseline)",
    )
    highlights = choose_highlights(case_rows, limit=min(5, len(case_rows)))
    failures = choose_failures(case_rows, limit=min(5, len(case_rows)), use_coord=False)
    save_case_table_plot(
        rows=highlights,
        output_path=figures_dir / "case_highlights.png",
        title="Best Baseline Cases",
    )
    save_case_table_plot(
        rows=failures,
        output_path=figures_dir / "case_failures.png",
        title="Worst Baseline Cases",
    )

    config = with_experiment_metadata(config_to_dict(script_args))
    config.update(config_to_dict(sampler_config))
    config.update(
        {
            "experiment_dir": str(experiment_dir),
            "experiment_name": experiment_dir.name,
            "split_desc": f"固定 8:2 切分，train={len(train_examples)}，test={len(test_examples)}，seed={script_args.seed}",
            "split_manifest_path": str(experiment_dir / "split_manifest.json"),
            "sample_count": len(test_examples),
            "token_filter_desc": f"仅保留 target token <= {script_args.max_target_tokens} 的样本；保留 {filter_stats['kept']} 条，过滤 {filter_stats['dropped']} 条",
            "eval_duration_sec": eval_seconds,
        }
    )
    metrics = {
        "baseline": baseline_agg,
        "task_level": task_summary,
        "filter_stats": filter_stats,
        "split_manifest": split_manifest,
        "eval_duration_sec": eval_seconds,
    }
    dump_json(experiment_dir / "config.json", config)
    dump_json(experiment_dir / "metrics.json", metrics)
    dump_jsonl(experiment_dir / "cases.jsonl", case_rows)
    figures = {
        "metric_bar": figures_dir / "metric_compare.png",
        "task_bar": figures_dir / "task_joint_bar.png",
        "case_table": figures_dir / "case_highlights.png",
    }
    conclusion_lines = [
        "这份报告作为正式 baseline 参考线，后续所有 coordinated 结果都必须与它在同一 test split 上比较。",
        "如果 baseline 指标已经极高，coordination 的提升空间会被压缩；如果 baseline 很低，则更应关注 task-level 的稳定性而不是单点案例。",
        "正式结论以与 coordinated 的同 split 对比为准。",
    ]
    build_sample_readme(
        report_path=experiment_dir / "README.md",
        config=config,
        aggregate_baseline=baseline_agg,
        aggregate_coordinated=None,
        figures=figures,
        highlights=highlights,
        task_summary=task_summary,
        conclusion_lines=conclusion_lines,
        baseline_only=True,
    )
    print(f"baseline eval report written to {experiment_dir}")


if __name__ == "__main__":
    main()
