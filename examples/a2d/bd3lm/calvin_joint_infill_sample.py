"""
Run joint infill sampling on real CALVIN data and write a Chinese README report.

Run:
  cd /data/ytw/VLA_baseline/dllm
  CUDA_VISIBLE_DEVICES=0 /home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_sample.py --experiment_name sample_smoke
  CUDA_VISIBLE_DEVICES=0 /home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_sample.py --experiment_name sample_float2 --action_representation float_2dp
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
    choose_highlights,
    compute_region_metrics,
    config_to_dict,
    default_coord_module_path,
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
    shuffled_examples,
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
    sample_count: int = 10
    max_pool_size: int = 200
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


def _run_once(sampler, inputs, config):
    start = time.perf_counter()
    output = sampler.infill([inputs], config, return_dict=True)
    latency = time.perf_counter() - start
    return output, latency


def main() -> None:
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)
    if sampler_config.coord_module_path is None:
        sampler_config.coord_module_path = default_coord_module_path()

    experiment_dir = make_experiment_dir(
        prefix="sample",
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
    example_pool = shuffled_examples(examples, seed=script_args.seed)[: script_args.max_pool_size]
    chosen = example_pool[: script_args.sample_count]
    split_manifest = export_split_manifest(
        output_path=experiment_dir / "split_manifest.json",
        train_examples=example_pool[script_args.sample_count :],
        test_examples=chosen,
        seed=script_args.seed,
        train_ratio=max(0.0, 1.0 - (len(chosen) / max(len(example_pool), 1))),
    )

    case_rows = []
    for example in chosen:
        inputs, targets, action_text = build_case(
            tokenizer=tokenizer,
            example=example,
            action_representation=script_args.action_representation,
            round_digits=script_args.round_digits,
            action_bucket_count=script_args.action_bucket_count,
        )

        baseline_config = SamplerConfig(**sampler_config.__dict__)
        baseline_config.enable_coordination = False
        baseline_output, baseline_latency = _run_once(sampler, inputs, baseline_config)

        coordinated_config = SamplerConfig(**sampler_config.__dict__)
        coordinated_config.enable_coordination = True
        coordinated_output, coordinated_latency = _run_once(
            sampler, inputs, coordinated_config
        )

        baseline_ids = baseline_output.sequences[0].tolist()
        coordinated_ids = coordinated_output.sequences[0].tolist()
        baseline_metrics = compute_region_metrics(
            tokenizer, prediction_ids=baseline_ids, target_ids=targets
        )
        coordinated_metrics = compute_region_metrics(
            tokenizer, prediction_ids=coordinated_ids, target_ids=targets
        )
        baseline_metrics["effective_steps"] = float(baseline_output.effective_steps)
        coordinated_metrics["effective_steps"] = float(coordinated_output.effective_steps)
        baseline_metrics["latency_sec"] = baseline_latency
        coordinated_metrics["latency_sec"] = coordinated_latency

        target_text = tokenizer.decode(targets, skip_special_tokens=False)
        baseline_text = tokenizer.decode(baseline_ids, skip_special_tokens=False)
        coordinated_text = tokenizer.decode(coordinated_ids, skip_special_tokens=False)
        case_rows.append(
            {
                "sample_id": example.sample_id,
                "task_name": example.task_name,
                "instruction": example.instruction,
                "action_steps": example.action_steps,
                "serialized_actions": action_text,
                "target_text": target_text,
                "baseline_prediction": baseline_text,
                "coordinated_prediction": coordinated_text,
                "baseline_metrics": baseline_metrics,
                "coordinated_metrics": coordinated_metrics,
                "baseline_joint": baseline_metrics["joint_region_token_acc"],
                "coord_joint": coordinated_metrics["joint_region_token_acc"],
                "delta_joint": coordinated_metrics["joint_region_token_acc"]
                - baseline_metrics["joint_region_token_acc"],
                "baseline_preview": trim_text(baseline_text, limit=150),
                "coordinated_preview": trim_text(coordinated_text, limit=150),
                "target_preview": trim_text(target_text, limit=150),
            }
        )

    baseline_agg = aggregate_metric_rows([row["baseline_metrics"] for row in case_rows])
    coordinated_agg = aggregate_metric_rows(
        [row["coordinated_metrics"] for row in case_rows]
    )
    task_summary = aggregate_task_metrics(case_rows)

    save_grouped_metric_chart(
        baseline=baseline_agg,
        coordinated=coordinated_agg,
        metric_keys=[
            "text_region_token_acc",
            "action_region_token_acc",
            "joint_region_token_acc",
            "joint_region_exact",
        ],
        output_path=figures_dir / "metric_compare.png",
        title="Baseline vs Coordinated Metrics",
    )
    save_task_delta_chart(
        task_summary=task_summary,
        metric_key="joint_region_token_acc",
        output_path=figures_dir / "task_joint_delta.png",
        title="Top Task Delta on Joint Accuracy",
    )
    highlights = choose_highlights(case_rows, limit=min(5, len(case_rows)))
    save_case_table_plot(
        rows=highlights,
        output_path=figures_dir / "case_highlights.png",
        title="Cases with Largest Joint Accuracy Gain",
    )

    config = with_experiment_metadata(config_to_dict(script_args))
    config.update(config_to_dict(sampler_config))
    config.update(
        {
            "experiment_dir": str(experiment_dir),
            "experiment_name": experiment_dir.name,
            "split_desc": f"smoke 评估；候选池={len(example_pool)}，评估={len(chosen)}，seed={script_args.seed}",
            "split_manifest_path": str(experiment_dir / "split_manifest.json"),
            "token_filter_desc": f"仅保留 target token <= {script_args.max_target_tokens} 的样本；保留 {filter_stats['kept']} 条，过滤 {filter_stats['dropped']} 条",
        }
    )
    metrics = {
        "baseline": baseline_agg,
        "coordinated": coordinated_agg,
        "task_level": task_summary,
        "num_cases": len(case_rows),
        "filter_stats": filter_stats,
        "split_manifest": split_manifest,
    }
    dump_json(experiment_dir / "config.json", config)
    dump_json(experiment_dir / "metrics.json", metrics)
    dump_jsonl(experiment_dir / "cases.jsonl", case_rows)

    figures = {
        "metric_bar": figures_dir / "metric_compare.png",
        "case_table": figures_dir / "case_highlights.png",
        "task_bar": figures_dir / "task_joint_delta.png",
    }
    conclusion_lines = [
        "如果 `joint_region_token_acc` 或 `action_region_token_acc` 在 coordinated 下高于 baseline，说明真实 CALVIN 数据上已经出现了 text/action 协调恢复的信号。",
        "如果增益不稳定，优先检查样本长度、动作序列化粒度和 coord module 初始化来源，而不是直接否定 joint infill 思路。",
        "本轮仍然是代理表示验证；action 还没有进入最终离散 token/block 设计。",
    ]
    build_sample_readme(
        report_path=experiment_dir / "README.md",
        config=config,
        aggregate_baseline=baseline_agg,
        aggregate_coordinated=coordinated_agg,
        figures=figures,
        highlights=highlights,
        task_summary=task_summary,
        conclusion_lines=conclusion_lines,
    )
    print(f"sampling report written to {experiment_dir}")


if __name__ == "__main__":
    main()
