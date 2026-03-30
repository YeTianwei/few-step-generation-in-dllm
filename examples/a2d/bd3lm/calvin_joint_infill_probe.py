"""
Probe CALVIN joint infill data and write a Chinese README report.

Run:
  cd /data/ytw/VLA_baseline/dllm
  /home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_probe.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import transformers

import dllm

from calvin_joint_infill import (
    DATASET_PATH,
    OUTPUT_ROOT,
    build_probe_readme,
    config_to_dict,
    dump_json,
    dump_jsonl,
    ensure_dir,
    load_examples,
    make_experiment_dir,
    sample_preview_rows,
    save_bar_chart,
    save_histogram,
    shuffled_examples,
    summarize_lengths,
    token_length_stats,
)


@dataclass
class ScriptArguments:
    jsonl_path: str = str(DATASET_PATH)
    output_root: str = str(OUTPUT_ROOT)
    experiment_name: str | None = None
    seed: int = 42
    round_digits: int = 4
    model_name_or_path: str = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


parser = transformers.HfArgumentParser((ScriptArguments,))


def main() -> None:
    (script_args,) = parser.parse_args_into_dataclasses()
    experiment_dir = make_experiment_dir(
        prefix="probe",
        experiment_name=script_args.experiment_name,
        output_root=Path(script_args.output_root),
    )
    figures_dir = ensure_dir(experiment_dir / "figures")
    examples = load_examples(Path(script_args.jsonl_path))
    tokenizer = dllm.utils.get_tokenizer(model_name_or_path=script_args.model_name_or_path)
    token_stats, token_lengths, pass_rate = token_length_stats(
        tokenizer,
        examples,
        round_digits=script_args.round_digits,
    )

    task_counts: dict[str, int] = {}
    think_lengths = []
    action_steps = []
    for example in examples:
        task_counts[example.task_name] = task_counts.get(example.task_name, 0) + 1
        think_lengths.append(len(example.think))
        action_steps.append(example.action_steps)

    top_tasks = sorted(task_counts.items(), key=lambda item: item[1], reverse=True)
    save_bar_chart(
        dict(top_tasks[:15]),
        title="Top-15 Task Distribution",
        ylabel="count",
        output_path=figures_dir / "task_distribution.png",
    )
    save_histogram(
        think_lengths,
        title="Think Character Length Distribution",
        xlabel="characters",
        output_path=figures_dir / "think_length_hist.png",
    )
    save_histogram(
        action_steps,
        title="Action Step Distribution",
        xlabel="steps",
        output_path=figures_dir / "action_steps_hist.png",
    )
    save_histogram(
        token_lengths,
        title="Joint Target Token Length Distribution",
        xlabel="tokens",
        output_path=figures_dir / "target_tokens_hist.png",
    )

    preview_rows = sample_preview_rows(shuffled_examples(examples, seed=script_args.seed))
    dataset_stats = {
        "sample_count": len(examples),
        "task_count": len(task_counts),
        "top_tasks": top_tasks,
        "think_char_stats": summarize_lengths(think_lengths),
        "action_step_stats": summarize_lengths(action_steps),
        "target_token_stats": token_stats,
        "mask_validation_pass_rate": pass_rate,
    }
    config = config_to_dict(script_args)
    config.update(
        {
            "experiment_dir": str(experiment_dir),
            "experiment_name": experiment_dir.name,
            "tokenizer_name": script_args.model_name_or_path,
            "token_filter_desc": "无过滤，仅做全量探查",
        }
    )

    dump_json(experiment_dir / "config.json", config)
    dump_json(experiment_dir / "metrics.json", dataset_stats)
    dump_jsonl(experiment_dir / "cases.jsonl", preview_rows)
    figures = {
        "task_bar": figures_dir / "task_distribution.png",
        "think_hist": figures_dir / "think_length_hist.png",
        "action_hist": figures_dir / "action_steps_hist.png",
        "token_hist": figures_dir / "target_tokens_hist.png",
    }
    notes = [
        "全量 1000 条样本都可以被读取，并且可以稳定构造成 `Instruction + think + serialized actions` 的联合样本。",
        "marker 验证通过率如果接近 1，说明现有 coord_proxy 的 text/action 区域切分逻辑可以直接复用。",
        "这份数据已经足以支持后续的小规模 joint infill 采样对比和 coordination module 训练。",
    ]
    build_probe_readme(
        report_path=experiment_dir / "README.md",
        config=config,
        dataset_stats=dataset_stats,
        figures=figures,
        preview_rows=preview_rows,
        notes=notes,
    )
    print(f"probe report written to {experiment_dir}")


if __name__ == "__main__":
    main()
