"""
Train only the coordination module on CALVIN joint infill data and write a Chinese README report.

Run:
  cd /data/ytw/VLA_baseline/dllm
  CUDA_VISIBLE_DEVICES=0 /home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_fit.py --experiment_name coord_train_eval_formal
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers

import dllm

from calvin_joint_infill import (
    DATASET_PATH,
    OUTPUT_ROOT,
    aggregate_metric_rows,
    aggregate_task_metrics,
    build_case,
    build_fit_readme,
    choose_failures,
    choose_highlights,
    compute_region_metrics,
    config_to_dict,
    detect_device,
    dump_json,
    dump_jsonl,
    ensure_dir,
    export_split_manifest,
    filter_by_token_length,
    load_examples,
    make_experiment_dir,
    save_case_table_plot,
    save_curve,
    save_grouped_metric_chart,
    save_task_delta_chart,
    split_examples_by_ratio,
    trim_text,
    with_experiment_metadata,
    iso_now,
)


@dataclass
class ScriptArguments:
    model_name_or_path: str = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
    jsonl_path: str = str(DATASET_PATH)
    output_root: str = str(OUTPUT_ROOT)
    experiment_name: str | None = None
    train_ratio: float = 0.8
    max_target_tokens: int = 4096
    num_epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    round_digits: int = 4

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.core.samplers.CoordinationProxySamplerConfig):
    coord_tokens: int = 64
    coord_confidence_scale: float = 0.25
    few_step_budget: int = 8
    steps: int = 24
    block_size: int | None = None


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))


def _evaluate_cases(sampler, tokenizer, examples, sampler_config, round_digits):
    rows = []
    for example in examples:
        inputs, targets, _ = build_case(
            tokenizer=tokenizer,
            example=example,
            round_digits=round_digits,
        )

        baseline_config = SamplerConfig(**sampler_config.__dict__)
        baseline_config.enable_coordination = False
        start = time.perf_counter()
        baseline = sampler.infill([inputs], baseline_config, return_dict=True)
        baseline_latency = time.perf_counter() - start

        coordinated_config = SamplerConfig(**sampler_config.__dict__)
        coordinated_config.enable_coordination = True
        start = time.perf_counter()
        coordinated = sampler.infill([inputs], coordinated_config, return_dict=True)
        coordinated_latency = time.perf_counter() - start

        baseline_ids = baseline.sequences[0].tolist()
        coordinated_ids = coordinated.sequences[0].tolist()
        baseline_metrics = compute_region_metrics(
            tokenizer, prediction_ids=baseline_ids, target_ids=targets
        )
        coordinated_metrics = compute_region_metrics(
            tokenizer, prediction_ids=coordinated_ids, target_ids=targets
        )
        baseline_metrics["effective_steps"] = float(baseline.effective_steps)
        coordinated_metrics["effective_steps"] = float(coordinated.effective_steps)
        baseline_metrics["latency_sec"] = baseline_latency
        coordinated_metrics["latency_sec"] = coordinated_latency
        rows.append(
            {
                "sample_id": example.sample_id,
                "task_name": example.task_name,
                "instruction": example.instruction,
                "baseline_metrics": baseline_metrics,
                "coordinated_metrics": coordinated_metrics,
                "baseline_joint": baseline_metrics["joint_region_token_acc"],
                "coord_joint": coordinated_metrics["joint_region_token_acc"],
                "delta_joint": coordinated_metrics["joint_region_token_acc"]
                - baseline_metrics["joint_region_token_acc"],
                "baseline_preview": trim_text(
                    tokenizer.decode(baseline_ids, skip_special_tokens=False), limit=150
                ),
                "coordinated_preview": trim_text(
                    tokenizer.decode(coordinated_ids, skip_special_tokens=False), limit=150
                ),
            }
        )
    return rows


def main() -> None:
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)
    device = detect_device()

    experiment_dir = make_experiment_dir(
        prefix="coord_train_eval",
        experiment_name=script_args.experiment_name,
        output_root=Path(script_args.output_root),
    )
    figures_dir = ensure_dir(experiment_dir / "figures")

    model = dllm.utils.get_model(model_name_or_path=script_args.model_name_or_path).eval()
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    tokenizer = dllm.utils.get_tokenizer(model_name_or_path=script_args.model_name_or_path)
    sampler = dllm.core.samplers.CoordinationProxySampler(model=model, tokenizer=tokenizer)

    examples = load_examples(Path(script_args.jsonl_path))
    examples, filter_stats = filter_by_token_length(
        tokenizer,
        examples,
        round_digits=script_args.round_digits,
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

    coord_module = sampler.ensure_coordination_module(
        config=sampler_config,
        hidden_size=getattr(model.config, "hidden_size", 1024),
    )
    coord_module.train()
    optimizer = torch.optim.AdamW(coord_module.parameters(), lr=script_args.learning_rate)

    train_start_timestamp = iso_now()
    epoch_losses = []
    for epoch in range(script_args.num_epochs):
        epoch_loss = 0.0
        steps = 0
        for example in train_examples:
            prompt_ids, target_ids, _ = build_case(
                tokenizer=tokenizer,
                example=example,
                round_digits=script_args.round_digits,
            )
            inputs = [torch.tensor(prompt_ids, dtype=torch.long, device=model.device)]
            targets = [torch.tensor(target_ids, dtype=torch.long, device=model.device)]

            x, attention_mask, text_mask, action_mask, prompt_mask, _, _ = (
                sampler._prepare_proxy_batch(inputs, sampler_config)
            )
            target_x, _, _, _, _, _, _ = sampler._prepare_proxy_batch(
                targets, sampler_config
            )
            masked_positions = (x == tokenizer.mask_token_id) & (text_mask | action_mask)
            if not masked_positions.any():
                continue

            logits, hidden_states = sampler.forward_proxy_model(
                x=x,
                attention_mask=attention_mask,
            )
            coord_features = sampler.compute_coordination_features(
                hidden_states=hidden_states,
                text_mask=text_mask & attention_mask.bool(),
                action_mask=action_mask & attention_mask.bool(),
                prompt_mask=prompt_mask & attention_mask.bool(),
                config=sampler_config,
                enable_coordination=True,
            )
            biased_logits = logits.clone()
            text_delta_logits, action_delta_logits = sampler.compute_region_delta_logits(
                text_delta=coord_features["text_delta"],
                action_delta=coord_features["action_delta"],
                hidden_size=hidden_states.size(-1),
            )
            text_bias = coord_features["text_bias"].view(-1, 1, 1)
            action_bias = coord_features["action_bias"].view(-1, 1, 1)
            biased_logits = biased_logits + text_bias * text_mask.unsqueeze(-1).to(
                biased_logits.dtype
            )
            biased_logits = biased_logits + action_bias * action_mask.unsqueeze(-1).to(
                biased_logits.dtype
            )
            biased_logits = biased_logits + (
                text_mask.unsqueeze(-1).to(biased_logits.dtype)
                * text_delta_logits.unsqueeze(1).to(biased_logits.dtype)
            )
            biased_logits = biased_logits + (
                action_mask.unsqueeze(-1).to(biased_logits.dtype)
                * action_delta_logits.unsqueeze(1).to(biased_logits.dtype)
            )

            loss = torch.nn.functional.cross_entropy(
                biased_logits[masked_positions],
                target_x[masked_positions],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1

        epoch_losses.append(epoch_loss / max(steps, 1))
        print(f"epoch={epoch + 1} loss={epoch_losses[-1]:.6f}")
    train_end_timestamp = iso_now()

    coord_output_dir = ensure_dir(experiment_dir / "coordination_module")
    sampler.save_coordination_module(coord_output_dir)
    sampler.coordination_module.eval()

    eval_timestamp = iso_now()
    eval_rows = _evaluate_cases(
        sampler=sampler,
        tokenizer=tokenizer,
        examples=test_examples,
        sampler_config=sampler_config,
        round_digits=script_args.round_digits,
    )
    baseline_agg = aggregate_metric_rows([row["baseline_metrics"] for row in eval_rows])
    coordinated_agg = aggregate_metric_rows(
        [row["coordinated_metrics"] for row in eval_rows]
    )
    task_summary = aggregate_task_metrics(eval_rows)

    save_curve(
        epoch_losses,
        title="Coordination Module Training Loss",
        ylabel="loss",
        output_path=figures_dir / "loss_curve.png",
    )
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
        title="Test Metric Comparison",
    )
    save_task_delta_chart(
        task_summary=task_summary,
        metric_key="joint_region_token_acc",
        output_path=figures_dir / "task_joint_delta.png",
        title="Top Task Delta on Joint Accuracy",
    )
    highlights = choose_highlights(eval_rows, limit=min(5, len(eval_rows)))
    failures = choose_failures(eval_rows, limit=min(5, len(eval_rows)), use_coord=True)
    save_case_table_plot(
        rows=highlights,
        output_path=figures_dir / "case_highlights.png",
        title="Best Coordinated Cases",
    )
    save_case_table_plot(
        rows=failures,
        output_path=figures_dir / "case_failures.png",
        title="Worst Coordinated Cases",
    )

    config = with_experiment_metadata(config_to_dict(script_args))
    config.update(config_to_dict(sampler_config))
    config.update(
        {
            "experiment_dir": str(experiment_dir),
            "experiment_name": experiment_dir.name,
            "split_desc": f"固定 8:2 切分，train={len(train_examples)}，test={len(test_examples)}，seed={script_args.seed}",
            "split_manifest_path": str(experiment_dir / "split_manifest.json"),
            "token_filter_desc": f"仅保留 target token <= {script_args.max_target_tokens} 的样本；保留 {filter_stats['kept']} 条，过滤 {filter_stats['dropped']} 条",
            "train_start_timestamp": train_start_timestamp,
            "train_end_timestamp": train_end_timestamp,
            "eval_timestamp": eval_timestamp,
            "train_size": len(train_examples),
            "holdout_size": len(test_examples),
        }
    )
    training_summary = {
        "epochs_completed": script_args.num_epochs,
        "final_train_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "coord_module_output": str(coord_output_dir),
        "device": device,
    }
    metrics = {
        "baseline": baseline_agg,
        "coordinated": coordinated_agg,
        "task_level": task_summary,
        "epoch_losses": epoch_losses,
        "filter_stats": filter_stats,
        "split_manifest": split_manifest,
        "training_summary": training_summary,
    }

    dump_json(experiment_dir / "config.json", config)
    dump_json(experiment_dir / "metrics.json", metrics)
    dump_jsonl(experiment_dir / "cases.jsonl", eval_rows)

    figures = {
        "loss_curve": figures_dir / "loss_curve.png",
        "metric_bar": figures_dir / "metric_compare.png",
        "task_bar": figures_dir / "task_joint_delta.png",
    }
    conclusion_lines = [
        "如果训练后的 coordinated 在 test split 上优于 baseline，说明 coordination module 在真实 CALVIN 标注上具备学习价值。",
        "如果提升有限，可以先缩短 action 序列、调整四舍五入位数，或增加 coord_tokens/few_step_budget 再继续验证。",
        "这仍然是代理表示训练，不是最终离散 action token/block 方案。",
    ]
    build_fit_readme(
        report_path=experiment_dir / "README.md",
        config=config,
        training_summary=training_summary,
        aggregate_baseline=baseline_agg,
        aggregate_coordinated=coordinated_agg,
        figures=figures,
        highlights=highlights,
        task_summary=task_summary,
        conclusion_lines=conclusion_lines,
    )
    print(f"fit report written to {experiment_dir}")


if __name__ == "__main__":
    main()
