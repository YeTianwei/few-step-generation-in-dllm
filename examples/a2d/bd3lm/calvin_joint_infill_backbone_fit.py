"""
Train CALVIN joint infill adaptation baselines and write a Chinese README report.

Run:
  cd /data/ytw/VLA_baseline/dllm
  source ~/.zshrc
  conda activate ~/miniconda3/envs/dllm
  CUDA_VISIBLE_DEVICES=0 python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_backbone_fit.py --experiment_name backbone_lora_smoke --train_mode backbone_lora --max_train_samples 8 --max_eval_samples 4 --num_epochs 1
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers

import dllm

from calvin_joint_infill import (
    DATASET_PATH,
    OUTPUT_ROOT,
    aggregate_metric_rows,
    build_case,
    compute_region_metrics,
    config_to_dict,
    default_coord_module_path,
    detect_device,
    dump_json,
    dump_jsonl,
    ensure_dir,
    export_split_manifest,
    filter_by_token_length,
    iso_now,
    load_examples,
    make_experiment_dir,
    normalize_action_representation,
    split_examples_by_ratio,
    trim_text,
    with_experiment_metadata,
)


TRAIN_MODES = ("coord_only", "backbone_lora", "backbone_lora_plus_coord")


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
    lora: bool = False
    load_in_4bit: bool = False


@dataclass
class ExperimentArguments:
    jsonl_path: str = str(DATASET_PATH)
    output_root: str = str(OUTPUT_ROOT)
    experiment_name: str | None = None
    train_mode: str = "backbone_lora"
    train_ratio: float = 0.8
    max_target_tokens: int = 4096
    max_train_samples: int | None = 64
    max_eval_samples: int | None = 32
    num_epochs: int = 1
    learning_rate: float = 1e-4
    seed: int = 42
    round_digits: int = 4
    action_representation: str = "float_4dp"
    action_bucket_count: int = 8

    def __post_init__(self):
        if self.train_mode not in TRAIN_MODES:
            raise ValueError(f"train_mode must be one of {TRAIN_MODES}")
        self.action_representation = normalize_action_representation(
            self.action_representation
        )


@dataclass
class SamplerConfig(dllm.core.samplers.CoordinationProxySamplerConfig):
    coord_tokens: int = 64
    coord_confidence_scale: float = 0.25
    few_step_budget: int = 8
    steps: int = 24
    block_size: int | None = None


parser = transformers.HfArgumentParser((ModelArguments, ExperimentArguments, SamplerConfig))


def _clone_sampler_config(config: SamplerConfig) -> SamplerConfig:
    return SamplerConfig(**config.__dict__)


def _set_requires_grad(module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def _prepare_trainables(
    model,
    sampler,
    sampler_config: SamplerConfig,
    train_mode: str,
) -> list[torch.nn.Parameter]:
    trainables: list[torch.nn.Parameter] = []
    if train_mode == "coord_only":
        _set_requires_grad(model, False)
        coord_module = sampler.ensure_coordination_module(
            config=sampler_config,
            hidden_size=getattr(model.config, "hidden_size", 1024),
        )
        coord_module.train()
        _set_requires_grad(coord_module, True)
        trainables.extend(param for param in coord_module.parameters() if param.requires_grad)
        return trainables

    for param in model.parameters():
        if param.requires_grad:
            trainables.append(param)

    if train_mode == "backbone_lora_plus_coord":
        coord_module = sampler.ensure_coordination_module(
            config=sampler_config,
            hidden_size=getattr(model.config, "hidden_size", 1024),
        )
        coord_module.train()
        trainables.extend(param for param in coord_module.parameters() if param.requires_grad)
    return trainables


def _compute_biased_logits(
    sampler,
    tokenizer,
    x,
    attention_mask,
    text_mask,
    action_mask,
    prompt_mask,
    sampler_config: SamplerConfig,
    *,
    enable_coordination: bool,
):
    if not enable_coordination:
        outputs = sampler.model(
            x,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        return outputs.logits

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
    return biased_logits


def _compute_training_loss(
    sampler,
    tokenizer,
    prompt_ids: list[int],
    target_ids: list[int],
    sampler_config: SamplerConfig,
    *,
    enable_coordination: bool,
) -> torch.Tensor | None:
    inputs = [torch.tensor(prompt_ids, dtype=torch.long, device=sampler.model.device)]
    targets = [torch.tensor(target_ids, dtype=torch.long, device=sampler.model.device)]
    x, attention_mask, text_mask, action_mask, prompt_mask, _, _ = (
        sampler._prepare_proxy_batch(inputs, sampler_config)
    )
    target_x, _, _, _, _, _, _ = sampler._prepare_proxy_batch(targets, sampler_config)
    masked_positions = (x == tokenizer.mask_token_id) & (text_mask | action_mask)
    if not masked_positions.any():
        return None
    logits = _compute_biased_logits(
        sampler,
        tokenizer,
        x,
        attention_mask,
        text_mask,
        action_mask,
        prompt_mask,
        sampler_config,
        enable_coordination=enable_coordination,
    )
    return torch.nn.functional.cross_entropy(
        logits[masked_positions],
        target_x[masked_positions],
    )


def _run_infill(sampler, inputs, config):
    start = time.perf_counter()
    output = sampler.infill([inputs], config, return_dict=True)
    latency = time.perf_counter() - start
    return output, latency


def _evaluate_model(
    sampler,
    tokenizer,
    examples,
    sampler_config: SamplerConfig,
    *,
    action_representation: str,
    round_digits: int,
    action_bucket_count: int,
    enabled_coordination: bool,
):
    rows = []
    for example in examples:
        inputs, targets, action_text = build_case(
            tokenizer=tokenizer,
            example=example,
            action_representation=action_representation,
            round_digits=round_digits,
            action_bucket_count=action_bucket_count,
        )
        config = _clone_sampler_config(sampler_config)
        config.enable_coordination = enabled_coordination
        output, latency = _run_infill(sampler, inputs, config)
        prediction_ids = output.sequences[0].tolist()
        metrics = compute_region_metrics(
            tokenizer,
            prediction_ids=prediction_ids,
            target_ids=targets,
        )
        metrics["effective_steps"] = float(output.effective_steps)
        metrics["latency_sec"] = latency
        prediction_text = tokenizer.decode(prediction_ids, skip_special_tokens=False)
        target_text = tokenizer.decode(targets, skip_special_tokens=False)
        rows.append(
            {
                "sample_id": example.sample_id,
                "task_name": example.task_name,
                "instruction": example.instruction,
                "serialized_actions": action_text,
                "target_text": target_text,
                "prediction": prediction_text,
                "preview": trim_text(prediction_text, limit=160),
                "metrics": metrics,
            }
        )
    return rows


def _save_optional_artifacts(
    experiment_dir: Path,
    sampler,
    train_mode: str,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    if train_mode in {"backbone_lora", "backbone_lora_plus_coord"} and hasattr(
        sampler.model, "save_pretrained"
    ):
        adapter_dir = ensure_dir(experiment_dir / "backbone_adapter")
        sampler.model.save_pretrained(adapter_dir)
        paths["backbone_adapter"] = str(adapter_dir)
    if train_mode in {"coord_only", "backbone_lora_plus_coord"}:
        coord_dir = ensure_dir(experiment_dir / "coordination_module")
        sampler.save_coordination_module(coord_dir)
        paths["coordination_module"] = str(coord_dir)
    return paths


def _build_readme(
    *,
    report_path: Path,
    config: dict[str, Any],
    training_summary: dict[str, Any],
    baseline_metrics: dict[str, float],
    trained_metrics: dict[str, float],
    coordinated_metrics: dict[str, float] | None,
    highlights: list[dict[str, Any]],
) -> None:
    lines = [
        "# CALVIN Joint Infill Backbone 适配实验报告",
        "",
        f"- 实验日期：`{config['experiment_date']}`",
        f"- 训练开始时间：`{config['train_start_timestamp']}`",
        f"- 训练结束时间：`{config['train_end_timestamp']}`",
        "",
        "## 1. 实验目的",
        "本次实验用于比较 coord_only、backbone_lora 和 backbone_lora_plus_coord 三种适配方式，判断瓶颈更偏向 backbone 还是 coordination。",
        "",
        "## 2. 数据与配置",
        f"- 数据路径：`{config['jsonl_path']}`",
        f"- train_mode：`{config['train_mode']}`",
        f"- 训练样本：`{config['train_size']}`",
        f"- 测试样本：`{config['holdout_size']}`",
        f"- action 表示：`{config['action_representation']}`",
        f"- action_bucket_count：`{config['action_bucket_count']}`",
        f"- 输出目录：`{config['experiment_dir']}`",
        "",
        "## 3. 训练摘要",
        f"- 最终平均训练损失：`{training_summary['final_train_loss']:.6f}`",
        f"- 可训练参数量：`{training_summary['trainable_param_count']}`",
        f"- 保存产物：`{json.dumps(training_summary['artifacts'], ensure_ascii=False)}`",
        "",
        "## 4. 指标对比",
        "| metric | baseline | trained | coordinated |",
        "|---|---:|---:|---:|",
    ]
    metric_keys = [
        "text_region_token_acc",
        "action_region_token_acc",
        "joint_region_token_acc",
        "joint_region_exact",
        "effective_steps",
        "latency_sec",
    ]
    for key in metric_keys:
        coord_value = "-" if coordinated_metrics is None else f"{coordinated_metrics.get(key, 0.0):.4f}"
        lines.append(
            f"| {key} | {baseline_metrics.get(key, 0.0):.4f} | {trained_metrics.get(key, 0.0):.4f} | {coord_value} |"
        )
    lines.extend(
        [
            "",
            "## 5. 典型案例",
            "| sample_id | baseline_joint | trained_joint | coordinated_joint | baseline_preview |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in highlights:
        coord_value = row.get("coordinated_joint")
        coord_text = "-" if coord_value is None else f"{coord_value:.4f}"
        lines.append(
            f"| {row['sample_id']} | {row['baseline_joint']:.4f} | {row['trained_joint']:.4f} | {coord_text} | {row['baseline_preview']} |"
        )
    lines.extend(
        [
            "",
            "## 6. 结论",
            "- 如果 trained 明显优于 baseline，说明 backbone 适配比 coordination-only 更关键。",
            "- 如果 coordinated 继续优于 trained，说明在 backbone 已适配后 coordination 仍有增益空间。",
            "- 如果 trained 仍然接近 baseline，则更应优先降低动作表示难度而不是继续堆协调模块。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    model_args, experiment_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(experiment_args.seed)
    device = detect_device()

    if experiment_args.train_mode in {"backbone_lora", "backbone_lora_plus_coord"}:
        model_args.lora = True

    experiment_dir = make_experiment_dir(
        prefix="backbone_fit",
        experiment_name=experiment_args.experiment_name,
        output_root=Path(experiment_args.output_root),
    )

    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    examples = load_examples(Path(experiment_args.jsonl_path))
    examples, filter_stats = filter_by_token_length(
        tokenizer,
        examples,
        action_representation=experiment_args.action_representation,
        round_digits=experiment_args.round_digits,
        action_bucket_count=experiment_args.action_bucket_count,
        max_target_tokens=experiment_args.max_target_tokens,
    )
    train_examples, test_examples = split_examples_by_ratio(
        examples,
        train_ratio=experiment_args.train_ratio,
        seed=experiment_args.seed,
    )
    if experiment_args.max_train_samples is not None:
        train_examples = train_examples[: experiment_args.max_train_samples]
    if experiment_args.max_eval_samples is not None:
        test_examples = test_examples[: experiment_args.max_eval_samples]

    split_manifest = export_split_manifest(
        output_path=experiment_dir / "split_manifest.json",
        train_examples=train_examples,
        test_examples=test_examples,
        seed=experiment_args.seed,
        train_ratio=experiment_args.train_ratio,
    )

    model = dllm.utils.get_model(model_args=model_args).eval().to(device)
    sampler = dllm.core.samplers.CoordinationProxySampler(model=model, tokenizer=tokenizer)
    trainables = _prepare_trainables(model, sampler, sampler_config, experiment_args.train_mode)
    optimizer = torch.optim.AdamW(trainables, lr=experiment_args.learning_rate)

    train_start_timestamp = iso_now()
    epoch_losses: list[float] = []
    use_coordination_for_training = experiment_args.train_mode in {
        "coord_only",
        "backbone_lora_plus_coord",
    }
    for epoch in range(experiment_args.num_epochs):
        epoch_loss = 0.0
        steps = 0
        for example in train_examples:
            prompt_ids, target_ids, _ = build_case(
                tokenizer=tokenizer,
                example=example,
                action_representation=experiment_args.action_representation,
                round_digits=experiment_args.round_digits,
                action_bucket_count=experiment_args.action_bucket_count,
            )
            loss = _compute_training_loss(
                sampler,
                tokenizer,
                prompt_ids,
                target_ids,
                sampler_config,
                enable_coordination=use_coordination_for_training,
            )
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1
        epoch_losses.append(epoch_loss / max(steps, 1))
        print(f"epoch={epoch + 1} loss={epoch_losses[-1]:.6f}")
    train_end_timestamp = iso_now()

    artifacts = _save_optional_artifacts(experiment_dir, sampler, experiment_args.train_mode)

    baseline_model = dllm.utils.get_model(
        model_name_or_path=model_args.model_name_or_path
    ).eval().to(device)
    baseline_sampler = dllm.core.samplers.CoordinationProxySampler(
        model=baseline_model,
        tokenizer=tokenizer,
    )
    baseline_rows = _evaluate_model(
        baseline_sampler,
        tokenizer,
        test_examples,
        sampler_config,
        action_representation=experiment_args.action_representation,
        round_digits=experiment_args.round_digits,
        action_bucket_count=experiment_args.action_bucket_count,
        enabled_coordination=False,
    )
    trained_rows = _evaluate_model(
        sampler,
        tokenizer,
        test_examples,
        sampler_config,
        action_representation=experiment_args.action_representation,
        round_digits=experiment_args.round_digits,
        action_bucket_count=experiment_args.action_bucket_count,
        enabled_coordination=False,
    )
    coordinated_rows = None
    if experiment_args.train_mode in {"coord_only", "backbone_lora_plus_coord"}:
        coordinated_rows = _evaluate_model(
            sampler,
            tokenizer,
            test_examples,
            sampler_config,
            action_representation=experiment_args.action_representation,
            round_digits=experiment_args.round_digits,
            action_bucket_count=experiment_args.action_bucket_count,
            enabled_coordination=True,
        )

    baseline_metrics = aggregate_metric_rows([row["metrics"] for row in baseline_rows])
    trained_metrics = aggregate_metric_rows([row["metrics"] for row in trained_rows])
    coordinated_metrics = (
        aggregate_metric_rows([row["metrics"] for row in coordinated_rows])
        if coordinated_rows is not None
        else None
    )

    case_rows = []
    for index, baseline_row in enumerate(baseline_rows):
        trained_row = trained_rows[index]
        row = {
            "sample_id": baseline_row["sample_id"],
            "task_name": baseline_row["task_name"],
            "instruction": baseline_row["instruction"],
            "baseline_metrics": baseline_row["metrics"],
            "trained_metrics": trained_row["metrics"],
            "baseline_joint": baseline_row["metrics"]["joint_region_token_acc"],
            "trained_joint": trained_row["metrics"]["joint_region_token_acc"],
            "baseline_preview": baseline_row["preview"],
            "trained_preview": trained_row["preview"],
        }
        if coordinated_rows is not None:
            coord_row = coordinated_rows[index]
            row["coordinated_metrics"] = coord_row["metrics"]
            row["coordinated_joint"] = coord_row["metrics"]["joint_region_token_acc"]
            row["coordinated_preview"] = coord_row["preview"]
        case_rows.append(row)

    highlights = sorted(
        case_rows,
        key=lambda row: row["trained_joint"] - row["baseline_joint"],
        reverse=True,
    )[: min(5, len(case_rows))]

    config = with_experiment_metadata(config_to_dict(experiment_args))
    config.update(config_to_dict(model_args))
    config.update(config_to_dict(sampler_config))
    config.update(
        {
            "experiment_dir": str(experiment_dir),
            "experiment_name": experiment_dir.name,
            "train_start_timestamp": train_start_timestamp,
            "train_end_timestamp": train_end_timestamp,
            "train_size": len(train_examples),
            "holdout_size": len(test_examples),
            "split_manifest_path": str(experiment_dir / "split_manifest.json"),
        }
    )
    training_summary = {
        "epochs_completed": experiment_args.num_epochs,
        "final_train_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "trainable_param_count": int(sum(param.numel() for param in trainables)),
        "artifacts": artifacts,
        "device": device,
    }
    metrics = {
        "baseline": baseline_metrics,
        "trained": trained_metrics,
        "coordinated": coordinated_metrics,
        "epoch_losses": epoch_losses,
        "filter_stats": filter_stats,
        "split_manifest": split_manifest,
        "training_summary": training_summary,
    }

    dump_json(experiment_dir / "config.json", config)
    dump_json(experiment_dir / "metrics.json", metrics)
    dump_jsonl(experiment_dir / "cases.jsonl", case_rows)
    _build_readme(
        report_path=experiment_dir / "README.md",
        config=config,
        training_summary=training_summary,
        baseline_metrics=baseline_metrics,
        trained_metrics=trained_metrics,
        coordinated_metrics=coordinated_metrics,
        highlights=highlights,
    )
    print(f"backbone adaptation report written to {experiment_dir}")


if __name__ == "__main__":
    main()
