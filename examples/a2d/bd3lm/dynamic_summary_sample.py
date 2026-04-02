"""
Compare proxy infilling with and without Dynamic Summary Tokens on CALVIN data.

Run:
  python -u examples/a2d/bd3lm/dynamic_summary_sample.py --steps 24
  python -u examples/a2d/bd3lm/dynamic_summary_sample.py --steps 24 --max_samples 50
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import transformers

import dllm
from dllm.core.samplers.coord_proxy import build_text_action_region_masks


# ---------------------------------------------------------------------------
# Constants (matching calvin_joint_infill.py)
# ---------------------------------------------------------------------------
TEXT_START_MARKER = "Assistant response:"
TEXT_END_MARKER = "Action sequence:"
ACTION_START_MARKER = "Action sequence:"
ACTION_END_MARKER = "End of plan."


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _bucket_action_value(value: float, bucket_count: int) -> int:
    clipped = max(-1.0, min(1.0, float(value)))
    scaled = (clipped + 1.0) / 2.0 * (bucket_count - 1)
    return int(round(scaled))


def serialize_actions_bucketed(
    actions: list[list[float]], bucket_count: int = 8
) -> str:
    steps = []
    for step in actions:
        values = ",".join(
            str(_bucket_action_value(v, bucket_count)) for v in step
        )
        steps.append(f"[{values}]")
    return "; ".join(steps)


def make_joint_target(
    instruction: str,
    think: str,
    serialized_actions: str,
) -> str:
    return (
        f"Instruction: {instruction}\n"
        f"{TEXT_START_MARKER}{think} {TEXT_END_MARKER}\n"
        f"{ACTION_START_MARKER}{serialized_actions} {ACTION_END_MARKER}"
    )


def load_calvin_examples(
    jsonl_path: str,
    max_samples: int | None = None,
    action_bucket_count: int = 8,
    split_manifest: str | None = None,
    eval_only: bool = False,
) -> list[dict]:
    # Load split filter if provided
    allowed_ids: set[str] | None = None
    if split_manifest:
        with open(split_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if eval_only:
            allowed_ids = set(manifest["test_sample_ids"])
        else:
            allowed_ids = set(manifest["train_sample_ids"])

    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if allowed_ids is not None and row["sample_id"] not in allowed_ids:
                continue
            serialized_actions = serialize_actions_bucketed(
                row["actions"], action_bucket_count
            )
            target_text = make_joint_target(
                instruction=row["instruction"],
                think=row["think"],
                serialized_actions=serialized_actions,
            )
            examples.append({
                "sample_id": row["sample_id"],
                "task_name": row["task_name"],
                "instruction": row["instruction"],
                "target_text": target_text,
            })
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


# ---------------------------------------------------------------------------
# Proxy case construction & metrics
# ---------------------------------------------------------------------------
def _make_proxy_case(
    tokenizer,
    target_text: str,
) -> tuple[list[int], list[int]]:
    target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
    text_mask, action_mask, _ = build_text_action_region_masks(
        tokenizer,
        [target_ids],
        text_start_marker=TEXT_START_MARKER,
        text_end_marker=TEXT_END_MARKER,
        action_start_marker=ACTION_START_MARKER,
        action_end_marker=ACTION_END_MARKER,
    )
    prompt_ids = list(target_ids)
    for idx, keep in enumerate((text_mask | action_mask)[0].tolist()):
        if keep:
            prompt_ids[idx] = tokenizer.mask_token_id
    return prompt_ids, target_ids


def _compute_metrics(
    tokenizer, pred_tokens: list[int], tgt_tokens: list[int]
) -> dict[str, float]:
    text_mask, action_mask, _ = build_text_action_region_masks(
        tokenizer,
        [tgt_tokens],
        text_start_marker=TEXT_START_MARKER,
        text_end_marker=TEXT_END_MARKER,
        action_start_marker=ACTION_START_MARKER,
        action_end_marker=ACTION_END_MARKER,
    )
    region_mask = (text_mask | action_mask)[0].tolist()

    masked_pairs = [
        (p, t) for p, t, keep in zip(pred_tokens, tgt_tokens, region_mask) if keep
    ]
    token_acc = sum(int(a == b) for a, b in masked_pairs) / max(len(masked_pairs), 1)
    exact = float(masked_pairs and all(a == b for a, b in masked_pairs))

    text_pairs = [
        (p, t) for p, t, m in zip(pred_tokens, tgt_tokens, text_mask[0].tolist()) if m
    ]
    action_pairs = [
        (p, t) for p, t, m in zip(pred_tokens, tgt_tokens, action_mask[0].tolist()) if m
    ]
    text_acc = sum(int(a == b) for a, b in text_pairs) / max(len(text_pairs), 1)
    action_acc = sum(int(a == b) for a, b in action_pairs) / max(len(action_pairs), 1)

    return {
        "token_acc": token_acc,
        "text_acc": text_acc,
        "action_acc": action_acc,
        "exact": exact,
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ScriptArguments:
    model_name_or_path: str = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
    adapter_path: str | None = None
    seed: int = 42
    output_dir: str = "outputs/dynamic_summary"
    data_path: str = ""
    max_samples: int | None = None
    action_bucket_count: int = 8
    split_manifest: str | None = None
    eval_only: bool = False

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )
        if not self.data_path:
            # Auto-resolve: look relative to repo root
            candidates = [
                Path(__file__).resolve().parents[3] / "task_ABC_D_batches" / "training.jsonl",
                Path("task_ABC_D_batches/training.jsonl"),
            ]
            for c in candidates:
                if c.exists():
                    self.data_path = str(c)
                    break
            if not self.data_path:
                raise FileNotFoundError(
                    "Cannot find training.jsonl. Specify --data_path explicitly."
                )


@dataclass
class SamplerConfig(dllm.core.samplers.DynamicSummarySamplerConfig):
    steps: int = 24
    block_size: int | None = None
    temperature: float = 0.0
    enable_summary: bool = True
    num_summary_tokens: int = 2
    summary_source: str = "hidden_states"


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    # --- Load data ---
    examples = load_calvin_examples(
        jsonl_path=script_args.data_path,
        max_samples=script_args.max_samples,
        action_bucket_count=script_args.action_bucket_count,
        split_manifest=script_args.split_manifest,
        eval_only=script_args.eval_only,
    )
    split_label = "eval" if script_args.eval_only else "all"
    print(f"Loaded {len(examples)} examples ({split_label}) from {script_args.data_path}")

    # --- Setup output directory ---
    summary_tag = (
        f"src-{sampler_config.summary_source}_ntok{sampler_config.num_summary_tokens}"
        if sampler_config.enable_summary
        else "off"
    )
    model_tag = "lora" if script_args.adapter_path else "base"
    run_name = (
        f"{model_tag}"
        f"_steps{sampler_config.steps}"
        f"_summary-{summary_tag}"
        f"_n{len(examples)}"
        f"_seed{script_args.seed}"
    )
    run_dir = os.path.join(script_args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # --- Load model ---
    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)

    if script_args.adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {script_args.adapter_path}")
        model = PeftModel.from_pretrained(model, script_args.adapter_path)
        model = model.merge_and_unload()
        model = model.eval()
        print("LoRA adapter merged.")
    sampler = dllm.core.samplers.DynamicSummarySampler(
        model=model, tokenizer=tokenizer
    )

    all_results = []
    skipped = 0

    for idx, example in enumerate(examples):
        try:
            inputs, targets = _make_proxy_case(
                tokenizer=tokenizer,
                target_text=example["target_text"],
            )
        except ValueError as e:
            skipped += 1
            print(f"  [skip] case {idx} ({example['sample_id']}): {e}")
            continue

        batch_inputs = [inputs]

        # --- Baseline: no summary ---
        baseline_config = SamplerConfig(**sampler_config.__dict__)
        baseline_config.enable_summary = False

        start = time.perf_counter()
        baseline = sampler.infill(batch_inputs, baseline_config, return_dict=True)
        baseline_latency = time.perf_counter() - start

        # --- With summary tokens ---
        start = time.perf_counter()
        summary_result = sampler.infill(batch_inputs, sampler_config, return_dict=True)
        summary_latency = time.perf_counter() - start

        baseline_pred = baseline.sequences[0].tolist()
        summary_pred = summary_result.sequences[0].tolist()
        baseline_metrics = _compute_metrics(tokenizer, baseline_pred, targets)
        summary_metrics = _compute_metrics(tokenizer, summary_pred, targets)

        all_results.append({
            "case_idx": idx,
            "sample_id": example["sample_id"],
            "task_name": example["task_name"],
            "instruction": example["instruction"],
            "baseline": {**baseline_metrics, "latency": baseline_latency},
            "summary": {**summary_metrics, "latency": summary_latency},
        })

        # Progress logging (every 10 cases or first/last)
        n_done = len(all_results)
        if n_done == 1 or n_done % 10 == 0 or idx == len(examples) - 1:
            print(
                f"  [{n_done:>4d}/{len(examples)}] {example['sample_id']:<40s}  "
                f"baseline_joint={baseline_metrics['token_acc']:.3f}  "
                f"summary_joint={summary_metrics['token_acc']:.3f}  "
                f"Δ={summary_metrics['token_acc'] - baseline_metrics['token_acc']:+.3f}"
            )

    if not all_results:
        print("No valid examples processed.")
        return

    # --- Aggregate ---
    def avg(key, method):
        return sum(r[method][key] for r in all_results) / len(all_results)

    summary_report = {
        "config": {
            "model": script_args.model_name_or_path,
            "adapter_path": script_args.adapter_path,
            "data_path": script_args.data_path,
            "steps": sampler_config.steps,
            "summary_source": sampler_config.summary_source,
            "num_summary_tokens": sampler_config.num_summary_tokens,
            "temperature": sampler_config.temperature,
            "seed": script_args.seed,
            "num_examples": len(all_results),
            "num_skipped": skipped,
            "action_bucket_count": script_args.action_bucket_count,
        },
        "avg_metrics": {
            "baseline": {
                "joint_acc": avg("token_acc", "baseline"),
                "text_acc": avg("text_acc", "baseline"),
                "action_acc": avg("action_acc", "baseline"),
                "exact": avg("exact", "baseline"),
                "latency": avg("latency", "baseline"),
            },
            "summary": {
                "joint_acc": avg("token_acc", "summary"),
                "text_acc": avg("text_acc", "summary"),
                "action_acc": avg("action_acc", "summary"),
                "exact": avg("exact", "summary"),
                "latency": avg("latency", "summary"),
            },
        },
        "per_case": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    # --- Save ---
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    # --- Print summary ---
    print("\n" + "=" * 80)
    print(f"SUMMARY  ({len(all_results)} examples, {skipped} skipped)")
    print("=" * 80)
    for method in ("baseline", "summary"):
        m = summary_report["avg_metrics"][method]
        print(
            f"  [{method:>8s}] joint={m['joint_acc']:.3f}  text={m['text_acc']:.3f}  "
            f"action={m['action_acc']:.3f}  exact={m['exact']:.2f}  "
            f"latency={m['latency']:.3f}s"
        )

    bm = summary_report["avg_metrics"]["baseline"]
    sm = summary_report["avg_metrics"]["summary"]
    print(
        f"\n  [  delta] joint={sm['joint_acc'] - bm['joint_acc']:+.3f}  "
        f"text={sm['text_acc'] - bm['text_acc']:+.3f}  "
        f"action={sm['action_acc'] - bm['action_acc']:+.3f}"
    )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
