"""
Compare proxy infilling with and without Dynamic Summary Tokens.

Run:
  python -u examples/a2d/bd3lm/dynamic_summary_sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime

import transformers

import dllm
from dllm.core.samplers.coord_proxy import build_text_action_region_masks


def _make_proxy_case(
    tokenizer,
    instruction: str,
    text_target: str,
    action_target: str,
    text_start_marker: str,
    text_end_marker: str,
    action_start_marker: str,
    action_end_marker: str,
) -> tuple[list[int], list[int]]:
    target = (
        f"Instruction: {instruction}\n"
        f"{text_start_marker}{text_target} {text_end_marker}\n"
        f"{action_start_marker}{action_target} {action_end_marker}"
    )
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    text_mask, action_mask, _ = build_text_action_region_masks(
        tokenizer,
        [target_ids],
        text_start_marker=text_start_marker,
        text_end_marker=text_end_marker,
        action_start_marker=action_start_marker,
        action_end_marker=action_end_marker,
    )
    prompt_ids = list(target_ids)
    for idx, keep in enumerate((text_mask | action_mask)[0].tolist()):
        if keep:
            prompt_ids[idx] = tokenizer.mask_token_id
    return prompt_ids, target_ids


def _report_metrics(tokenizer, prediction, target, label: str) -> dict[str, float]:
    pred_tokens = prediction.sequences[0].tolist()
    tgt_tokens = target
    text_mask, action_mask, _ = build_text_action_region_masks(
        tokenizer,
        [tgt_tokens],
        text_start_marker=SamplerConfig.text_start_marker,
        text_end_marker=SamplerConfig.text_end_marker,
        action_start_marker=SamplerConfig.action_start_marker,
        action_end_marker=SamplerConfig.action_end_marker,
    )
    region_mask = (text_mask | action_mask)[0].tolist()

    # Overall metrics
    masked_pairs = [
        (pred, tgt)
        for pred, tgt, keep in zip(pred_tokens, tgt_tokens, region_mask)
        if keep
    ]
    token_acc = sum(int(a == b) for a, b in masked_pairs) / max(len(masked_pairs), 1)
    exact = float(masked_pairs and all(a == b for a, b in masked_pairs))

    # Per-region metrics
    text_pairs = [
        (pred, tgt)
        for pred, tgt, t, a in zip(
            pred_tokens, tgt_tokens, text_mask[0].tolist(), action_mask[0].tolist()
        )
        if t
    ]
    action_pairs = [
        (pred, tgt)
        for pred, tgt, t, a in zip(
            pred_tokens, tgt_tokens, text_mask[0].tolist(), action_mask[0].tolist()
        )
        if a
    ]
    text_acc = sum(int(a == b) for a, b in text_pairs) / max(len(text_pairs), 1)
    action_acc = sum(int(a == b) for a, b in action_pairs) / max(len(action_pairs), 1)

    decoded_pred = tokenizer.decode(pred_tokens, skip_special_tokens=False)
    decoded_tgt = tokenizer.decode(tgt_tokens, skip_special_tokens=False)

    print(
        f"\n[{label}] joint_acc={token_acc:.3f} text_acc={text_acc:.3f} "
        f"action_acc={action_acc:.3f} exact={exact:.1f}"
    )
    print("prediction:")
    print(decoded_pred)
    print("target:")
    print(decoded_tgt)
    return {
        "token_acc": token_acc,
        "text_acc": text_acc,
        "action_acc": action_acc,
        "exact": exact,
    }


@dataclass
class ScriptArguments:
    model_name_or_path: str = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
    seed: int = 42
    visualize: bool = False
    output_dir: str = "outputs/dynamic_summary"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
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


def get_default_cases() -> list[tuple[str, str, str]]:
    return [
        (
            "Pick up the red block and place it on the tray.",
            " I will pick up the red block and place it on the tray.",
            " reach for the red block, grasp it, move to the tray, and release it.",
        ),
        (
            "Open the drawer and then press the green button.",
            " I will open the drawer and then press the green button.",
            " pull the drawer open, move to the green button, and press it once.",
        ),
        (
            "Move the blue cup next to the kettle.",
            " I will move the blue cup so that it ends up next to the kettle.",
            " approach the blue cup, pick it up, move beside the kettle, and set it down.",
        ),
        (
            "Press the red switch after closing the lid.",
            " I will close the lid first and then press the red switch.",
            " close the lid, move to the red switch, and press it after the lid is shut.",
        ),
    ]


def main() -> None:
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    # --- Setup output directory ---
    run_name = (
        f"steps{sampler_config.steps}"
        f"_src-{sampler_config.summary_source}"
        f"_ntok{sampler_config.num_summary_tokens}"
        f"_seed{script_args.seed}"
    )
    run_dir = os.path.join(script_args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    sampler = dllm.core.samplers.DynamicSummarySampler(
        model=model, tokenizer=tokenizer
    )
    terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

    all_results = []

    for idx, (instruction, text_target, action_target) in enumerate(get_default_cases()):
        inputs, targets = _make_proxy_case(
            tokenizer=tokenizer,
            instruction=instruction,
            text_target=text_target,
            action_target=action_target,
            text_start_marker=sampler_config.text_start_marker,
            text_end_marker=sampler_config.text_end_marker,
            action_start_marker=sampler_config.action_start_marker,
            action_end_marker=sampler_config.action_end_marker,
        )
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

        print("\n" + "=" * 80)
        print(f"Case {idx}")
        print("=" * 80)
        print(
            f"baseline_latency={baseline_latency:.3f}s "
            f"summary_latency={summary_latency:.3f}s"
        )
        baseline_metrics = _report_metrics(tokenizer, baseline, targets, label="baseline")
        summary_metrics = _report_metrics(tokenizer, summary_result, targets, label="summary")

        all_results.append({
            "case_idx": idx,
            "instruction": instruction,
            "baseline": {
                **baseline_metrics,
                "latency": baseline_latency,
                "prediction": tokenizer.decode(
                    baseline.sequences[0].tolist(), skip_special_tokens=False
                ),
            },
            "summary": {
                **summary_metrics,
                "latency": summary_latency,
                "prediction": tokenizer.decode(
                    summary_result.sequences[0].tolist(), skip_special_tokens=False
                ),
            },
            "target": tokenizer.decode(targets, skip_special_tokens=False),
        })

        if script_args.visualize:
            terminal_visualizer.visualize(
                summary_result.histories,
                rich=True,
                title=f"summary case {idx}",
            )

    # --- Aggregate and save ---
    avg = lambda key, method: sum(r[method][key] for r in all_results) / len(all_results)
    summary_report = {
        "config": {
            "model": script_args.model_name_or_path,
            "steps": sampler_config.steps,
            "summary_source": sampler_config.summary_source,
            "num_summary_tokens": sampler_config.num_summary_tokens,
            "temperature": sampler_config.temperature,
            "seed": script_args.seed,
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

    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    # --- Print summary table ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for method in ("baseline", "summary"):
        m = summary_report["avg_metrics"][method]
        print(
            f"  [{method:>8s}] joint={m['joint_acc']:.3f}  text={m['text_acc']:.3f}  "
            f"action={m['action_acc']:.3f}  exact={m['exact']:.2f}  "
            f"latency={m['latency']:.3f}s"
        )
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
