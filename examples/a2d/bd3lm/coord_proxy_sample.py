"""
Compare BD3LM-style proxy infilling with and without z_c coordination.

Run:
  python -u examples/a2d/bd3lm/coord_proxy_sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import transformers

import dllm


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
    text_target_ids = tokenizer(text_target, add_special_tokens=False).input_ids
    action_target_ids = tokenizer(action_target, add_special_tokens=False).input_ids

    prompt = (
        f"Instruction: {instruction}\n"
        f"{text_start_marker}"
        + (" " + tokenizer.mask_token) * len(text_target_ids)
        + f" {text_end_marker}\n"
        f"{action_start_marker}"
        + (" " + tokenizer.mask_token) * len(action_target_ids)
        + f" {action_end_marker}"
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids

    target = (
        f"Instruction: {instruction}\n"
        f"{text_start_marker} {text_target} {text_end_marker}\n"
        f"{action_start_marker} {action_target} {action_end_marker}"
    )
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    return prompt_ids, target_ids


def _report_metrics(tokenizer, prediction, target, label: str) -> dict[str, float]:
    pred_tokens = prediction.sequences[0].tolist()
    tgt_tokens = target
    shared = min(len(pred_tokens), len(tgt_tokens))
    token_acc = sum(int(a == b) for a, b in zip(pred_tokens[:shared], tgt_tokens[:shared])) / max(shared, 1)
    decoded_pred = tokenizer.decode(pred_tokens, skip_special_tokens=False)
    decoded_tgt = tokenizer.decode(tgt_tokens, skip_special_tokens=False)
    exact = float(pred_tokens[: len(tgt_tokens)] == tgt_tokens)
    print(f"\n[{label}] token_acc={token_acc:.3f} exact={exact:.1f} effective_steps={prediction.effective_steps}")
    print("prediction:")
    print(decoded_pred)
    print("target:")
    print(decoded_tgt)
    return {"token_acc": token_acc, "exact": exact}


@dataclass
class ScriptArguments:
    model_name_or_path: str = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
    seed: int = 42
    visualize: bool = False

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.core.samplers.CoordinationProxySamplerConfig):
    steps: int = 16
    few_step_budget: int = 4
    block_size: int | None = 32
    temperature: float = 0.0
    coord_tokens: int = 64
    text_transfer_ratio: float = 0.7
    action_transfer_ratio: float = 1.3
    coord_confidence_scale: float = 0.25


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))

def get_default_cases() -> list[tuple[str, str, str]]:
    return [
        (
            "Pick up the red block and place it on the tray.",
            "assistant: pick the red block and place it on the tray.",
            "ACT_PICK OBJ_RED OBJ_BLOCK ACT_PLACE OBJ_TRAY",
        ),
        (
            "Open the drawer and then press the green button.",
            "assistant: open the drawer and press the green button.",
            "ACT_OPEN OBJ_DRAWER ACT_PRESS OBJ_GREEN OBJ_BUTTON",
        ),
    ]


def main() -> None:
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    sampler = dllm.core.samplers.CoordinationProxySampler(
        model=model, tokenizer=tokenizer
    )
    terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

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

        baseline_config = SamplerConfig(**sampler_config.__dict__)
        baseline_config.enable_coordination = False

        start = time.perf_counter()
        baseline = sampler.infill(batch_inputs, baseline_config, return_dict=True)
        baseline_latency = time.perf_counter() - start

        start = time.perf_counter()
        coordinated = sampler.infill(batch_inputs, sampler_config, return_dict=True)
        coordinated_latency = time.perf_counter() - start

        print("\n" + "=" * 80)
        print(f"Case {idx}")
        print("=" * 80)
        print(
            f"baseline_latency={baseline_latency:.3f}s coordinated_latency={coordinated_latency:.3f}s"
        )
        _report_metrics(tokenizer, baseline, targets, label="baseline")
        _report_metrics(tokenizer, coordinated, targets, label="coordinated")

        if script_args.visualize:
            terminal_visualizer.visualize(
                coordinated.histories,
                rich=True,
                title=f"coord case {idx}",
            )


if __name__ == "__main__":
    main()
