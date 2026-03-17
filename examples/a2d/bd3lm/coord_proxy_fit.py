"""
Train only the coordination module for the proxy text+action infilling task.

Run:
  python -u examples/a2d/bd3lm/coord_proxy_fit.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
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


@dataclass
class ScriptArguments:
    model_name_or_path: str = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
    output_dir: str = ".models/a2d/proxy-coordination"
    num_epochs: int = 20
    learning_rate: float = 1e-3
    seed: int = 42

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.core.samplers.CoordinationProxySamplerConfig):
    coord_tokens: int = 64
    coord_confidence_scale: float = 0.25
    few_step_budget: int = 4


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = dllm.utils.get_model(model_args=script_args).eval().to(device)
for param in model.parameters():
    param.requires_grad_(False)

tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.core.samplers.CoordinationProxySampler(model=model, tokenizer=tokenizer)
coord_module = sampler.ensure_coordination_module(
    config=sampler_config,
    hidden_size=getattr(model.config, "hidden_size", 1024),
)
coord_module.train()

optimizer = torch.optim.AdamW(coord_module.parameters(), lr=script_args.learning_rate)

cases = [
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

prepared = []
for instruction, text_target, action_target in cases:
    prompt_ids, target_ids = _make_proxy_case(
        tokenizer=tokenizer,
        instruction=instruction,
        text_target=text_target,
        action_target=action_target,
        text_start_marker=sampler_config.text_start_marker,
        text_end_marker=sampler_config.text_end_marker,
        action_start_marker=sampler_config.action_start_marker,
        action_end_marker=sampler_config.action_end_marker,
    )
    prepared.append((prompt_ids, target_ids))

for epoch in range(script_args.num_epochs):
    epoch_loss = 0.0
    for prompt_ids, target_ids in prepared:
        inputs = [torch.tensor(prompt_ids, dtype=torch.long, device=model.device)]
        targets = [torch.tensor(target_ids, dtype=torch.long, device=model.device)]

        x, attention_mask, text_mask, action_mask, prompt_mask, _, _ = (
            sampler._prepare_proxy_batch(inputs, sampler_config)
        )
        target_x, _, _, _, _, _, _ = sampler._prepare_proxy_batch(targets, sampler_config)
        masked_positions = (x == tokenizer.mask_token_id) & (text_mask | action_mask)

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
        text_bias = coord_features["text_bias"].view(-1, 1, 1)
        action_bias = coord_features["action_bias"].view(-1, 1, 1)
        biased_logits = biased_logits + text_bias * text_mask.unsqueeze(-1).to(
            biased_logits.dtype
        )
        biased_logits = biased_logits + action_bias * action_mask.unsqueeze(-1).to(
            biased_logits.dtype
        )

        if not masked_positions.any():
            continue
        loss = torch.nn.functional.cross_entropy(
            biased_logits[masked_positions],
            target_x[masked_positions],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.item())

    print(f"epoch={epoch} loss={epoch_loss / max(len(prepared), 1):.6f}")

os.makedirs(script_args.output_dir, exist_ok=True)
sampler.save_coordination_module(script_args.output_dir)
print(f"saved coordination module to {script_args.output_dir}")
