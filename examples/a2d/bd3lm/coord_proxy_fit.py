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
    few_step_budget: int = 8
    steps: int = 24
    block_size: int | None = None


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

objects = [
    ("red block", "tray"),
    ("blue cup", "kettle"),
    ("green sponge", "sink"),
    ("yellow bottle", "rack"),
]
actions = [
    (
        "Pick up the {obj} and place it near the {dst}.",
        " I will pick up the {obj} and place it near the {dst}.",
        " move to the {obj}, grasp it, carry it to the {dst}, and put it down.",
    ),
    (
        "Move the {obj} next to the {dst}.",
        " I will move the {obj} so it ends up next to the {dst}.",
        " approach the {obj}, pick it up, move beside the {dst}, and release it there.",
    ),
]
stateful = [
    (
        "Open the drawer and then press the green button.",
        " I will open the drawer and then press the green button.",
        " pull the drawer open, move to the green button, and press it once.",
    ),
    (
        "Close the lid before touching the red switch.",
        " I will close the lid before I touch the red switch.",
        " close the lid, move to the red switch, and press it after the lid is shut.",
    ),
]

cases = []
for obj, dst in objects:
    for instruction_fmt, text_fmt, action_fmt in actions:
        cases.append(
            (
                instruction_fmt.format(obj=obj, dst=dst),
                text_fmt.format(obj=obj, dst=dst),
                action_fmt.format(obj=obj, dst=dst),
            )
        )
cases.extend(stateful)

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
