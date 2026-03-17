"""
Proxy text+action coordination sampler for few-step diffusion infilling.

Run example:
  python -u examples/a2d/bd3lm/coord_proxy_sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


@dataclass
class CoordinationSamplerOutput(BaseSamplerOutput):
    effective_steps: int = 0


class CoordinationModule(nn.Module):
    """
    Lightweight trainable coordinator that predicts residual updates on top of the
    heuristic z_c pathway. Zero-init on the output heads keeps the initial
    behavior close to the previous heuristic sampler.
    """

    def __init__(self, coord_hidden_size: int, coord_tokens: int):
        super().__init__()
        self.coord_hidden_size = coord_hidden_size
        self.coord_tokens = coord_tokens
        feature_size = coord_hidden_size * 4

        self.input_norm = nn.LayerNorm(feature_size)
        self.mlp = nn.Sequential(
            nn.Linear(feature_size, coord_hidden_size),
            nn.Tanh(),
            nn.Linear(coord_hidden_size, coord_hidden_size),
            nn.Tanh(),
        )
        self.state_head = nn.Linear(coord_hidden_size, coord_hidden_size)
        self.bias_head = nn.Linear(coord_hidden_size, 2)
        self.text_delta_head = nn.Linear(coord_hidden_size, coord_hidden_size)
        self.action_delta_head = nn.Linear(coord_hidden_size, coord_hidden_size)

        nn.init.zeros_(self.state_head.weight)
        nn.init.zeros_(self.state_head.bias)
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)
        nn.init.zeros_(self.text_delta_head.weight)
        nn.init.zeros_(self.text_delta_head.bias)
        nn.init.zeros_(self.action_delta_head.weight)
        nn.init.zeros_(self.action_delta_head.bias)

    def forward(
        self,
        coord_state: torch.Tensor,
        prompt_summary: torch.Tensor,
        text_summary: torch.Tensor,
        action_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_dtype = coord_state.dtype
        prev_summary = coord_state.mean(dim=1)
        features = torch.cat(
            [prev_summary, prompt_summary, text_summary, action_summary], dim=-1
        )
        features = features.to(self.input_norm.weight.dtype)
        hidden = self.mlp(self.input_norm(features))
        prev_summary = prev_summary.to(hidden.dtype)
        next_summary = torch.tanh(prev_summary + self.state_head(hidden))
        bias_delta = self.bias_head(hidden)
        text_delta = self.text_delta_head(hidden)
        action_delta = self.action_delta_head(hidden)
        next_state = _repeat_coord_tokens(next_summary, self.coord_tokens)
        return (
            next_state.to(input_dtype),
            bias_delta[:, 0].to(input_dtype),
            bias_delta[:, 1].to(input_dtype),
            text_delta.to(input_dtype),
            action_delta.to(input_dtype),
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "coordination_module.pt")
        with (output_dir / "coordination_module_config.json").open("w") as f:
            json.dump(
                {
                    "coord_hidden_size": self.coord_hidden_size,
                    "coord_tokens": self.coord_tokens,
                },
                f,
                indent=2,
            )

    @classmethod
    def from_pretrained(cls, output_dir: str | Path) -> "CoordinationModule":
        output_dir = Path(output_dir)
        with (output_dir / "coordination_module_config.json").open() as f:
            config = json.load(f)
        module = cls(**config)
        state_dict = torch.load(
            output_dir / "coordination_module.pt",
            map_location="cpu",
            weights_only=False,
        )
        module.load_state_dict(state_dict)
        return module


def _find_subsequence(sequence: list[int], pattern: list[int]) -> tuple[int, int]:
    if not pattern:
        raise ValueError("pattern must be non-empty")
    limit = len(sequence) - len(pattern) + 1
    for start in range(max(0, limit)):
        if sequence[start : start + len(pattern)] == pattern:
            return start, start + len(pattern)
    raise ValueError(f"unable to find marker subsequence {pattern!r}")


def _tokenize_marker(tokenizer, marker: str) -> list[int]:
    token_ids = tokenizer(marker, add_special_tokens=False).input_ids
    if not token_ids:
        raise ValueError(f"marker {marker!r} tokenized to an empty sequence")
    return token_ids


def _find_marker_span(tokenizer, sequence: list[int], marker: str) -> tuple[int, int]:
    variants = [
        marker,
        f" {marker}",
        f"{marker}\n",
        f" {marker}\n",
    ]
    matches = []
    for variant in variants:
        token_ids = tokenizer(variant, add_special_tokens=False).input_ids
        if not token_ids:
            continue
        try:
            matches.append(_find_subsequence(sequence, token_ids))
        except ValueError:
            continue
    if not matches:
        raise ValueError(f"unable to find marker subsequence for {marker!r}")
    matches.sort(key=lambda span: (span[0], span[1] - span[0]))
    return matches[0]


def build_text_action_region_masks(
    tokenizer,
    sequences: list[list[int]],
    text_start_marker: str,
    text_end_marker: str,
    action_start_marker: str,
    action_end_marker: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    text_mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    action_mask = torch.zeros_like(text_mask)
    prompt_mask = torch.zeros_like(text_mask)

    for row, seq in enumerate(sequences):
        text_start, text_start_stop = _find_marker_span(
            tokenizer, seq, text_start_marker
        )
        text_end, _ = _find_marker_span(tokenizer, seq, text_end_marker)
        action_start, action_start_stop = _find_marker_span(
            tokenizer, seq, action_start_marker
        )
        action_end, _ = _find_marker_span(tokenizer, seq, action_end_marker)

        if not (text_start_stop <= text_end <= action_start <= action_start_stop <= action_end):
            raise ValueError("invalid proxy layout; expected context -> text -> action ordering")

        text_mask[row, text_start_stop:text_end] = True
        action_mask[row, action_start_stop:action_end] = True
        prompt_mask[row, :text_start] = True

    return text_mask, action_mask, prompt_mask


def _pool_masked(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden_states.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_states * weights).sum(dim=1) / denom


def _project_vector(vector: torch.Tensor, target_size: int) -> torch.Tensor:
    if vector.size(-1) == target_size:
        return vector
    return F.adaptive_avg_pool1d(vector.unsqueeze(1), target_size).squeeze(1)


def _repeat_coord_tokens(vector: torch.Tensor, coord_tokens: int) -> torch.Tensor:
    return vector.unsqueeze(1).expand(-1, coord_tokens, -1).contiguous()


def _allocate_region_budgets(
    total_budget: int,
    text_count: int,
    action_count: int,
    other_count: int,
    text_weight: float,
    action_weight: float,
    other_weight: float = 1.0,
) -> tuple[int, int, int]:
    if total_budget <= 0:
        return 0, 0, 0

    counts = [text_count, action_count, other_count]
    weights = [text_weight, action_weight, other_weight]
    scores = [c * max(w, 0.0) for c, w in zip(counts, weights)]

    if sum(counts) == 0:
        return 0, 0, 0
    if sum(scores) == 0:
        scores = [float(c) for c in counts]

    quotas = [0, 0, 0]
    fractions = []
    remaining = min(total_budget, sum(counts))
    total_score = sum(scores)
    for idx, (count, score) in enumerate(zip(counts, scores)):
        if count == 0 or total_score == 0:
            fractions.append((0.0, idx))
            continue
        raw = remaining * score / total_score
        quotas[idx] = min(count, int(math.floor(raw)))
        fractions.append((raw - quotas[idx], idx))

    leftover = remaining - sum(quotas)
    for _, idx in sorted(fractions, reverse=True):
        if leftover <= 0:
            break
        if quotas[idx] < counts[idx]:
            quotas[idx] += 1
            leftover -= 1

    if leftover > 0:
        for idx, count in enumerate(counts):
            if leftover <= 0:
                break
            room = count - quotas[idx]
            if room <= 0:
                continue
            take = min(room, leftover)
            quotas[idx] += take
            leftover -= take

    return tuple(quotas)


@dataclass
class CoordinationProxySamplerConfig(BaseSamplerConfig):
    block_size: int | None = 32
    steps: int = 16
    few_step_budget: int = 4
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    coord_tokens: int = 64
    coord_hidden_size: int | None = None
    coord_update: str = "mlp"
    text_action_layout: str = "serial_sections"
    text_transfer_ratio: float = 0.7
    action_transfer_ratio: float = 1.3
    coord_confidence_scale: float = 0.25
    early_stop_threshold: float = 0.9
    enable_coordination: bool = True
    coord_module_path: str | None = None
    text_start_marker: str = "Assistant response:"
    text_end_marker: str = "Action sequence:"
    action_start_marker: str = "Action sequence:"
    action_end_marker: str = "End of plan."


@dataclass
class CoordinationProxySampler(BaseSampler):
    coordination_module: CoordinationModule | None = None

    def _extract_last_hidden_state(self, outputs) -> torch.Tensor:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            return hidden_states[-1]
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is not None:
            return last_hidden_state
        raise ValueError(
            "model outputs do not expose hidden_states or last_hidden_state"
        )

    def forward_proxy_model(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(
            x,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        logits = outputs.logits
        try:
            hidden_states = self._extract_last_hidden_state(outputs)
        except ValueError:
            if not hasattr(self.model, "model"):
                raise
            base_outputs = self.model.model(
                input_ids=x,
                attention_mask=attention_mask,
                use_cache=False,
            )
            hidden_states = self._extract_last_hidden_state(base_outputs)
        return logits, hidden_states

    def _get_coord_hidden_size(
        self, config: CoordinationProxySamplerConfig, fallback_size: int
    ) -> int:
        if config.coord_hidden_size is not None:
            return config.coord_hidden_size
        hidden_size = getattr(self.model.config, "hidden_size", None)
        return hidden_size or fallback_size

    def ensure_coordination_module(
        self,
        config: CoordinationProxySamplerConfig,
        hidden_size: int,
    ) -> CoordinationModule:
        coord_hidden = self._get_coord_hidden_size(config, hidden_size)
        if self.coordination_module is None:
            if config.coord_module_path:
                self.coordination_module = CoordinationModule.from_pretrained(
                    config.coord_module_path
                )
            else:
                self.coordination_module = CoordinationModule(
                    coord_hidden_size=coord_hidden,
                    coord_tokens=config.coord_tokens,
                )
        self.coordination_module = self.coordination_module.to(self.model.device)
        return self.coordination_module

    def save_coordination_module(self, output_dir: str | Path) -> None:
        if self.coordination_module is None:
            raise ValueError("coordination_module has not been initialized")
        self.coordination_module.save_pretrained(output_dir)

    def _update_coord_state(
        self,
        coord_state: torch.Tensor,
        prompt_summary: torch.Tensor,
        text_summary: torch.Tensor,
        action_summary: torch.Tensor,
        config: CoordinationProxySamplerConfig,
        coordination_module: CoordinationModule | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        combined = torch.stack(
            [prompt_summary, text_summary, action_summary],
            dim=1,
        ).mean(dim=1)
        if config.coord_update == "mlp":
            heuristic_state = torch.tanh(coord_state.mean(dim=1) + combined)
        elif config.coord_update == "transformer":
            heuristic_state = torch.tanh(
                0.5 * coord_state.mean(dim=1) + 0.5 * combined
            )
        else:
            raise NotImplementedError(config.coord_update)
        heuristic_state = _repeat_coord_tokens(heuristic_state, config.coord_tokens)
        if coordination_module is None:
            zero = prompt_summary.new_zeros(prompt_summary.size(0))
            zero_delta = prompt_summary.new_zeros(prompt_summary.shape)
            return heuristic_state, zero, zero, zero_delta, zero_delta
        (
            learned_state,
            learned_text_bias,
            learned_action_bias,
            text_delta,
            action_delta,
        ) = coordination_module(
            coord_state=heuristic_state,
            prompt_summary=prompt_summary,
            text_summary=text_summary,
            action_summary=action_summary,
        )
        return (
            learned_state,
            learned_text_bias,
            learned_action_bias,
            text_delta,
            action_delta,
        )

    def _region_transfer(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        confidence: torch.Tensor,
        current_mask: torch.Tensor,
        region_mask: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        transfer = torch.zeros_like(current_mask)
        if budget <= 0:
            return transfer
        candidate_mask = current_mask & region_mask
        valid = candidate_mask.sum().item()
        if valid == 0:
            return transfer
        budget = min(budget, valid)
        scores = torch.where(candidate_mask, confidence, torch.full_like(confidence, -torch.inf))
        _, idx = torch.topk(scores, k=budget)
        transfer[idx] = True
        return transfer

    def _prepare_proxy_batch(
        self,
        inputs: list[torch.Tensor],
        config: CoordinationProxySamplerConfig,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], int]:
        eos_id = self.tokenizer.eos_token_id
        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)
        block_size = kwargs.get("block_size", config.block_size)
        if block_size is None:
            block_size = T

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, tokens in enumerate(inputs):
            x[i, : seq_lens[i]] = tokens

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, seq_len in enumerate(seq_lens):
            attention_mask[i, :seq_len] = 1

        text_mask, action_mask, prompt_mask = build_text_action_region_masks(
            self.tokenizer,
            [seq.tolist() for seq in inputs],
            text_start_marker=kwargs.get("text_start_marker", config.text_start_marker),
            text_end_marker=kwargs.get("text_end_marker", config.text_end_marker),
            action_start_marker=kwargs.get(
                "action_start_marker", config.action_start_marker
            ),
            action_end_marker=kwargs.get("action_end_marker", config.action_end_marker),
        )
        return (
            x,
            attention_mask,
            text_mask.to(x.device),
            action_mask.to(x.device),
            prompt_mask.to(x.device),
            seq_lens,
            block_size,
        )

    def compute_coordination_features(
        self,
        hidden_states: torch.Tensor,
        text_mask: torch.Tensor,
        action_mask: torch.Tensor,
        prompt_mask: torch.Tensor,
        config: CoordinationProxySamplerConfig,
        coord_state: torch.Tensor | None = None,
        enable_coordination: bool = True,
    ) -> dict[str, torch.Tensor]:
        hidden_dim = hidden_states.size(-1)
        coord_hidden = self._get_coord_hidden_size(config, hidden_dim)
        prompt_summary = _project_vector(
            _pool_masked(hidden_states, prompt_mask), coord_hidden
        )
        text_summary = _project_vector(_pool_masked(hidden_states, text_mask), coord_hidden)
        action_summary = _project_vector(
            _pool_masked(hidden_states, action_mask), coord_hidden
        )
        if coord_state is None:
            coord_state = _repeat_coord_tokens(prompt_summary, config.coord_tokens)

        coordination_module = None
        if enable_coordination:
            coordination_module = self.ensure_coordination_module(
                config=config, hidden_size=hidden_dim
            )

        (
            coord_state,
            learned_text_bias,
            learned_action_bias,
            text_delta,
            action_delta,
        ) = self._update_coord_state(
            coord_state=coord_state,
            prompt_summary=prompt_summary,
            text_summary=text_summary,
            action_summary=action_summary,
            config=config,
            coordination_module=coordination_module,
        )

        coord_summary = coord_state.mean(dim=1)
        text_alignment = F.cosine_similarity(
            coord_summary, text_summary, dim=-1
        ).nan_to_num(0.0)
        action_alignment = F.cosine_similarity(
            coord_summary, action_summary, dim=-1
        ).nan_to_num(0.0)
        cross_alignment = F.cosine_similarity(
            text_summary, action_summary, dim=-1
        ).nan_to_num(0.0)

        heuristic_text_bias = config.coord_confidence_scale * (
            text_alignment + cross_alignment
        )
        heuristic_action_bias = config.coord_confidence_scale * (
            action_alignment + cross_alignment
        )

        if enable_coordination:
            text_bias = heuristic_text_bias + learned_text_bias
            action_bias = heuristic_action_bias + learned_action_bias
        else:
            text_bias = prompt_summary.new_zeros(prompt_summary.size(0))
            action_bias = prompt_summary.new_zeros(prompt_summary.size(0))

        return {
            "coord_state": coord_state,
            "prompt_summary": prompt_summary,
            "text_summary": text_summary,
            "action_summary": action_summary,
            "text_alignment": text_alignment,
            "action_alignment": action_alignment,
            "cross_alignment": cross_alignment,
            "text_bias": text_bias,
            "action_bias": action_bias,
            "heuristic_text_bias": heuristic_text_bias,
            "heuristic_action_bias": heuristic_action_bias,
            "text_delta": text_delta,
            "action_delta": action_delta,
        }

    def compute_region_delta_logits(
        self,
        text_delta: torch.Tensor,
        action_delta: torch.Tensor,
        hidden_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is None:
            raise ValueError("model does not expose output embeddings")
        lm_weight = output_embeddings.weight
        text_hidden = _project_vector(text_delta, hidden_size).to(lm_weight.dtype)
        action_hidden = _project_vector(action_delta, hidden_size).to(lm_weight.dtype)
        text_delta_logits = torch.matmul(text_hidden, lm_weight.transpose(0, 1))
        action_delta_logits = torch.matmul(action_hidden, lm_weight.transpose(0, 1))
        return text_delta_logits, action_delta_logits

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: CoordinationProxySamplerConfig | None = None,
        **kwargs,
    ) -> CoordinationSamplerOutput | torch.Tensor:
        return self.infill(inputs=inputs, config=config, **kwargs)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: CoordinationProxySamplerConfig | None = None,
        **kwargs,
    ) -> CoordinationSamplerOutput | torch.Tensor:
        config = config or CoordinationProxySamplerConfig()
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        few_step_budget = kwargs.get("few_step_budget", config.few_step_budget)
        temperature = kwargs.get("temperature", config.temperature)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        enable_coordination = kwargs.get(
            "enable_coordination", config.enable_coordination
        )

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        x, attention_mask, text_mask, action_mask, prompt_mask, seq_lens, block_size = (
            self._prepare_proxy_batch(inputs, config, **kwargs)
        )
        B, T = x.shape

        histories = [x.clone()] if return_dict else None
        num_blocks = math.ceil(T / block_size)
        steps_per_block = min(max(1, few_step_budget), max(1, math.ceil(steps / num_blocks)))
        effective_steps_total = 0
        coord_state = None

        for block_idx in range(num_blocks):
            start = block_idx * block_size
            stop = min(start + block_size, T)
            widths = [max(0, min(seq_lens[j], stop) - start) for j in range(B)]
            block_mask_index = torch.zeros((B, block_size), dtype=torch.bool, device=x.device)
            for j, width in enumerate(widths):
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            for step_idx in range(effective_steps):
                block_region = torch.zeros_like(x, dtype=torch.bool)
                for j, width in enumerate(widths):
                    if width > 0:
                        block_region[j, start : start + width] = True

                current_mask = (x == mask_id) & block_region
                if not current_mask.any():
                    break

                logits, hidden_states = self.forward_proxy_model(
                    x=x,
                    attention_mask=attention_mask,
                )

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    probs = F.softmax(logits, dim=-1)
                    confidence = torch.gather(
                        probs, dim=-1, index=x0.unsqueeze(-1)
                    ).squeeze(-1)
                elif remasking == "random":
                    confidence = torch.rand((B, T), device=x.device)
                else:
                    raise NotImplementedError(remasking)

                x0 = torch.where(current_mask, x0, x)
                confidence = torch.where(
                    current_mask, confidence, torch.full_like(confidence, -torch.inf)
                )

                coord_features = self.compute_coordination_features(
                    hidden_states=hidden_states,
                    text_mask=text_mask & attention_mask.bool(),
                    action_mask=action_mask & attention_mask.bool(),
                    prompt_mask=prompt_mask & attention_mask.bool(),
                    config=config,
                    coord_state=coord_state,
                    enable_coordination=enable_coordination,
                )
                coord_state = coord_features["coord_state"]
                text_bias = coord_features["text_bias"]
                action_bias = coord_features["action_bias"]
                if enable_coordination:
                    text_delta_logits, action_delta_logits = self.compute_region_delta_logits(
                        text_delta=coord_features["text_delta"],
                        action_delta=coord_features["action_delta"],
                        hidden_size=hidden_states.size(-1),
                    )
                    logits = logits + (
                        text_mask.unsqueeze(-1).to(logits.dtype)
                        * text_delta_logits.unsqueeze(1).to(logits.dtype)
                    )
                    logits = logits + (
                        action_mask.unsqueeze(-1).to(logits.dtype)
                        * action_delta_logits.unsqueeze(1).to(logits.dtype)
                    )

                text_region = current_mask & text_mask
                action_region = current_mask & action_mask
                other_region = current_mask & ~(text_mask | action_mask)

                transfer_index = torch.zeros_like(current_mask)
                for row in range(B):
                    total_budget = int(num_transfer_tokens[row, step_idx].item())
                    total_budget = min(total_budget, int(current_mask[row].sum().item()))
                    text_weight = config.text_transfer_ratio * (
                        1.0 + max(0.0, float(text_bias[row].item()))
                    )
                    action_weight = config.action_transfer_ratio * (
                        1.0 + max(0.0, float(action_bias[row].item()))
                    )
                    text_budget, action_budget, other_budget = _allocate_region_budgets(
                        total_budget=total_budget,
                        text_count=int(text_region[row].sum().item()),
                        action_count=int(action_region[row].sum().item()),
                        other_count=int(other_region[row].sum().item()),
                        text_weight=text_weight,
                        action_weight=action_weight,
                    )

                    row_confidence = confidence[row].clone()
                    if enable_coordination:
                        row_confidence[text_mask[row]] += text_bias[row]
                        row_confidence[action_mask[row]] += action_bias[row]

                    row_transfer = torch.zeros_like(current_mask[row])
                    row_transfer |= self._region_transfer(
                        x=x[row],
                        x0=x0[row],
                        confidence=row_confidence,
                        current_mask=current_mask[row],
                        region_mask=text_mask[row],
                        budget=text_budget,
                    )
                    row_transfer |= self._region_transfer(
                        x=x[row],
                        x0=x0[row],
                        confidence=row_confidence,
                        current_mask=current_mask[row],
                        region_mask=action_mask[row],
                        budget=action_budget,
                    )
                    row_transfer |= self._region_transfer(
                        x=x[row],
                        x0=x0[row],
                        confidence=row_confidence,
                        current_mask=current_mask[row],
                        region_mask=~(text_mask[row] | action_mask[row]),
                        budget=other_budget,
                    )

                    missing = total_budget - int(row_transfer.sum().item())
                    if missing > 0:
                        residual = current_mask[row] & ~row_transfer
                        if residual.any():
                            residual_scores = torch.where(
                                residual,
                                row_confidence,
                                torch.full_like(row_confidence, -torch.inf),
                            )
                            _, residual_idx = torch.topk(
                                residual_scores, k=min(missing, int(residual.sum().item()))
                            )
                            row_transfer[residual_idx] = True

                    if (
                        config.early_stop_threshold is not None
                        and config.early_stop_threshold > 0
                    ):
                        row_scores = torch.where(
                            current_mask[row],
                            row_confidence,
                            torch.full_like(row_confidence, float("inf")),
                        )
                        if current_mask[row].any() and torch.all(
                            row_scores[current_mask[row]] >= config.early_stop_threshold
                        ):
                            row_transfer[current_mask[row]] = True

                    transfer_index[row] = row_transfer

                x[transfer_index] = x0[transfer_index]
                effective_steps_total += 1
                if histories is not None:
                    histories.append(x.clone())

        if not return_dict:
            return x
        return CoordinationSamplerOutput(
            sequences=x,
            histories=histories,
            effective_steps=effective_steps_total,
        )
