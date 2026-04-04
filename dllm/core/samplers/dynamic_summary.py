"""
Dynamic Summary Token sampler for training-free few-step diffusion acceleration.

At each denoising step, pools hidden states of already-revealed tokens into
summary vectors (one per region) and injects them as virtual prefix tokens
into the next step's forward pass via `inputs_embeds`.  This gives the model
explicit global context without introducing any new learnable parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.coord_proxy import (
    _pool_masked,
    build_text_action_region_masks,
)
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


@dataclass
class DynamicSummarySamplerConfig(BaseSamplerConfig):
    block_size: int | None = None
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    # Summary token parameters
    enable_summary: bool = True
    num_summary_tokens: int = 2  # 2 = one per region, 1 = merged
    summary_source: str = "hidden_states"  # "hidden_states" | "embeddings"
    summary_position_id: int = 0
    # Rollback remasking
    remask_ratio: float = 0.0  # 0.0 = disabled, 0.1 = re-mask 10% of fixed tokens per step
    # Region markers
    text_start_marker: str = "Assistant response:"
    text_end_marker: str = "Action sequence:"
    action_start_marker: str = "Action sequence:"
    action_end_marker: str = "End of plan."


@dataclass
class DynamicSummarySampler(BaseSampler):

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _extract_last_hidden_state(self, outputs) -> torch.Tensor:
        """Return the last-layer hidden state from model outputs."""
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            return hidden_states[-1]
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is not None:
            return last_hidden_state
        raise ValueError(
            "model outputs do not expose hidden_states or last_hidden_state"
        )

    def _forward_with_hidden(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normal forward pass that also returns the last hidden state."""
        outputs = self.model(
            x,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        logits = outputs.logits
        try:
            hidden = self._extract_last_hidden_state(outputs)
        except ValueError:
            # Fallback: try calling the base model directly (handles LoRA wrappers)
            base_model = getattr(self.model, "model", None)
            if base_model is None:
                raise
            base_out = base_model(
                input_ids=x,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = self._extract_last_hidden_state(base_out)
        return logits, hidden

    def _forward_with_summary(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        prev_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        action_mask: torch.Tensor,
        config: DynamicSummarySamplerConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with dynamic summary tokens prepended."""
        mask_id = self.tokenizer.mask_token_id
        B, T = x.shape
        num_s = config.num_summary_tokens
        device = x.device

        # 1. Identify revealed (non-mask, non-padding) positions
        revealed = (x != mask_id) & attention_mask.bool()

        # 2. Choose pooling source
        if config.summary_source == "embeddings":
            embed_layer = self.model.get_input_embeddings()
            source = embed_layer(x)
        else:
            source = prev_hidden

        # 3. Pool by region
        g_text = _pool_masked(source, revealed & text_mask)      # [B, D]
        g_action = _pool_masked(source, revealed & action_mask)  # [B, D]

        # 4. Construct summary token embeddings
        if num_s == 2:
            summary = torch.stack([g_text, g_action], dim=1)  # [B, 2, D]
        else:
            summary = ((g_text + g_action) / 2).unsqueeze(1)  # [B, 1, D]

        # 5. Build extended inputs_embeds
        embed_layer = self.model.get_input_embeddings()
        embeds = embed_layer(x)                                   # [B, T, D]
        embeds_ext = torch.cat([summary.to(embeds.dtype), embeds], dim=1)

        # 6. Extended attention mask
        attn_ext = torch.cat(
            [
                torch.ones(B, num_s, dtype=attention_mask.dtype, device=device),
                attention_mask,
            ],
            dim=1,
        )

        # 7. Position IDs (summary → fixed position, real tokens → 0..T-1)
        summary_pos = torch.full(
            (B, num_s), config.summary_position_id, dtype=torch.long, device=device
        )
        real_pos = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        pos_ids = torch.cat([summary_pos, real_pos], dim=1)

        # 8. Forward with inputs_embeds
        outputs = self.model(
            inputs_embeds=embeds_ext,
            attention_mask=attn_ext,
            position_ids=pos_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # 9. Slice off summary positions
        logits = outputs.logits[:, num_s:, :]
        try:
            hidden = self._extract_last_hidden_state(outputs)
        except ValueError:
            base_model = getattr(self.model, "model", None)
            if base_model is None:
                raise
            base_out = base_model(
                inputs_embeds=embeds_ext,
                attention_mask=attn_ext,
                position_ids=pos_ids,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = self._extract_last_hidden_state(base_out)
        hidden = hidden[:, num_s:, :]
        return logits, hidden

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: DynamicSummarySamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        return self.infill(inputs=inputs, config=config, **kwargs)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: DynamicSummarySamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fill in-place ``<|mdm_mask|>`` tokens with dynamic summary token injection.

        The loop mirrors ``MDLMSampler.infill`` with the addition of summary
        token construction and injection at each step after the first.
        """
        if config is None:
            config = DynamicSummarySamplerConfig()

        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict = kwargs.get("return_dict", config.return_dict)
        enable_summary = kwargs.get("enable_summary", config.enable_summary)
        remask_ratio = kwargs.get("remask_ratio", config.remask_ratio)

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Prepare canvas -----
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        if block_size is None:
            block_size = T
        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # ----- Region masks -----
        text_mask, action_mask, _prompt_mask = build_text_action_region_masks(
            self.tokenizer,
            [seq.tolist() for seq in inputs],
            text_start_marker=kwargs.get("text_start_marker", config.text_start_marker),
            text_end_marker=kwargs.get("text_end_marker", config.text_end_marker),
            action_start_marker=kwargs.get("action_start_marker", config.action_start_marker),
            action_end_marker=kwargs.get("action_end_marker", config.action_end_marker),
        )
        text_mask = text_mask.to(x.device)
        action_mask = action_mask.to(x.device)

        # ----- Block scheduling -----
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None
        prev_hidden: torch.Tensor | None = None

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                # ----- Forward pass (with or without summary) -----
                if prev_hidden is not None and enable_summary:
                    logits, hidden = self._forward_with_summary(
                        x, attention_mask, prev_hidden,
                        text_mask, action_mask, config,
                    )
                else:
                    logits, hidden = self._forward_with_hidden(x, attention_mask)
                prev_hidden = hidden

                # ----- Decode predictions -----
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # ----- Confidence scores -----
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Restrict to current block
                for j in range(B):
                    end_j = start + widths[j]
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                # ----- Top-k transfer -----
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                x[transfer_index] = x0[transfer_index]

                # ----- Rollback remasking: re-mask low-confidence fixed tokens -----
                if remask_ratio > 0.0 and s < effective_steps - 1:
                    infill_region = (text_mask | action_mask) & attention_mask.bool()
                    fixed_in_region = (x != mask_id) & infill_region
                    fixed_conf = torch.where(fixed_in_region, x0_p, torch.inf)

                    for j in range(B):
                        end_j = start + widths[j]
                        fixed_conf[j, :start] = torch.inf
                        fixed_conf[j, end_j:] = torch.inf

                        n_fixed = int(fixed_in_region[j, start:end_j].sum().item())
                        m = max(1, int(n_fixed * remask_ratio))
                        if n_fixed > m:
                            _, remask_idx = torch.topk(fixed_conf[j], k=m, largest=False)
                            x[j, remask_idx] = mask_id

                if histories is not None:
                    histories.append(x.clone())

        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)
