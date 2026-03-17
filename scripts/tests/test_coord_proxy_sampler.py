"""
Unit tests for the coordination proxy sampler helpers.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /data/ytw/VLA_baseline/dllm/scripts/tests/test_coord_proxy_sampler.py -v
"""

from types import SimpleNamespace

import torch

from dllm.core.samplers.coord_proxy import (
    _allocate_region_budgets,
    _project_vector,
    _repeat_coord_tokens,
    CoordinationModule,
    build_text_action_region_masks,
)


class _MockTokenizer:
    def __init__(self):
        self.vocab = {
            "[TEXT_START]": [11],
            "[TEXT_END]": [12],
            "[ACT_START]": [13],
            "[ACT_END]": [14],
        }

    def __call__(self, text, add_special_tokens=False):
        pieces = text.split()
        token_ids = []
        for piece in pieces:
            token_ids.extend(self.vocab.get(piece, [99]))
        return SimpleNamespace(input_ids=token_ids)


def test_build_text_action_region_masks_partitions_layout():
    tokenizer = _MockTokenizer()
    seq = [[1, 2, 11, 31, 32, 12, 13, 41, 42, 14]]
    text_mask, action_mask, prompt_mask = build_text_action_region_masks(
        tokenizer,
        seq,
        text_start_marker="[TEXT_START]",
        text_end_marker="[TEXT_END]",
        action_start_marker="[ACT_START]",
        action_end_marker="[ACT_END]",
    )
    assert prompt_mask[0, :2].all()
    assert text_mask[0, 3:5].all()
    assert action_mask[0, 7:9].all()
    assert not (text_mask & action_mask).any()


def test_coord_repeat_and_projection_shapes_are_stable():
    vector = torch.randn(2, 16)
    projected = _project_vector(vector, 8)
    repeated = _repeat_coord_tokens(projected, 4)
    assert projected.shape == (2, 8)
    assert repeated.shape == (2, 4, 8)


def test_allocate_region_budgets_respects_capacity():
    text_budget, action_budget, other_budget = _allocate_region_budgets(
        total_budget=5,
        text_count=2,
        action_count=10,
        other_count=1,
        text_weight=0.5,
        action_weight=2.0,
    )
    assert text_budget <= 2
    assert action_budget <= 10
    assert other_budget <= 1
    assert text_budget + action_budget + other_budget == 5


def test_allocate_region_budgets_handles_empty_regions():
    quotas = _allocate_region_budgets(
        total_budget=3,
        text_count=0,
        action_count=2,
        other_count=0,
        text_weight=1.0,
        action_weight=1.0,
    )
    assert quotas == (0, 2, 0)


def test_coordination_module_shapes_and_zero_init_bias():
    module = CoordinationModule(coord_hidden_size=8, coord_tokens=4)
    coord_state = torch.randn(2, 4, 8)
    prompt_summary = torch.randn(2, 8)
    text_summary = torch.randn(2, 8)
    action_summary = torch.randn(2, 8)
    next_state, text_bias, action_bias = module(
        coord_state=coord_state,
        prompt_summary=prompt_summary,
        text_summary=text_summary,
        action_summary=action_summary,
    )
    assert next_state.shape == (2, 4, 8)
    assert text_bias.shape == (2,)
    assert action_bias.shape == (2,)
    assert torch.allclose(text_bias, torch.zeros_like(text_bias))
    assert torch.allclose(action_bias, torch.zeros_like(action_bias))
