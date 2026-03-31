"""
Unit tests for CALVIN joint infill helpers.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /data/ytw/VLA_baseline/dllm/scripts/tests/test_calvin_joint_infill.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT / "examples" / "a2d" / "bd3lm"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from calvin_joint_infill import (  # noqa: E402
    ACTION_REPRESENTATIONS,
    CalvinJointInfillExample,
    build_case,
    filter_by_token_length,
    normalize_action_representation,
    serialize_actions_with_representation,
    token_length_stats,
)


class _CharTokenizer:
    def __init__(self):
        self.mask_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(ch) for ch in text])

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(idx) for idx in ids if idx != self.mask_token_id)


def _make_example() -> CalvinJointInfillExample:
    return CalvinJointInfillExample(
        sample_id="sample-1",
        task_name="open_drawer",
        instruction="open the drawer",
        think="I will open the drawer and then place the block inside.",
        actions=[
            [0.35, -0.29, 0.10, -0.09, 0.03, 0.03, 1.0],
            [0.10, -0.15, 0.12, -0.05, 0.02, 0.04, 0.0],
            [-0.20, 0.25, -0.10, 0.05, -0.02, -0.03, 0.0],
        ],
        action_steps=3,
        duration_sec=1.0,
        sequence_start=0,
        sequence_end=3,
    )


def test_normalize_action_representation_accepts_supported_modes():
    assert normalize_action_representation("FLOAT_4DP") == "float_4dp"
    assert normalize_action_representation("float_2dp") == "float_2dp"
    assert normalize_action_representation("bucketed_int") == "bucketed_int"


def test_serialize_actions_compresses_bucketed_int():
    actions = _make_example().actions
    float_4dp = serialize_actions_with_representation(
        actions, action_representation="float_4dp", round_digits=4
    )
    float_2dp = serialize_actions_with_representation(
        actions, action_representation="float_2dp", round_digits=4
    )
    bucketed = serialize_actions_with_representation(
        actions,
        action_representation="bucketed_int",
        round_digits=4,
        action_bucket_count=8,
    )

    assert ".0000" in float_4dp
    assert ".00" in float_2dp
    assert "." not in bucketed
    assert len(bucketed) < len(float_2dp) < len(float_4dp)


def test_token_length_stats_reports_all_representations():
    tokenizer = _CharTokenizer()
    example = _make_example()
    stats_by_representation, token_lengths, pass_rate = token_length_stats(
        tokenizer,
        [example],
        action_representation="float_4dp",
        round_digits=4,
        action_bucket_count=8,
    )

    assert set(stats_by_representation) == set(ACTION_REPRESENTATIONS)
    assert token_lengths == [stats_by_representation["float_4dp"]["mean"]]
    assert stats_by_representation["float_2dp"]["mean"] < stats_by_representation[
        "float_4dp"
    ]["mean"]
    assert stats_by_representation["bucketed_int"]["mean"] < stats_by_representation[
        "float_2dp"
    ]["mean"]
    assert pass_rate == 1.0


def test_filter_by_token_length_uses_selected_representation():
    tokenizer = _CharTokenizer()
    example = _make_example()
    _, float_4dp_target, _ = build_case(
        tokenizer=tokenizer,
        example=example,
        action_representation="float_4dp",
        round_digits=4,
        action_bucket_count=8,
    )
    _, bucketed_target, _ = build_case(
        tokenizer=tokenizer,
        example=example,
        action_representation="bucketed_int",
        round_digits=4,
        action_bucket_count=8,
    )

    kept_float, stats_float = filter_by_token_length(
        tokenizer,
        [example],
        action_representation="float_4dp",
        round_digits=4,
        action_bucket_count=8,
        max_target_tokens=len(bucketed_target),
    )
    kept_bucket, stats_bucket = filter_by_token_length(
        tokenizer,
        [example],
        action_representation="bucketed_int",
        round_digits=4,
        action_bucket_count=8,
        max_target_tokens=len(bucketed_target),
    )

    assert len(float_4dp_target) > len(bucketed_target)
    assert stats_float == {"kept": 0, "dropped": 1}
    assert stats_bucket == {"kept": 1, "dropped": 0}
    assert len(kept_float) == 0
    assert len(kept_bucket) == 1
