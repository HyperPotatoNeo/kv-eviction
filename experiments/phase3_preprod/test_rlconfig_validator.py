#!/usr/bin/env python3
"""Sanity check for the RLConfig cross-compaction validator.

Verifies:
  1. window_size mismatch between trainer and inference is rejected
  2. stride mismatch is rejected
  3. block_size mismatch is rejected
  4. A correctly-mirrored config validates successfully
  5. Both-disabled is valid (no-op path)

Does not need a GPU — this is pure pydantic validation.
"""
from __future__ import annotations

import sys

from pydantic import ValidationError

from prime_rl.configs.rl import RLConfig


# Minimal kwargs to construct an RLConfig. Fill in only what's required
# to exercise the validator; everything else gets pydantic defaults.
def _base_kwargs(
    trainer_window: int,
    trainer_stride: int,
    trainer_block_size: int,
    inf_window: int | None,
    inf_stride: int | None,
    inf_block_size: int | None,
) -> dict:
    vllm_extra: dict = {}
    if inf_window is not None:
        vllm_extra["compaction_window_size"] = inf_window
    if inf_stride is not None:
        vllm_extra["compaction_stride"] = inf_stride
    if inf_block_size is not None:
        vllm_extra["block_size"] = inf_block_size
    return {
        "trainer": {
            "model": {
                "name": "Qwen/Qwen3-4B-Instruct-2507",
                "impl": "hf",
                "attn": "flash_attention_2",
            },
            "compaction": {
                "window_size": trainer_window,
                "stride": trainer_stride,
                "block_size": trainer_block_size,
            },
        },
        "orchestrator": {
            "model": {"name": "Qwen/Qwen3-4B-Instruct-2507"},
        },
        "inference": {
            "model": {"name": "Qwen/Qwen3-4B-Instruct-2507"},
            "vllm_extra": vllm_extra,
        },
    }


def assert_fails(kwargs: dict, needle: str, label: str) -> None:
    try:
        RLConfig.model_validate(kwargs)
    except ValidationError as e:
        msg = str(e)
        if needle not in msg:
            print(f"FAIL: {label}: expected error to contain {needle!r}")
            print(f"     got: {msg}")
            sys.exit(1)
        print(f"PASS: {label}")
        return
    print(f"FAIL: {label}: expected ValidationError, got successful validation")
    sys.exit(1)


def assert_passes(kwargs: dict, label: str) -> None:
    try:
        RLConfig.model_validate(kwargs)
    except ValidationError as e:
        print(f"FAIL: {label}: unexpected ValidationError")
        print(f"     {e}")
        sys.exit(1)
    print(f"PASS: {label}")


def main() -> None:
    # 1. window_size mismatch
    assert_fails(
        _base_kwargs(4096, 512, 16, 2048, 512, 16),
        needle="window_size",
        label="window_size mismatch rejected",
    )
    # 2. stride mismatch
    assert_fails(
        _base_kwargs(4096, 512, 16, 4096, 256, 16),
        needle="stride",
        label="stride mismatch rejected",
    )
    # 3. block_size mismatch
    assert_fails(
        _base_kwargs(4096, 512, 16, 4096, 512, 32),
        needle="block_size",
        label="block_size mismatch rejected",
    )
    # 4. Correctly mirrored
    assert_passes(
        _base_kwargs(4096, 512, 16, 4096, 512, 16),
        label="mirrored config accepted",
    )
    # 5. Both disabled (no compaction anywhere)
    assert_passes(
        _base_kwargs(0, 0, 16, None, None, None),
        label="both-disabled accepted",
    )
    # 6. Inference enabled, trainer disabled — mismatch, must be rejected
    assert_fails(
        _base_kwargs(0, 0, 16, 4096, 512, 16),
        needle="window_size",
        label="inference-only enabled is rejected (bidirectional)",
    )
    print("All validator checks passed.")


if __name__ == "__main__":
    main()
