# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM KV-eviction event plumbing through verifiers extras."""

from types import SimpleNamespace

from kv_eviction.env import (
    attach_compaction_events_from_response,
    attach_compaction_metrics_to_state,
)


def test_attach_response_events_merges_compaction_and_am_metadata():
    response = SimpleNamespace(
        compaction_events=[[10, 512, 512]],
        shuffle_events=[
            {
                "num_output_tokens_at_shuffle": "3",
                "chunk_index": 1,
                "chunk_start": 64,
                "chunk_end": 128,
            }
        ],
        noise_events=[
            SimpleNamespace(
                num_output_tokens_at_noise=4,
                chunk_index=2,
                chunk_start=128,
                chunk_end=192,
                target="values",
                std="0.05",
            )
        ],
    )
    step = {}

    attach_compaction_events_from_response(step, response)

    assert step["extras"]["compaction_events"] == [
        {
            "num_output_tokens_at_compaction": 10,
            "tokens_evicted": 512,
            "position_offset_after": 512,
        }
    ]
    assert step["extras"]["shuffle_events"] == [
        {
            "num_output_tokens_at_shuffle": 3,
            "chunk_index": 1,
            "chunk_start": 64,
            "chunk_end": 128,
        }
    ]
    assert step["extras"]["noise_events"] == [
        {
            "num_output_tokens_at_noise": 4,
            "chunk_index": 2,
            "chunk_start": 128,
            "chunk_end": 192,
            "target": "values",
            "std": 0.05,
        }
    ]


def test_attach_metrics_counts_all_kv_eviction_event_families():
    state = {
        "trajectory": [
            {
                "extras": {
                    "compaction_events": [[10, 512, 512]],
                    "shuffle_events": [[3, 1, 64, 128]],
                    "noise_events": [[4, 2, 128, 192, "values", "0.05"]],
                }
            }
        ]
    }

    attach_compaction_metrics_to_state(state)

    assert state["num_compaction_events"] == 1
    assert state["num_shuffle_events"] == 1
    assert state["num_noise_events"] == 1
    assert state["metrics"] == {
        "num_compaction_events": 1.0,
        "num_shuffle_events": 1.0,
        "num_noise_events": 1.0,
    }
