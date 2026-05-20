#!/usr/bin/env python3
"""Test protected-prefix compaction with BabyAI multi-turn inference.

Runs a few BabyAI rollouts against a vLLM server with compaction enabled
and protected_prefix_tokens set. Logs compaction events per turn and saves
results to JSON for analysis.

Prerequisites:
  1. Install balrog deps: bash scripts/install_balrog.sh
  2. Start vLLM server with compaction + protected prefix:

     python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen3-4B-Instruct-2507 \
       --compaction-window-size 2048 \
       --compaction-stride 512 \
       --compaction-protected-prefix-tokens 256 \
       --enable-prefix-caching false \
       --async-scheduling false \
       --enforce-eager \
       --max-model-len 8192 \
       --port 8000

  3. Run this script:
     python experiments/test_protected_prefix.py

     Or with custom URL:
     python experiments/test_protected_prefix.py --base-url http://localhost:8000/v1

Check vLLM server stderr for [COMPACT] diagnostic lines showing:
  - effective_prompt < num_prompt (protected prefix active)
  - prompt_evicted > 0 (old conversation tokens being evicted)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure kv_eviction monkey-patches are active (captures compaction events
# from vLLM responses onto trajectory step extras).
import kv_eviction  # noqa: F401

import verifiers as vf
from verifiers.types import ClientConfig


def make_client_config(base_url: str, api_key: str) -> ClientConfig:
    return ClientConfig(
        client_type="openai_chat_completions",
        api_key_var="__inline__",
        api_base_url=base_url,
        timeout=300.0,
        max_retries=2,
    )


async def run_rollouts(
    base_url: str,
    model: str,
    num_rollouts: int,
    max_turns: int,
    output_path: Path,
):
    # Set API key env var (vLLM doesn't check it but the client needs one).
    os.environ["__inline__"] = "dummy"

    print(f"Loading BabyAI environment...", flush=True)
    env = vf.load_environment(
        "balrog-bench",
        environments=["babyai"],
        max_text_history=max_turns,
    )
    dataset = env.get_eval_dataset()
    print(f"  Dataset: {len(dataset)} rows", flush=True)

    client_config = make_client_config(base_url, "dummy")
    sampling_args = {
        "temperature": 1.0,
        "max_completion_tokens": 512,
    }

    results = []
    for i in range(min(num_rollouts, len(dataset))):
        row = dataset[i]
        task_info = row.get("info", {})
        print(
            f"\n{'='*60}\n"
            f"Rollout {i+1}/{num_rollouts}: "
            f"{task_info.get('environment', '?')}/{task_info.get('task', '?')}\n"
            f"{'='*60}",
            flush=True,
        )

        t0 = time.time()
        try:
            output = await env.run_rollout(
                input=row,
                client=client_config,
                model=model,
                sampling_args=sampling_args,
            )
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results.append({
                "rollout_idx": i,
                "error": str(e),
                "task_info": task_info,
            })
            continue
        elapsed = time.time() - t0

        trajectory = output.get("trajectory", [])
        reward = output.get("reward", None)
        stop = output.get("stop_condition", None)

        print(f"  Turns: {len(trajectory)}", flush=True)
        print(f"  Reward: {reward}", flush=True)
        print(f"  Stop: {stop}", flush=True)
        print(f"  Time: {elapsed:.1f}s", flush=True)

        turn_details = []
        for step_idx, step in enumerate(trajectory):
            tokens = step.get("tokens")
            extras = step.get("extras") or {}
            compaction_events = extras.get("compaction_events")

            prompt_len = len(tokens["prompt_ids"]) if tokens else 0
            completion_len = len(tokens["completion_ids"]) if tokens else 0

            print(
                f"  Turn {step_idx}: prompt={prompt_len} "
                f"completion={completion_len} "
                f"compaction_events={len(compaction_events) if compaction_events else 0}",
                flush=True,
            )

            if compaction_events:
                for ev_idx, ev in enumerate(compaction_events):
                    if isinstance(ev, dict):
                        print(
                            f"    event[{ev_idx}]: "
                            f"output_tokens={ev.get('num_output_tokens_at_compaction')} "
                            f"evicted={ev.get('tokens_evicted')} "
                            f"pos_offset={ev.get('position_offset_after')} "
                            f"prompt_tokens={ev.get('num_prompt_tokens', 'N/A')}",
                            flush=True,
                        )

            turn_details.append({
                "step_idx": step_idx,
                "prompt_len": prompt_len,
                "completion_len": completion_len,
                "compaction_events": compaction_events,
            })

        results.append({
            "rollout_idx": i,
            "task_info": task_info,
            "num_turns": len(trajectory),
            "reward": reward,
            "stop_condition": stop,
            "elapsed_s": elapsed,
            "turns": turn_details,
        })

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}", flush=True)

    # Summary
    total_turns = sum(r.get("num_turns", 0) for r in results if "error" not in r)
    turns_with_events = sum(
        1
        for r in results
        if "error" not in r
        for t in r.get("turns", [])
        if t.get("compaction_events")
    )
    print(f"\nSummary: {len(results)} rollouts, {total_turns} total turns, "
          f"{turns_with_events} turns with compaction events", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM server base URL",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=5,
        help="Number of BabyAI episodes to run",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=16,
        help="Max turns per episode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/test_protected_prefix/results.json"),
    )
    args = parser.parse_args()

    asyncio.run(run_rollouts(
        base_url=args.base_url,
        model=args.model,
        num_rollouts=args.num_rollouts,
        max_turns=args.max_turns,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
