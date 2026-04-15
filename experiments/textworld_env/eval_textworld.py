"""Run textworld-env rollouts against a live vLLM inference server.

Smoke test for the textworld port: loads N eval samples, rolls out each
one against a running vLLM OpenAI-compatible endpoint, and reports reward,
completion length, and compaction event counts.

Designed to be run from inside the kv-eviction venv on a node (or
container) that has network access to the vLLM server:

    python eval_textworld.py \
        --dataset /pscratch/.../textworld_cooking_mix \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --num-examples 100 \
        --max-episode-steps 50 \
        --max-concurrent 32 \
        --output-json /path/to/results.json

When --padding-block-size > 0, this script:
  1. Loads the model's tokenizer
  2. Resolves `<|im_end|>` and filler token ids via
     kv_eviction.padding.resolve_im_end_token_id / resolve_filler_token_id
  3. Calls kv_eviction.env.configure_message_padding(...) which installs
     the openai.AsyncCompletions.create interceptor
Every subsequent chat request is pre-padded so its `<|im_end|>`s land on
block-size boundaries — required for the server-side
`compaction_assume_aligned_turn_boundaries=true` path.

Pass --padding-block-size 0 to disable padding (full-context baseline).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to textworld_cooking_mix directory")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--num-examples", type=int, default=100)
    p.add_argument("--max-episode-steps", type=int, default=50)
    p.add_argument("--max-concurrent", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=512)
    # Default 1.0 + no top_p/top_k = ancestral sampling (matches mkv-rl prod).
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default="textworld_eval_results.json")
    p.add_argument(
        "--api-key-var",
        default="DUMMY_API_KEY",
        help="Name of env var holding the API key (vLLM ignores but openai client requires one).",
    )
    p.add_argument(
        "--padding-block-size",
        type=int,
        default=16,
        help="Block size for block-aligned message padding. 0 = disable padding "
             "(use for full-context baseline).",
    )
    return p.parse_args()


async def main():
    args = _args()

    # Must import kv_eviction BEFORE verifiers so the module-level
    # monkey-patches (AsyncCompletions interceptor, from_native_response
    # forwarding) are installed before verifiers' client is built.
    import kv_eviction  # noqa: F401 — side-effect import
    from kv_eviction.env import configure_message_padding
    from kv_eviction.padding import (
        resolve_filler_token_id,
        resolve_im_end_token_id,
    )

    import verifiers as vf

    # Dummy API key — vLLM accepts any bearer token but the openai-python
    # client insists on one being set.
    os.environ.setdefault(args.api_key_var, "dummy")

    # Install block-aligned message padding so the client's outgoing chat
    # completion requests have filler tokens after each <|im_end|> that
    # land the next turn on a block boundary. Server-side
    # compaction_assume_aligned_turn_boundaries=true relies on this.
    # When block_size=0, padding is disabled (full-context baseline).
    if args.padding_block_size > 0:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        im_end_id = resolve_im_end_token_id(tokenizer)
        filler_id = resolve_filler_token_id(tokenizer, override=None)
        configure_message_padding(
            enabled=True,
            tokenizer=tokenizer,
            block_size=args.padding_block_size,
            filler_token_id=filler_id,
            im_end_token_id=im_end_id,
        )
        print(
            f"[eval] block-aligned padding ON: block_size={args.padding_block_size} "
            f"im_end={im_end_id} filler={filler_id}",
            flush=True,
        )
    else:
        print("[eval] block-aligned padding OFF (full-context baseline)", flush=True)

    # Load the ENTIRE dataset (not just the first num_examples) so we can
    # draw an iid sample across all difficulty tiers. metadata.json orders
    # rows by difficulty: the first 1250 are all easy-nav, the next 500 are
    # "current", then hard, hard-12room, hard-drop. Taking rows[:100] hits
    # only easy-nav and gives an unrepresentative reward distribution.
    env = vf.load_environment(
        "textworld-env",
        dataset_path=args.dataset,
        max_episode_steps=args.max_episode_steps,
        num_train_examples=None,     # None = use all rows in metadata
        num_eval_examples=0,
        seed=args.seed,
    )
    total_rows = len(env.dataset)
    env.dataset = env.dataset.shuffle(seed=args.seed)
    print(
        f"[eval] shuffled {total_rows} rows with seed={args.seed}; "
        f"first {args.num_examples} will be used",
        flush=True,
    )
    # verifiers' env.evaluate(num_examples=N) falls back to get_dataset()
    # when eval_dataset is None and slices the first N rows, which are now
    # our shuffled iid sample.

    client_config = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=args.base_url,
        api_key_var=args.api_key_var,
        timeout=3600.0,
        connect_timeout=30.0,
        max_retries=3,
    )

    sampling_args = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    print(
        f"[eval] dataset={args.dataset} model={args.model} base_url={args.base_url}",
        flush=True,
    )
    print(
        f"[eval] num_examples={args.num_examples} max_episode_steps={args.max_episode_steps} "
        f"max_concurrent={args.max_concurrent}",
        flush=True,
    )

    t0 = time.time()
    results = await env.evaluate(
        client=client_config,
        model=args.model,
        sampling_args=sampling_args,
        num_examples=args.num_examples,
        rollouts_per_example=1,
        max_concurrent=args.max_concurrent,
        state_columns=["tw_score", "tw_max_score", "tw_done"],
        save_results=False,
    )
    elapsed = time.time() - t0

    # GenerateOutputs is a TypedDict at runtime — access via dict keys.
    outputs = results.get("outputs", []) if isinstance(results, dict) else []

    rewards: list[float] = []
    completion_lens: list[int] = []
    compaction_event_counts: list[int] = []
    trajectory_lens: list[int] = []
    finished: list[bool] = []
    truncated: list[bool] = []
    tw_scores: list[int] = []
    tw_max_scores: list[int] = []

    for out in outputs:
        r = out.get("reward")
        if r is not None:
            rewards.append(float(r))
        finished.append(bool(out.get("is_completed", False)))
        truncated.append(bool(out.get("is_truncated", False)))
        # state_columns promote per-rollout state entries to top-level output keys.
        if "tw_score" in out:
            tw_scores.append(int(out["tw_score"] or 0))
        if "tw_max_score" in out:
            tw_max_scores.append(int(out["tw_max_score"] or 1))

        traj = out.get("trajectory") or []
        trajectory_lens.append(len(traj))

        total_events = 0
        last_step_completion = 0
        for step in traj:
            if not isinstance(step, dict):
                continue
            extras = step.get("extras") or {}
            events = extras.get("compaction_events") or []
            total_events += len(events)
            cids = step.get("completion_ids") or []
            if cids:
                last_step_completion = len(cids)
        compaction_event_counts.append(total_events)
        completion_lens.append(last_step_completion)

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": statistics.fmean(xs),
            "min": min(xs),
            "max": max(xs),
            "median": statistics.median(xs),
        }

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "base_url": args.base_url,
        "num_examples": args.num_examples,
        "max_episode_steps": args.max_episode_steps,
        "max_concurrent": args.max_concurrent,
        "elapsed_sec": round(elapsed, 2),
        "reward": _stats(rewards),
        "finished_rate": (sum(finished) / len(finished)) if finished else 0.0,
        "truncated_rate": (sum(truncated) / len(truncated)) if truncated else 0.0,
        "tw_score_mean": (statistics.fmean(tw_scores) if tw_scores else 0.0),
        "tw_max_score_mean": (statistics.fmean(tw_max_scores) if tw_max_scores else 0.0),
        "compaction_events_per_rollout": _stats([float(x) for x in compaction_event_counts]),
        "trajectory_len": _stats([float(x) for x in trajectory_lens]),
        "completion_len": _stats([float(x) for x in completion_lens]),
        "padding_block_size": args.padding_block_size,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("[eval] summary:", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print(f"[eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
