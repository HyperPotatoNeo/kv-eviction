"""Run textworld-env rollouts against a live vLLM inference server.

Smoke test for the textworld port: loads N eval samples, rolls out each
one against a running vLLM OpenAI-compatible endpoint, and reports reward,
completion length, compaction event counts, soft success rate (mean
normalized TextWorld score), and hard success rate (1 iff final score reaches
max score / the game is won).

Designed to be run from inside the kv-eviction venv on a node (or
container) that has network access to the vLLM server:

    python eval_textworld.py \
        --dataset /pscratch/.../textworld_cooking_mix \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --num-examples 100 \
        --eval-source eval \
        --eval-set-json experiments/textworld_env/eval_sets/textworld_eval_100_seed42.json \
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

from datasets import Dataset, concatenate_datasets


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to textworld_cooking_mix directory")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--num-examples", type=int, default=100)
    p.add_argument(
        "--eval-source",
        choices=("eval", "train-shuffle", "all-shuffle"),
        default="eval",
        help=(
            "Where to draw examples from. 'eval' uses the held-out "
            "eval_dataset split; '*-shuffle' modes deterministically shuffle "
            "train or train+eval rows with --seed."
        ),
    )
    p.add_argument(
        "--eval-set-json",
        default=None,
        help=(
            "Optional manifest of fixed TextWorld answer/game ids. If the file "
            "exists, those ids are loaded and --eval-source/--seed selection is "
            "ignored. If it does not exist, the selected eval set is written."
        ),
    )
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


def _game_seed(game_file: str) -> int | None:
    try:
        return int(Path(game_file).stem.split("_")[-1])
    except (ValueError, IndexError):
        return None


def _manifest_examples(env, rows: list[dict]) -> list[dict]:
    examples = []
    for row in rows:
        answer = int(row["answer"])
        game_file = env._game_files[answer]
        examples.append(
            {
                "answer": answer,
                "task": row.get("task"),
                "game_file": game_file,
                "game_seed": _game_seed(game_file),
            }
        )
    return examples


def _dataset_from_manifest(env, manifest_path: Path) -> Dataset:
    payload = json.loads(manifest_path.read_text())
    answers = payload.get("answers")
    if answers is None:
        answers = [ex["answer"] for ex in payload.get("examples", [])]
    if not answers:
        raise ValueError(f"Eval manifest {manifest_path} has no answers/examples.")

    rows_by_answer: dict[int, dict] = {}
    for ds in [env.dataset, getattr(env, "eval_dataset", None)]:
        if ds is None:
            continue
        for row in ds.to_list():
            rows_by_answer[int(row["answer"])] = row

    missing = [int(a) for a in answers if int(a) not in rows_by_answer]
    if missing:
        raise ValueError(
            f"Eval manifest {manifest_path} references {len(missing)} answer ids "
            f"that are not present in dataset/eval_dataset: {missing[:10]}"
        )

    return Dataset.from_list([rows_by_answer[int(a)] for a in answers])


def _select_eval_dataset(env, args) -> Dataset:
    manifest_path = Path(args.eval_set_json) if args.eval_set_json else None
    if manifest_path is not None and manifest_path.exists():
        selected = _dataset_from_manifest(env, manifest_path)
        print(
            f"[eval] loaded fixed eval set from {manifest_path} "
            f"({len(selected)} examples)",
            flush=True,
        )
        return selected

    if args.eval_source == "eval":
        if env.eval_dataset is None:
            raise ValueError(
                "Requested --eval-source eval, but this dataset has no "
                "eval_dataset split. Regenerate it with "
                "experiments/textworld_env/prepare_dataset.sh."
            )
        selected = env.eval_dataset
    elif args.eval_source == "train-shuffle":
        selected = env.dataset.shuffle(seed=args.seed)
    else:
        datasets = [env.dataset]
        if env.eval_dataset is not None:
            datasets.append(env.eval_dataset)
        selected = concatenate_datasets(datasets).shuffle(seed=args.seed)

    if args.num_examples > 0:
        if len(selected) < args.num_examples:
            raise ValueError(
                f"Requested {args.num_examples} eval examples from "
                f"{args.eval_source}, but only {len(selected)} are available. "
                "For the 100-question TextWorld eval set, regenerate the "
                "dataset with --eval-per-difficulty 20."
            )
        selected = selected.select(range(args.num_examples))

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        rows = selected.to_list()
        payload = {
            "version": 1,
            "dataset": str(args.dataset),
            "source": args.eval_source,
            "seed": args.seed,
            "num_examples": len(rows),
            "answers": [int(row["answer"]) for row in rows],
            "examples": _manifest_examples(env, rows),
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[eval] wrote fixed eval set manifest to {manifest_path}", flush=True)

    return selected


def _hard_success(reward: float | None, score: int | None, max_score: int | None, won: bool | None) -> bool:
    if won:
        return True
    if score is not None and max_score is not None:
        return int(score) >= int(max_score)
    return reward is not None and float(reward) >= 1.0


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

    # Load train + held-out eval splits, then replace env.eval_dataset with
    # the fixed subset used by this run. This keeps TextWorldEnv's game-file
    # and max-score metadata while making the evaluated question set explicit.
    env = vf.load_environment(
        "textworld-env",
        dataset_path=args.dataset,
        max_episode_steps=args.max_episode_steps,
        num_train_examples=None,
        num_eval_examples=None,
        seed=args.seed,
    )
    selected_eval = _select_eval_dataset(env, args)
    env.eval_dataset = selected_eval
    total_rows = len(env.dataset)
    total_eval_rows = len(env.eval_dataset)
    print(
        f"[eval] train_rows={total_rows} selected_eval_rows={total_eval_rows} "
        f"source={args.eval_source} seed={args.seed}",
        flush=True,
    )

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
        num_examples=len(selected_eval),
        rollouts_per_example=1,
        max_concurrent=args.max_concurrent,
        state_columns=["tw_score", "tw_max_score", "tw_done", "tw_won"],
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
    hard_successes: list[bool] = []
    rollout_rows: list[dict] = []

    for out in outputs:
        r = out.get("reward")
        reward = float(r) if r is not None else None
        if r is not None:
            rewards.append(reward)
        finished.append(bool(out.get("is_completed", False)))
        truncated.append(bool(out.get("is_truncated", False)))
        # state_columns promote per-rollout state entries to top-level output keys.
        score = None
        max_score = None
        if "tw_score" in out:
            score = int(out["tw_score"] or 0)
            tw_scores.append(score)
        if "tw_max_score" in out:
            max_score = int(out["tw_max_score"] or 1)
            tw_max_scores.append(max_score)
        won = bool(out.get("tw_won", False))
        hard = _hard_success(reward, score, max_score, won)
        hard_successes.append(hard)

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
        rollout_rows.append(
            {
                "example_id": out.get("example_id"),
                "task": out.get("task"),
                "answer": out.get("answer"),
                "reward": reward,
                "tw_score": score,
                "tw_max_score": max_score,
                "tw_won": won,
                "hard_success": hard,
                "is_completed": bool(out.get("is_completed", False)),
                "is_truncated": bool(out.get("is_truncated", False)),
                "compaction_events": total_events,
                "trajectory_len": len(traj),
                "completion_len": last_step_completion,
            }
        )

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
        "num_examples": len(selected_eval),
        "eval_source": args.eval_source,
        "eval_set_json": args.eval_set_json,
        "max_episode_steps": args.max_episode_steps,
        "max_concurrent": args.max_concurrent,
        "elapsed_sec": round(elapsed, 2),
        "reward": _stats(rewards),
        "success_rate": statistics.fmean(rewards) if rewards else 0.0,
        "hard_success_rate": (
            sum(hard_successes) / len(hard_successes) if hard_successes else 0.0
        ),
        "hard_success_count": int(sum(hard_successes)),
        "finished_rate": (sum(finished) / len(finished)) if finished else 0.0,
        "truncated_rate": (sum(truncated) / len(truncated)) if truncated else 0.0,
        "tw_score_mean": (statistics.fmean(tw_scores) if tw_scores else 0.0),
        "tw_max_score_mean": (statistics.fmean(tw_max_scores) if tw_max_scores else 0.0),
        "compaction_events_per_rollout": _stats([float(x) for x in compaction_event_counts]),
        "trajectory_len": _stats([float(x) for x in trajectory_lens]),
        "completion_len": _stats([float(x) for x in completion_lens]),
        "padding_block_size": args.padding_block_size,
        "rollouts": rollout_rows,
    }

    per_task: dict[str, dict] = {}
    for task in sorted({r["task"] for r in rollout_rows}):
        task_rows = [r for r in rollout_rows if r["task"] == task]
        task_rewards = [r["reward"] for r in task_rows if r["reward"] is not None]
        per_task[str(task)] = {
            "n": len(task_rows),
            "success_rate": statistics.fmean(task_rewards) if task_rewards else 0.0,
            "hard_success_rate": (
                sum(bool(r["hard_success"]) for r in task_rows) / len(task_rows)
                if task_rows
                else 0.0
            ),
            "hard_success_count": sum(bool(r["hard_success"]) for r in task_rows),
        }
    summary["per_task"] = per_task

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("[eval] summary:", flush=True)
    for k, v in summary.items():
        if k == "rollouts":
            print(f"  {k}: {len(v)} rows", flush=True)
            continue
        print(f"  {k}: {v}", flush=True)
    print(f"[eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
