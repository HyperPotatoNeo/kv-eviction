#!/usr/bin/env python3
"""Phase 3.4 live KL test — inference side.

Runs two vLLM phases in sequence on a single GPU:
  1. COMPACTION: Qwen3-4B-Instruct-2507 with --compaction-window-size 4096
     --compaction-stride 512. Collects per-token logprobs AND
     compaction_events per sample.
  2. BASELINE: same model, no compaction. Collects per-token logprobs only.

Both phases use the same 10 rg-mix problems and same seed so token sequences
are comparable (they will differ because attention dynamics differ, but the
KL metric per phase is internally consistent).

Outputs: results/rollouts_compaction.json and results/rollouts_baseline.json

These JSONs are loaded by run_kl_test.py on the trainer node.

Usage (inside podman-hpc container with kv-eviction .venv activated):
    python run_inference.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ─── Config ───
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
NUM_PROBLEMS = 10
MAX_TOKENS = 16384
MAX_MODEL_LEN = 16384
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = -1
SEED = 43
COMPACTION_WINDOW = 4096
COMPACTION_STRIDE = 512
BLOCK_SIZE = 16

OUTPUT_DIR = Path("/pscratch/sd/s/siddart2/kv-eviction/experiments/phase3_kl_test/results")
RG_MIX_DIR = "/pscratch/sd/s/siddart2/mkv-rl/experiments/rg_mix"

sys.stdout.reconfigure(line_buffering=True)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_prompts() -> tuple[list[str], list[list[int]], list[int]]:
    """Build 10 rg-mix chat prompts, return (rendered, prompt_ids_per_sample,
    prompt_token_counts)."""
    if RG_MIX_DIR not in sys.path:
        sys.path.insert(0, RG_MIX_DIR)
    import rg_mix_env  # type: ignore
    from transformers import AutoTokenizer

    env = rg_mix_env.RGMixEnv(
        num_train_examples=100, num_eval_examples=NUM_PROBLEMS, seed=SEED,
    )
    eval_ds = env.get_eval_dataset()
    shard = [eval_ds[i] for i in range(NUM_PROBLEMS)]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rendered_prompts: list[str] = []
    prompt_ids_list: list[list[int]] = []
    prompt_token_counts: list[int] = []
    for row in shard:
        rendered = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        rendered_prompts.append(rendered)
        prompt_ids_list.append(ids)
        prompt_token_counts.append(len(ids))

    log(
        f"Loaded {len(rendered_prompts)} rg-mix prompts, prompt token counts: "
        f"min={min(prompt_token_counts)}, max={max(prompt_token_counts)}, "
        f"mean={sum(prompt_token_counts)/len(prompt_token_counts):.0f}"
    )
    return rendered_prompts, prompt_ids_list, prompt_token_counts


def run_one_phase(
    condition: str,
    rendered_prompts: list[str],
    prompt_ids_list: list[list[int]],
    prompt_token_counts: list[int],
) -> dict:
    """Launch vLLM, generate, extract logprobs + compaction events, unload.

    Returns a dict suitable for JSON dump. Must be called ONCE per process
    lifetime per condition; the import order of vLLM ensures a clean state.
    """
    assert condition in ("compaction", "baseline")
    is_compaction = condition == "compaction"
    from vllm import LLM, SamplingParams

    log(f"=== Phase: {condition} ===")
    load_start = time.time()
    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enforce_eager=True,  # avoid torch.compile; irrelevant for logprob accuracy
        async_scheduling=False,
        seed=SEED,
    )
    if is_compaction:
        kwargs["compaction_window_size"] = COMPACTION_WINDOW
        kwargs["compaction_stride"] = COMPACTION_STRIDE
    llm = LLM(**kwargs)
    log(f"  vLLM loaded in {time.time()-load_start:.0f}s")

    # logprobs=0 → return only the sampled-token logprob per position.
    # Some vLLM versions require logprobs>=1; use 1 and pick the sampled
    # token's logprob out of the returned dict per position.
    sampling_params = SamplingParams(
        n=1,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        seed=SEED,
        logprobs=1,
        # ignore_eos so the rollout reliably exceeds the compaction window
        # and triggers 20+ evictions per sample. Without this, short
        # answers wouldn't trigger any compaction and we'd have nothing to
        # test.
        ignore_eos=True,
    )

    log(f"  generating {len(rendered_prompts)} samples (max_tokens={MAX_TOKENS}, "
        f"ignore_eos=True)")
    t0 = time.time()
    outputs = llm.generate(rendered_prompts, sampling_params)
    elapsed = time.time() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    log(
        f"  generated {total_tokens} tokens in {elapsed:.1f}s "
        f"({total_tokens/elapsed:.0f} tok/s)"
    )

    # Extract per-sample rollout data.
    samples: list[dict] = []
    total_compaction_events = 0
    for i, output in enumerate(outputs):
        assert len(output.outputs) == 1, "n=1 expected"
        choice = output.outputs[0]
        comp_token_ids = list(choice.token_ids)
        # Extract the sampled-token logprob per position.
        # choice.logprobs is a list of dicts {token_id: Logprob(logprob=float, ...)}
        per_token_logprobs: list[float] = []
        for pos, logprob_dict in enumerate(choice.logprobs or []):
            tok_id = comp_token_ids[pos]
            entry = logprob_dict.get(tok_id) if logprob_dict else None
            if entry is None:
                # Fallback: the sampled token wasn't in the top-K. Take the
                # logprob from the RequestOutput's overall sampled-token path.
                # This branch shouldn't hit with logprobs=1 since the sampled
                # token is always included, but guard for safety.
                per_token_logprobs.append(float("nan"))
            else:
                per_token_logprobs.append(float(entry.logprob))
        assert len(per_token_logprobs) == len(comp_token_ids), (
            f"logprobs length {len(per_token_logprobs)} != tokens "
            f"{len(comp_token_ids)} for sample {i}"
        )

        # Compaction events (only set when compaction is enabled).
        events_raw = getattr(output, "compaction_events", None) or []
        events = [
            {
                "num_output_tokens_at_compaction": int(e.num_output_tokens_at_compaction),
                "tokens_evicted": int(e.tokens_evicted),
                "position_offset_after": int(e.position_offset_after),
            }
            for e in events_raw
        ]
        total_compaction_events += len(events)

        samples.append({
            "idx": i,
            "prompt_ids": prompt_ids_list[i],
            "prompt_len": prompt_token_counts[i],
            "completion_ids": comp_token_ids,
            "completion_len": len(comp_token_ids),
            "inference_logprobs": per_token_logprobs,
            "compaction_events": events,
        })

    log(
        f"  extracted {len(samples)} samples, total compaction events: "
        f"{total_compaction_events}"
    )
    if is_compaction:
        avg_events = total_compaction_events / len(samples)
        log(f"  average compaction events per sample: {avg_events:.1f}")
        assert total_compaction_events > 0, (
            "Compaction phase produced zero events — rollouts may have been "
            "shorter than the window. Increase max_tokens or check the "
            "compaction config."
        )

    # Release GPU memory before returning so the next phase can load cleanly.
    del llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "condition": condition,
        "config": {
            "model": MODEL,
            "num_problems": NUM_PROBLEMS,
            "max_tokens": MAX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "seed": SEED,
            "compaction_window_size": COMPACTION_WINDOW if is_compaction else 0,
            "compaction_stride": COMPACTION_STRIDE if is_compaction else 0,
            "block_size": BLOCK_SIZE if is_compaction else 0,
            "ignore_eos": True,
        },
        "samples": samples,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("Phase 3.4 KL test — inference side")
    log("=" * 60)

    rendered_prompts, prompt_ids_list, prompt_token_counts = load_prompts()

    # Phase 1: compaction (loads vLLM, generates, saves, unloads)
    compaction_result = run_one_phase(
        "compaction", rendered_prompts, prompt_ids_list, prompt_token_counts
    )
    compaction_path = OUTPUT_DIR / "rollouts_compaction.json"
    compaction_path.write_text(json.dumps(compaction_result))
    log(f"Wrote {compaction_path} ({compaction_path.stat().st_size/1024:.0f} KB)")

    # Phase 2: baseline (loads vLLM, generates, saves, unloads)
    baseline_result = run_one_phase(
        "baseline", rendered_prompts, prompt_ids_list, prompt_token_counts
    )
    baseline_path = OUTPUT_DIR / "rollouts_baseline.json"
    baseline_path.write_text(json.dumps(baseline_result))
    log(f"Wrote {baseline_path} ({baseline_path.stat().st_size/1024:.0f} KB)")

    log("=" * 60)
    log("Inference phase complete")
    log("=" * 60)


if __name__ == "__main__":
    main()
