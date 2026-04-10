#!/usr/bin/env python3
"""DP chunk worker: one vLLM engine on one GPU, handling a shard of prompts.

Called by _run_single_condition.py as a subprocess with CUDA_VISIBLE_DEVICES
set to the target GPU. Writes results for its shard to a JSON file.

Usage: python _run_dp_chunk.py <full_context|compaction> <dp_rank> <dp_size>
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# Config (mirrors _run_single_condition.py)
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
NUM_EVAL = 100
NUM_SAMPLES = 4
MAX_TOKENS = 16384
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = -1
SEED = 43
MAX_MODEL_LEN = 16384
COMPACTION_WINDOW = 4096
COMPACTION_STRIDE = 512
OUTPUT_DIR = Path("/pscratch/sd/s/siddart2/kv-eviction/experiments/eval/results")

sys.stdout.reconfigure(line_buffering=True)


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()


def score_one(ds, entry, text: str) -> int:
    extracted = extract_answer(text)
    try:
        score = ds.score_answer(answer=extracted, entry=entry)
    except Exception:
        score = 0.0
    if score < 0.5:
        try:
            score = max(score, ds.score_answer(answer=text, entry=entry))
        except Exception:
            pass
    return 1 if score >= 0.5 else 0


def main():
    condition = sys.argv[1]
    dp_rank = int(sys.argv[2])
    dp_size = int(sys.argv[3])
    assert condition in ("full_context", "compaction")
    is_compaction = condition == "compaction"

    prefix = f"[dp{dp_rank}:{condition}]"
    print(f"{prefix} start on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", "/pscratch/sd/s/siddart2/hf_cache")

    rg_mix_dir = "/pscratch/sd/s/siddart2/mkv-rl/experiments/rg_mix"
    if rg_mix_dir not in sys.path:
        sys.path.insert(0, rg_mix_dir)
    from transformers import AutoTokenizer
    import rg_mix_env

    env = rg_mix_env.RGMixEnv(
        num_train_examples=100, num_eval_examples=NUM_EVAL, seed=SEED,
    )
    eval_ds = env.get_eval_dataset()

    # Deterministic shard: problems indexed by (idx % dp_size) == dp_rank
    shard_indices = [i for i in range(len(eval_ds)) if i % dp_size == dp_rank]
    shard = [eval_ds[i] for i in shard_indices]
    print(f"{prefix} shard has {len(shard)} problems (indices {shard_indices[:5]}...)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    prompt_token_counts = []
    for row in shard:
        p = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompts.append(p)
        prompt_token_counts.append(len(tokenizer.encode(p)))

    from vllm import LLM, SamplingParams

    print(f"{prefix} loading LLM (compaction={'on' if is_compaction else 'off'})", flush=True)
    t_load = time.time()
    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        async_scheduling=False,
        seed=SEED + dp_rank,  # distinct seeds per shard for independent sampling
    )
    if is_compaction:
        kwargs["compaction_window_size"] = COMPACTION_WINDOW
        kwargs["compaction_stride"] = COMPACTION_STRIDE
    llm = LLM(**kwargs)
    print(f"{prefix} loaded in {time.time()-t_load:.0f}s", flush=True)

    sp = SamplingParams(
        n=NUM_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
        max_tokens=MAX_TOKENS, seed=SEED + dp_rank,
    )
    print(f"{prefix} generating {len(prompts)}x{NUM_SAMPLES}", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    total_out = sum(len(s.token_ids) for o in outputs for s in o.outputs)
    print(f"{prefix} inference {elapsed:.1f}s, {total_out} tokens, "
          f"{total_out/elapsed:.0f} tok/s", flush=True)

    # Score
    results = []
    for row, output, prompt_len, orig_idx in zip(
        shard, outputs, prompt_token_counts, shard_indices
    ):
        task = row["task"]
        answer_idx = int(row["answer"])
        vid, entry_idx = env._entry_map[answer_idx]
        ds = env._variant_datasets[vid]
        entry = ds[entry_idx]

        sample_corrects = []
        sample_token_counts = []
        for samp in output.outputs:
            c = score_one(ds, entry, samp.text)
            sample_corrects.append(c)
            sample_token_counts.append(len(samp.token_ids))

        num_correct = sum(sample_corrects)
        results.append({
            "orig_idx": orig_idx,
            "task": task,
            "prompt_tokens": prompt_len,
            "num_correct": num_correct,
            "any_correct": 1 if num_correct > 0 else 0,
            "tokens_per_sample": sample_token_counts,
        })

    shard_out = {
        "dp_rank": dp_rank,
        "dp_size": dp_size,
        "condition": condition,
        "elapsed_s": elapsed,
        "total_output_tokens": total_out,
        "results": results,
    }
    out_file = OUTPUT_DIR / f"{condition}_dp{dp_rank}.json"
    out_file.write_text(json.dumps(shard_out, indent=2))
    print(f"{prefix} wrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
