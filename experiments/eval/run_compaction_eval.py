#!/usr/bin/env python3
"""Evaluate Full Context vs Compaction on rg-mix-env (100 problems).

Runs two conditions back-to-back on the SAME problems:
  1. Full Context: standard vLLM, max_model_len=16384
  2. Compaction:   compaction_window_size=4096, compaction_stride=512, max_model_len=16384

Measures: throughput, accuracy (overall + per-task), avg seq length,
number of compactions per request, and success rate vs compaction count.

Usage (on compute node, inside container):
    source /pscratch/sd/s/siddart2/kv-eviction/.venv/bin/activate
    python run_compaction_eval.py
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── Config ──
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
NUM_EVAL = 100
MAX_TOKENS = 16384       # max generation per request
TEMPERATURE = 0.6
SEED = 43
DP = 4                   # data parallel (one GPU each via tp=1)
MAX_MODEL_LEN = 16384    # total seq length budget
COMPACTION_WINDOW = 4096
COMPACTION_STRIDE = 512
OUTPUT_DIR = Path("/pscratch/sd/s/siddart2/kv-eviction/experiments/eval/results")

sys.stdout.reconfigure(line_buffering=True)
LOG_FILE = OUTPUT_DIR / "eval.log"


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


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


def load_env_and_prompts():
    """Load rg-mix-env and prepare prompts."""
    # Add mkv-rl experiments to path for rg_mix_env import
    rg_mix_dir = "/pscratch/sd/s/siddart2/mkv-rl/experiments/rg_mix"
    if rg_mix_dir not in sys.path:
        sys.path.insert(0, rg_mix_dir)

    from transformers import AutoTokenizer
    import rg_mix_env

    log("Loading rg-mix-env...")
    env = rg_mix_env.RGMixEnv(
        num_train_examples=100,
        num_eval_examples=NUM_EVAL,
        seed=SEED,
    )
    eval_ds = env.get_eval_dataset()
    log(f"  {len(eval_ds)} eval problems loaded")

    task_counts = defaultdict(int)
    for row in eval_ds:
        task_counts[row["task"]] += 1
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        log(f"  {task:25s}: {count:4d}")

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    prompts = []
    for row in eval_ds:
        prompt_text = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)

    prompt_token_counts = []
    for p in prompts:
        prompt_token_counts.append(len(tokenizer.encode(p)))

    return env, eval_ds, prompts, prompt_token_counts


def run_inference(prompts, condition_name, **llm_kwargs):
    """Run vLLM offline inference. Returns (outputs, elapsed, llm)."""
    from vllm import LLM, SamplingParams

    log(f"\n{'='*70}")
    log(f"INFERENCE: {condition_name}")
    log(f"{'='*70}")

    log(f"  Loading model...")
    t_load = time.time()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        data_parallel_size=DP,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        **llm_kwargs,
    )
    log(f"  Model loaded in {time.time() - t_load:.0f}s")

    sampling_params = SamplingParams(
        n=1,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    log(f"  Generating {len(prompts)} completions (max_tokens={MAX_TOKENS})...")
    t_infer = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t_infer

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    avg_tokens = total_tokens / len(outputs)
    log(f"  Done: {elapsed:.1f}s, {total_tokens:,} tokens, "
        f"{total_tokens/elapsed:.0f} tok/s, avg={avg_tokens:.0f}")

    # Clean up to free GPU memory for the next condition
    del llm
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # vLLM needs the distributed process group torn down between runs
    from vllm.distributed.parallel_state import destroy_model_parallel
    try:
        destroy_model_parallel()
    except Exception:
        pass

    return outputs, elapsed


def score_outputs(env, eval_ds, outputs, prompt_token_counts, condition_name):
    """Score outputs and compute per-task + compaction stats."""
    log(f"\nSCORING: {condition_name}")

    per_task = defaultdict(lambda: {
        "correct": 0, "total": 0, "tokens": [], "seq_lens": [],
    })
    # Track success rate by number of compactions
    compaction_buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total = 0
    total_compactions = 0
    requests_with_compaction = 0
    all_seq_lens = []
    all_output_lens = []

    for i, (row, output) in enumerate(zip(eval_ds, outputs)):
        task = row["task"]
        answer_idx = int(row["answer"])
        vid, entry_idx = env._entry_map[answer_idx]
        ds = env._variant_datasets[vid]
        entry = ds[entry_idx]

        completion_text = output.outputs[0].text
        n_out_tokens = len(output.outputs[0].token_ids)
        prompt_len = prompt_token_counts[i]
        seq_len = prompt_len + n_out_tokens

        # Count compactions from finish_reason / output metadata
        # For compaction-enabled runs, we estimate from seq length:
        # If seq_len would have been > window without compaction,
        # num_compactions = max(0, (seq_len_original - window) // stride)
        # But since compaction truncates, we track via the output token count:
        # The physical seq stayed <= window+stride, but actual generated tokens
        # could be much more. Compactions = max(0, ceil((prompt+generated - window) / stride))
        total_generated = n_out_tokens  # This is post-compaction output length
        total_original_len = prompt_len + total_generated
        n_compactions = max(
            0,
            (total_original_len - COMPACTION_WINDOW + COMPACTION_STRIDE - 1)
            // COMPACTION_STRIDE
        ) if "compaction" in condition_name.lower() else 0
        # But only if the total actually exceeded the window
        if total_original_len <= COMPACTION_WINDOW:
            n_compactions = 0

        if n_compactions > 0:
            total_compactions += n_compactions
            requests_with_compaction += 1

        # Score
        extracted = extract_answer(completion_text)
        try:
            score = ds.score_answer(answer=extracted, entry=entry)
        except Exception:
            score = 0.0
        if score < 0.5:
            try:
                score_full = ds.score_answer(answer=completion_text, entry=entry)
                score = max(score, score_full)
            except Exception:
                pass

        correct = 1 if score >= 0.5 else 0
        total_correct += correct
        total += 1
        per_task[task]["correct"] += correct
        per_task[task]["total"] += 1
        per_task[task]["tokens"].append(n_out_tokens)
        per_task[task]["seq_lens"].append(seq_len)
        all_seq_lens.append(seq_len)
        all_output_lens.append(n_out_tokens)

        compaction_buckets[n_compactions]["correct"] += correct
        compaction_buckets[n_compactions]["total"] += 1

    # Print results
    log(f"\n  Overall: {total_correct}/{total} = {total_correct/total:.4f}")

    total_out_tokens = sum(all_output_lens)
    avg_seq_len = sum(all_seq_lens) / len(all_seq_lens)
    avg_out_len = sum(all_output_lens) / len(all_output_lens)
    log(f"  Avg seq length: {avg_seq_len:.0f}")
    log(f"  Avg output tokens: {avg_out_len:.0f}")
    log(f"  Total output tokens: {total_out_tokens:,}")

    if "compaction" in condition_name.lower():
        log(f"  Requests with compaction: {requests_with_compaction}/{total}")
        log(f"  Total compaction events: {total_compactions}")
        if requests_with_compaction > 0:
            log(f"  Avg compactions per compacted request: "
                f"{total_compactions/requests_with_compaction:.1f}")

    log(f"\n  {'Task':<25} {'pass@1':>8} {'n':>5} {'avg_tok':>8} {'avg_seq':>8}")
    log(f"  {'-'*25} {'-'*8} {'-'*5} {'-'*8} {'-'*8}")
    task_results = []
    for task in sorted(per_task.keys()):
        r = per_task[task]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        avg_tok = sum(r["tokens"]) / len(r["tokens"]) if r["tokens"] else 0
        avg_sl = sum(r["seq_lens"]) / len(r["seq_lens"]) if r["seq_lens"] else 0
        log(f"  {task:<25} {acc:>8.4f} {r['total']:>5d} {avg_tok:>8.0f} {avg_sl:>8.0f}")
        task_results.append({
            "task": task, "pass_at_1": acc, "correct": r["correct"],
            "total": r["total"], "avg_output_tokens": avg_tok,
            "avg_seq_len": avg_sl,
        })

    # Compaction bucket breakdown
    if compaction_buckets:
        log(f"\n  Success rate by # compactions:")
        log(f"  {'#compact':>10} {'pass@1':>8} {'correct':>8} {'total':>8}")
        for nc in sorted(compaction_buckets.keys()):
            b = compaction_buckets[nc]
            acc = b["correct"] / b["total"] if b["total"] > 0 else 0
            log(f"  {nc:>10d} {acc:>8.4f} {b['correct']:>8d} {b['total']:>8d}")

    return {
        "condition": condition_name,
        "overall_pass_at_1": total_correct / total,
        "total_correct": total_correct,
        "total": total,
        "total_output_tokens": total_out_tokens,
        "avg_output_tokens": avg_out_len,
        "avg_seq_len": avg_seq_len,
        "requests_with_compaction": requests_with_compaction,
        "total_compactions": total_compactions,
        "per_task": task_results,
        "compaction_buckets": {
            str(k): v for k, v in sorted(compaction_buckets.items())
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")

    os.environ.setdefault("HF_HOME", "/pscratch/sd/s/siddart2/hf_cache")

    log("=" * 70)
    log("KV Cache Compaction Evaluation")
    log(f"Model: {MODEL}")
    log(f"Problems: {NUM_EVAL}, Max tokens: {MAX_TOKENS}")
    log(f"Conditions: Full Context (16k) vs Compaction (window={COMPACTION_WINDOW})")
    log(f"DP={DP}, TP=1, max_model_len={MAX_MODEL_LEN}")
    log("=" * 70)

    # Load env and prompts (shared between both conditions)
    env, eval_ds, prompts, prompt_token_counts = load_env_and_prompts()

    all_results = {}

    # ── Condition 1: Full Context ──
    outputs_fc, elapsed_fc = run_inference(
        prompts, "Full Context",
        # No compaction args
    )
    result_fc = score_outputs(
        env, eval_ds, outputs_fc, prompt_token_counts, "Full Context"
    )
    result_fc["inference_seconds"] = elapsed_fc
    total_tok_fc = result_fc["total_output_tokens"]
    result_fc["throughput_tok_per_sec"] = total_tok_fc / elapsed_fc
    all_results["full_context"] = result_fc
    log(f"\nFull Context throughput: {total_tok_fc/elapsed_fc:.0f} tok/s")

    # Free outputs before next run
    del outputs_fc

    # ── Condition 2: Compaction ──
    outputs_cw, elapsed_cw = run_inference(
        prompts, "Compaction (window=4096)",
        compaction_window_size=COMPACTION_WINDOW,
        compaction_stride=COMPACTION_STRIDE,
    )
    result_cw = score_outputs(
        env, eval_ds, outputs_cw, prompt_token_counts, "Compaction (window=4096)"
    )
    result_cw["inference_seconds"] = elapsed_cw
    total_tok_cw = result_cw["total_output_tokens"]
    result_cw["throughput_tok_per_sec"] = total_tok_cw / elapsed_cw
    all_results["compaction"] = result_cw
    log(f"\nCompaction throughput: {total_tok_cw/elapsed_cw:.0f} tok/s")

    # ── Comparison ──
    log(f"\n{'='*70}")
    log("COMPARISON")
    log(f"{'='*70}")
    log(f"{'Metric':<35} {'Full Context':>15} {'Compaction':>15}")
    log(f"{'-'*35} {'-'*15} {'-'*15}")

    metrics = [
        ("pass@1", f"{result_fc['overall_pass_at_1']:.4f}",
         f"{result_cw['overall_pass_at_1']:.4f}"),
        ("Inference time (s)", f"{elapsed_fc:.1f}", f"{elapsed_cw:.1f}"),
        ("Throughput (tok/s)", f"{total_tok_fc/elapsed_fc:.0f}",
         f"{total_tok_cw/elapsed_cw:.0f}"),
        ("Avg output tokens", f"{result_fc['avg_output_tokens']:.0f}",
         f"{result_cw['avg_output_tokens']:.0f}"),
        ("Avg seq length", f"{result_fc['avg_seq_len']:.0f}",
         f"{result_cw['avg_seq_len']:.0f}"),
        ("Total output tokens", f"{result_fc['total_output_tokens']:,}",
         f"{result_cw['total_output_tokens']:,}"),
        ("Requests with compaction", "0",
         f"{result_cw['requests_with_compaction']}"),
        ("Total compactions", "0",
         f"{result_cw['total_compactions']}"),
    ]
    for label, fc_val, cw_val in metrics:
        log(f"  {label:<35} {fc_val:>15} {cw_val:>15}")

    # Speedup
    if elapsed_fc > 0 and elapsed_cw > 0:
        speedup = elapsed_fc / elapsed_cw
        log(f"\n  Speedup: {speedup:.2f}x")

    log(f"{'='*70}")

    # Save results
    final = {
        "config": {
            "model": MODEL, "num_eval": NUM_EVAL,
            "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
            "seed": SEED, "dp": DP, "max_model_len": MAX_MODEL_LEN,
            "compaction_window": COMPACTION_WINDOW,
            "compaction_stride": COMPACTION_STRIDE,
        },
        "results": all_results,
    }
    out_file = OUTPUT_DIR / "compaction_eval.json"
    with open(out_file, "w") as f:
        json.dump(final, f, indent=2)
    log(f"\nResults saved to {out_file}")
    log(f"Log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
