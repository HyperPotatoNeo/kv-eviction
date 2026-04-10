#!/usr/bin/env python3
"""Worker: run a single eval condition (full_context or compaction).

Spawns DP=4 subprocesses via CUDA_VISIBLE_DEVICES, one per GPU.
Each child runs _run_dp_chunk.py with TP=1 on its shard of the 100
problems. Aggregates pass@1 and pass@4 per task.
"""

import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

NUM_EVAL = 100
NUM_SAMPLES = 4
DP = 4
TP = 1
SEED = 43
MAX_TOKENS = 16384
MAX_MODEL_LEN = 16384
COMPACTION_WINDOW = 4096
COMPACTION_STRIDE = 512
TEMPERATURE = 1.0
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_DIR = Path("/pscratch/sd/s/siddart2/kv-eviction/experiments/eval/results")
SCRIPT_DIR = Path(__file__).parent

sys.stdout.reconfigure(line_buffering=True)


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(OUTPUT_DIR / "eval.log", "a") as f:
        f.write(line + "\n")


def main():
    condition = sys.argv[1]
    assert condition in ("full_context", "compaction"), f"Unknown: {condition}"
    is_compaction = condition == "compaction"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label = f"Compaction (window={COMPACTION_WINDOW})" if is_compaction else "Full Context"
    log(f"\n{'='*70}")
    log(f"CONDITION: {label}")
    log(f"{'='*70}")

    chunk_script = SCRIPT_DIR / "_run_dp_chunk.py"
    log(f"Spawning {DP} chunk processes (TP={TP}, "
        f"{NUM_EVAL} problems × {NUM_SAMPLES} samples = {NUM_EVAL*NUM_SAMPLES} generations)")

    procs = []
    logs = []
    for rank in range(DP):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        env["PYTHONUNBUFFERED"] = "1"
        log_path = OUTPUT_DIR / f"{condition}_dp{rank}.log"
        f = open(log_path, "w")
        p = subprocess.Popen(
            [sys.executable, str(chunk_script), condition, str(rank), str(DP)],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        procs.append(p)
        logs.append(f)
        log(f"  dp{rank} pid={p.pid} → {log_path.name}")

    t0 = time.time()
    last_report = 0.0
    finished = [False] * DP
    while not all(finished):
        time.sleep(10)
        # Periodic progress tail
        now = time.time()
        for rank, p in enumerate(procs):
            if finished[rank]:
                continue
            if p.poll() is not None:
                finished[rank] = True
                rc = p.returncode
                log(f"  dp{rank} finished with rc={rc} after {now-t0:.0f}s")
                if rc != 0:
                    log(f"  dp{rank} FAILED — tail of {condition}_dp{rank}.log:")
                    log_path = OUTPUT_DIR / f"{condition}_dp{rank}.log"
                    if log_path.exists():
                        tail = log_path.read_text().splitlines()[-30:]
                        for t in tail:
                            log(f"    {t}")
        if now - last_report > 60 and not all(finished):
            running = sum(1 for f in finished if not f)
            log(f"  ... {running}/{DP} chunks still running "
                f"({now-t0:.0f}s elapsed)")
            last_report = now

    for f in logs:
        f.close()

    # Check all succeeded
    failed = [i for i, p in enumerate(procs) if p.returncode != 0]
    if failed:
        log(f"ERROR: chunks {failed} failed, aborting")
        sys.exit(1)

    # Aggregate per-chunk JSON results
    per_problem = []
    total_elapsed = 0.0  # wall time per chunk (they run in parallel)
    total_tokens = 0
    max_elapsed = 0.0
    for rank in range(DP):
        chunk_path = OUTPUT_DIR / f"{condition}_dp{rank}.json"
        if not chunk_path.exists():
            log(f"ERROR: missing chunk output {chunk_path}")
            sys.exit(1)
        data = json.loads(chunk_path.read_text())
        per_problem.extend(data["results"])
        total_elapsed += data["elapsed_s"]
        max_elapsed = max(max_elapsed, data["elapsed_s"])
        total_tokens += data["total_output_tokens"]

    assert len(per_problem) == NUM_EVAL, (
        f"Expected {NUM_EVAL} problems, got {len(per_problem)}"
    )
    per_problem.sort(key=lambda r: r["orig_idx"])

    # Aggregate pass@1 and pass@4
    overall_samples_correct = sum(r["num_correct"] for r in per_problem)
    overall_samples_total = NUM_EVAL * NUM_SAMPLES
    overall_problems_any = sum(r["any_correct"] for r in per_problem)

    per_task = defaultdict(lambda: {
        "samples_correct": 0, "samples_total": 0,
        "problems_any_correct": 0, "problems_total": 0,
        "tokens": [],
    })
    for r in per_problem:
        t = per_task[r["task"]]
        t["samples_correct"] += r["num_correct"]
        t["samples_total"] += NUM_SAMPLES
        t["problems_any_correct"] += r["any_correct"]
        t["problems_total"] += 1
        t["tokens"].extend(r["tokens_per_sample"])

    pass_at_1 = overall_samples_correct / overall_samples_total
    pass_at_n = overall_problems_any / NUM_EVAL
    all_tokens = [tok for r in per_problem for tok in r["tokens_per_sample"]]
    all_seq = [r["prompt_tokens"] + tok for r in per_problem for tok in r["tokens_per_sample"]]
    avg_out = sum(all_tokens) / len(all_tokens)
    avg_seq = sum(all_seq) / len(all_seq)
    wall_elapsed = max_elapsed  # chunks run in parallel
    throughput = total_tokens / wall_elapsed if wall_elapsed > 0 else 0

    log(f"\n  Overall pass@1 = {overall_samples_correct}/{overall_samples_total} = {pass_at_1:.4f}")
    log(f"  Overall pass@{NUM_SAMPLES} = {overall_problems_any}/{NUM_EVAL} = {pass_at_n:.4f}")
    log(f"  Avg output tokens/sample: {avg_out:.0f}")
    log(f"  Avg seq length: {avg_seq:.0f}")
    log(f"  Wall time (max chunk): {wall_elapsed:.1f}s, "
        f"aggregate throughput: {throughput:.0f} tok/s")

    log(f"\n  {'Task':<22} {'pass@1':>8} {'pass@'+str(NUM_SAMPLES):>8} {'n':>5} {'avg_tok':>8}")
    log(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*5} {'-'*8}")
    task_results = []
    for task in sorted(per_task):
        r = per_task[task]
        p1 = r["samples_correct"] / r["samples_total"]
        pk = r["problems_any_correct"] / r["problems_total"]
        avg_t = sum(r["tokens"]) / len(r["tokens"])
        log(f"  {task:<22} {p1:>8.4f} {pk:>8.4f} {r['problems_total']:>5d} {avg_t:>8.0f}")
        task_results.append({
            "task": task,
            "pass_at_1": p1,
            f"pass_at_{NUM_SAMPLES}": pk,
            "samples_correct": r["samples_correct"],
            "samples_total": r["samples_total"],
            "problems_any_correct": r["problems_any_correct"],
            "problems_total": r["problems_total"],
            "avg_output_tokens": avg_t,
        })

    result = {
        "condition": label,
        "config": {
            "model": MODEL, "num_eval": NUM_EVAL, "num_samples": NUM_SAMPLES,
            "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
            "seed": SEED, "tp": TP, "dp": DP, "max_model_len": MAX_MODEL_LEN,
            "async_scheduling": False,
            "compaction_window_size": COMPACTION_WINDOW if is_compaction else 0,
            "compaction_stride": COMPACTION_STRIDE if is_compaction else 0,
        },
        "overall_pass_at_1": pass_at_1,
        f"overall_pass_at_{NUM_SAMPLES}": pass_at_n,
        "total_samples": overall_samples_total,
        "total_problems": NUM_EVAL,
        "inference_seconds": wall_elapsed,
        "throughput_tok_per_sec": throughput,
        "total_output_tokens": total_tokens,
        "avg_output_tokens_per_sample": avg_out,
        "avg_seq_len": avg_seq,
        "per_task": task_results,
        "per_problem": per_problem,
    }

    out_file = OUTPUT_DIR / f"{condition}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log(f"  Saved to {out_file}")


if __name__ == "__main__":
    main()
