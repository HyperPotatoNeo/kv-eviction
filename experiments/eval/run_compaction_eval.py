#!/usr/bin/env python3
"""Evaluate Full Context vs Compaction on rg-mix-env (100 problems).

Each condition runs as a SEPARATE subprocess to avoid vLLM distributed
teardown hangs. Results are saved to JSON and compared at the end.

Usage (inside container on compute node):
    source /pscratch/sd/s/siddart2/kv-eviction/.venv/bin/activate
    python run_compaction_eval.py
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── Config ──
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
NUM_EVAL = 100
MAX_TOKENS = 16384
TEMPERATURE = 0.6
SEED = 43
TP = 4                   # tensor parallel across 4 GPUs
MAX_MODEL_LEN = 16384
COMPACTION_WINDOW = 4096
COMPACTION_STRIDE = 512
OUTPUT_DIR = Path("/pscratch/sd/s/siddart2/kv-eviction/experiments/eval/results")
SCRIPT_DIR = Path(__file__).parent

sys.stdout.reconfigure(line_buffering=True)


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_file = OUTPUT_DIR / "eval.log"
    with open(log_file, "a") as f:
        f.write(line + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "eval.log").write_text("")

    log("=" * 70)
    log("KV Cache Compaction Evaluation")
    log(f"Model: {MODEL}, {NUM_EVAL} problems, max_tokens={MAX_TOKENS}")
    log(f"TP={TP}, max_model_len={MAX_MODEL_LEN}")
    log(f"Compaction: window={COMPACTION_WINDOW}, stride={COMPACTION_STRIDE}")
    log("=" * 70)

    worker_script = SCRIPT_DIR / "_run_single_condition.py"

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    # ── Condition 1: Full Context ──
    log("\nLaunching Full Context condition...")
    t0 = time.time()
    ret = subprocess.run(
        [sys.executable, str(worker_script), "full_context"],
        env=env,
        timeout=7200,  # 1h safety timeout
    )
    if ret.returncode != 0:
        log(f"ERROR: Full Context exited with code {ret.returncode}")
        sys.exit(1)
    log(f"Full Context done in {time.time()-t0:.0f}s")

    # ── Condition 2: Compaction ──
    log("\nLaunching Compaction condition...")
    t0 = time.time()
    ret = subprocess.run(
        [sys.executable, str(worker_script), "compaction"],
        env=env,
        timeout=7200,
    )
    if ret.returncode != 0:
        log(f"ERROR: Compaction exited with code {ret.returncode}")
        sys.exit(1)
    log(f"Compaction done in {time.time()-t0:.0f}s")

    # ── Load and compare results ──
    fc_path = OUTPUT_DIR / "full_context.json"
    cw_path = OUTPUT_DIR / "compaction.json"

    if not fc_path.exists() or not cw_path.exists():
        log("ERROR: Result files missing!")
        sys.exit(1)

    fc = json.loads(fc_path.read_text())
    cw = json.loads(cw_path.read_text())

    log(f"\n{'='*70}")
    log("COMPARISON: Full Context vs Compaction")
    log(f"{'='*70}")
    log(f"  {'Metric':<35} {'Full Context':>15} {'Compaction':>15}")
    log(f"  {'-'*35} {'-'*15} {'-'*15}")

    def fmt(key, label, fmt_str=".4f"):
        v1 = fc.get(key, 0)
        v2 = cw.get(key, 0)
        log(f"  {label:<35} {format(v1, fmt_str):>15} {format(v2, fmt_str):>15}")

    fmt("overall_pass_at_1", "pass@1")
    fmt("overall_pass_at_4", "pass@4")
    fmt("inference_seconds", "Inference time (s)", ".1f")
    fmt("throughput_tok_per_sec", "Throughput (tok/s)", ".0f")
    fmt("avg_output_tokens_per_sample", "Avg output tokens/sample", ".0f")
    fmt("avg_seq_len", "Avg seq length", ".0f")
    fmt("total_output_tokens", "Total output tokens", ",")

    t_fc = fc.get("inference_seconds", 1)
    t_cw = cw.get("inference_seconds", 1)
    if t_fc > 0 and t_cw > 0:
        log(f"\n  Speedup: {t_fc/t_cw:.2f}x")

    # Per-task comparison
    log(f"\n  Per-task:")
    log(f"  {'Task':<22} {'FC p@1':>8} {'CW p@1':>8} {'d p@1':>8}  "
        f"{'FC p@4':>8} {'CW p@4':>8} {'d p@4':>8}")
    fc_tasks = {t["task"]: t for t in fc.get("per_task", [])}
    cw_tasks = {t["task"]: t for t in cw.get("per_task", [])}
    for task in sorted(set(fc_tasks) | set(cw_tasks)):
        fct = fc_tasks.get(task, {})
        cwt = cw_tasks.get(task, {})
        a1 = fct.get("pass_at_1", 0)
        b1 = cwt.get("pass_at_1", 0)
        a4 = fct.get("pass_at_4", 0)
        b4 = cwt.get("pass_at_4", 0)
        log(f"  {task:<22} {a1:>8.4f} {b1:>8.4f} {b1-a1:>+8.4f}  "
            f"{a4:>8.4f} {b4:>8.4f} {b4-a4:>+8.4f}")

    log(f"\n{'='*70}")

    # Save combined results
    combined = {"config": fc.get("config", {}), "full_context": fc, "compaction": cw}
    out_file = OUTPUT_DIR / "compaction_eval.json"
    with open(out_file, "w") as f:
        json.dump(combined, f, indent=2)
    log(f"Combined results saved to {out_file}")


if __name__ == "__main__":
    main()
