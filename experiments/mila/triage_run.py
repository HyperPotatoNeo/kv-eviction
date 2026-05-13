#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys
from typing import Any


TRAINER_STEP_RE = re.compile(
    r"Step (?P<step>\d+) \| Time: (?P<time>[0-9.]+)s \| Loss: (?P<loss>[-+0-9.eE]+) "
    r"\| Entropy: (?P<entropy>[-+0-9.eE]+)(?: \| Mismatch KL: (?P<mismatch>[-+0-9.eE]+))?"
    r".*? \| Throughput: (?P<throughput>[-+0-9.eE]+) tokens/s .*? \| Peak Mem\.: "
    r"(?P<peak_mem>[-+0-9.eE]+) GiB"
)
ORCH_STEP_RE = re.compile(
    r"Step (?P<step>\d+) \| Time: (?P<time>[0-9.]+)s \| Reward: (?P<reward>[-+0-9.eE]+) "
    r"\| Seq\. Length: (?P<seq_len>[-+0-9.eE]+) tokens/sample"
)

FATAL_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"Traceback \(most recent call last\)",
        r"CUDA out of memory",
        r"torch\.OutOfMemoryError",
        r"RuntimeError:",
        r"AssertionError",
        r"\bNCCL\b",
        r"segmentation fault",
        r"core dumped",
        r"^\s*ERROR:",
        r"\bKilled\b",
    ]
]


def resolve_run_dir(target: str, run_root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(target).expanduser()
    if path.is_dir():
        return path
    for status_path in run_root.glob("*/status.json"):
        try:
            data = json.loads(status_path.read_text())
        except Exception:
            continue
        if data.get("job_id") == target:
            return status_path.parent
    raise FileNotFoundError(f"could not resolve run directory for {target}")


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_status_field(data: dict[str, Any], field: str, default: str = "") -> str:
    value = data.get(field, default)
    return str(value) if value is not None else default


def parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite value: {value}")
    return parsed


def parse_logs(log_path: pathlib.Path, trainer_steps: list[dict[str, float]], orch_steps: list[dict[str, float]]) -> list[str]:
    findings: list[str] = []
    if not log_path.exists():
        return findings

    for line in log_path.read_text(errors="replace").splitlines():
        for pattern in FATAL_PATTERNS:
            if pattern.search(line):
                findings.append(f"{log_path.name}: {line.strip()[:240]}")
                break
        match = TRAINER_STEP_RE.search(line)
        if match:
            try:
                trainer_steps.append(
                    {
                        "step": float(match.group("step")),
                        "loss": parse_float(match.group("loss")),
                        "entropy": parse_float(match.group("entropy")),
                        "mismatch_kl": parse_float(match.group("mismatch")) if match.group("mismatch") else 0.0,
                        "throughput": parse_float(match.group("throughput")),
                        "peak_mem": parse_float(match.group("peak_mem")),
                    }
                )
            except ValueError as exc:
                findings.append(f"{log_path.name}: failed to parse trainer step line: {exc}")
            continue
        match = ORCH_STEP_RE.search(line)
        if match:
            try:
                orch_steps.append(
                    {
                        "step": float(match.group("step")),
                        "reward": parse_float(match.group("reward")),
                        "seq_len": parse_float(match.group("seq_len")),
                    }
                )
            except ValueError as exc:
                findings.append(f"{log_path.name}: failed to parse orchestrator step line: {exc}")

    return findings


def latest_summary(output_dir: pathlib.Path) -> pathlib.Path | None:
    candidates = sorted(output_dir.glob("run-*/final_summary.json"))
    return candidates[-1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Triage a Mila run for hard errors and suspicious metrics.")
    parser.add_argument("target", help="Run directory or SLURM job id")
    parser.add_argument("--require-complete", action="store_true", help="Fail if status.json is not completed")
    args = parser.parse_args()

    run_root = pathlib.Path(os.environ.get("RUN_ROOT", "~/kv-runs")).expanduser()
    min_trainer_steps = int(os.environ.get("MILA_TRIAGE_MIN_TRAINER_STEPS", "3"))
    min_reward_steps = int(os.environ.get("MILA_TRIAGE_MIN_REWARD_STEPS", "3"))
    max_mismatch_kl = float(os.environ.get("MILA_TRIAGE_MAX_MISMATCH_KL", "0.01"))
    max_peak_mem = float(os.environ.get("MILA_TRIAGE_MAX_PEAK_MEM_GIB", "79.0"))
    min_throughput = float(os.environ.get("MILA_TRIAGE_MIN_THROUGHPUT", "1.0"))
    reward_const_window = int(os.environ.get("MILA_TRIAGE_REWARD_CONST_WINDOW", "5"))

    try:
        run_dir = resolve_run_dir(args.target, run_root)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    status_path = run_dir / "status.json"
    if not status_path.exists():
        print(f"ERROR: missing status file: {status_path}")
        return 1

    status = read_json(status_path)
    run_status = read_status_field(status, "status", "unknown")
    output_dir = pathlib.Path(read_status_field(status, "output_dir", str(run_dir / "outputs"))).expanduser()

    trainer_steps: list[dict[str, float]] = []
    orch_steps: list[dict[str, float]] = []
    errors: list[str] = []
    warnings: list[str] = []

    for log_name in ["trainer.log", "inference0.log", "inference1.log"]:
        errors.extend(parse_logs(run_dir / log_name, trainer_steps, orch_steps))

    if args.require_complete and run_status != "completed":
        errors.append(f"status.json reports {run_status!r}, expected 'completed'")

    if not trainer_steps:
        errors.append("trainer log does not contain any parsed trainer step lines")
    elif len(trainer_steps) < min_trainer_steps and run_status == "completed":
        warnings.append(
            f"trainer completed with only {len(trainer_steps)} parsed trainer steps (< {min_trainer_steps})"
        )

    if not orch_steps and run_status == "completed":
        warnings.append("completed run has no parsed reward/sequence-length step lines")
    elif orch_steps and len(orch_steps) < min_reward_steps and run_status == "completed":
        warnings.append(f"only {len(orch_steps)} parsed reward step lines (< {min_reward_steps})")

    if trainer_steps:
        latest = trainer_steps[-1]
        if latest["throughput"] <= min_throughput:
            warnings.append(f"latest throughput is suspiciously low: {latest['throughput']:.2f} tokens/s")
        if latest["peak_mem"] >= max_peak_mem:
            warnings.append(f"latest peak memory is too close to the device limit: {latest['peak_mem']:.2f} GiB")
        if latest["mismatch_kl"] > max_mismatch_kl:
            warnings.append(
                f"latest mismatch KL {latest['mismatch_kl']:.6f} exceeds threshold {max_mismatch_kl:.6f}"
            )
        if latest["entropy"] <= 0:
            warnings.append(f"latest entropy is non-positive: {latest['entropy']:.4f}")

    if orch_steps and len(orch_steps) >= reward_const_window:
        recent_rewards = [step["reward"] for step in orch_steps[-reward_const_window:]]
        if max(recent_rewards) == min(recent_rewards):
            warnings.append(
                "reward appears flat over the most recent "
                f"{reward_const_window} orchestrator steps ({recent_rewards[-1]:.4f})"
            )

    summary_path = latest_summary(output_dir) if output_dir.exists() else None
    if args.require_complete and summary_path is None:
        warnings.append("missing outputs/run-*/final_summary.json")

    triage_summary = {
        "run_dir": str(run_dir),
        "job_id": read_status_field(status, "job_id"),
        "status": run_status,
        "summary_path": str(summary_path) if summary_path else "",
        "trainer_steps": len(trainer_steps),
        "reward_steps": len(orch_steps),
        "latest_trainer": trainer_steps[-1] if trainer_steps else {},
        "latest_orchestrator": orch_steps[-1] if orch_steps else {},
        "errors": errors,
        "warnings": warnings,
        "passed": not errors and not warnings,
    }
    (run_dir / "triage_summary.json").write_text(json.dumps(triage_summary, indent=2, sort_keys=True) + "\n")

    print(f"Triage run: {run_dir}")
    print(f"  status:        {run_status}")
    print(f"  trainer steps: {len(trainer_steps)}")
    print(f"  reward steps:  {len(orch_steps)}")
    if trainer_steps:
        latest = trainer_steps[-1]
        print(
            "  latest train:  "
            f"loss={latest['loss']:.4f}, entropy={latest['entropy']:.4f}, "
            f"mismatch_kl={latest['mismatch_kl']:.6f}, throughput={latest['throughput']:.1f}, "
            f"peak_mem={latest['peak_mem']:.1f}"
        )
    if orch_steps:
        latest = orch_steps[-1]
        print(
            "  latest orch:   "
            f"reward={latest['reward']:.4f}, seq_len={latest['seq_len']:.1f}"
        )
    if summary_path is not None:
        print(f"  summary:       {summary_path}")

    if errors:
        print("")
        print("Errors:")
        for item in errors:
            print(f"  - {item}")
    if warnings:
        print("")
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")

    if errors:
        return 1
    if warnings:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
