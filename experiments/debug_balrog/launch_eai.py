#!/usr/bin/env python3
"""Launch RL TOML configs on EAI with H100-80GB GPUs.

This file lives under `experiments/debug_balrog/` for historical reasons, but
it does not imply the launched config is a Balrog job. The config path decides
the workload. For the current TextWorld grid, pass TOMLs from
`experiments/textworld_flex_grid/`.

Runtime contract:
    - EAI account comes from `eai account get` with `EAI_PROFILE=yul201`.
    - Repo home is mounted as `snow.home.xinghan_lu:/home/toolkit:rw`.
    - Scratch is mounted as `snow.research.mmteb.xinghan_scratch:/scratch`, so
      `/scratch/epp/...` paths in TOMLs are available inside the job.
    - Workdir is `/home/toolkit/emi_dir/kv-eviction`.
    - Jobs run through the repo-local venv:
      `source .venv/bin/activate && exec rl @ <config>`.
    - If `WANDB_API_KEY` is not already set inside the job, it is loaded from
      `.local_artifacts/.wandb_api_key` without putting the secret in EAI
      metadata.
    - EAI job names are compact and start with `vllm_server_id_...`; W&B keeps
      the full `[wandb].name` from the TOML. Keep `[wandb].name` short for
      eval runs, because W&B incremental table artifacts derive names from it
      and reject artifact names longer than 128 chars.
    - Self-distillation jobs request extra host RAM and enable
      `KVE_MEM_TRACE=1`; the teacher/distill path is memory-sensitive. The
      launcher treats names containing `distill` or the compact `d001` as
      distill runs. If a distill TOML name includes `noacoffload`, the TOML
      intentionally omits `[trainer.model.ac_offloading]` to avoid the
      host-RSS SIGKILL mode seen with activation offload.
    - Runs with `phase4debug` in `[wandb].name` export controlled Phase4
      diagnostics: pin lifecycle tracing, scheduler allocation-blocked
      tracing, and rate-limited pin-hit repeat logging.
    - Runs with `kldebug` in `[wandb].name` export trainer-side KL mismatch
      diagnostics: top mismatching tokens, per-call summaries, and a low
      per-call warning threshold. This is intentionally opt-in because the
      logs are noisy.
    - Job records are stored in `experiments/debug_balrog/jobs/<wandb.name>.json`.

Single-config usage:
    .venv/bin/python experiments/debug_balrog/launch_eai.py \
        experiments/textworld_flex_grid/rl_eai_full_context_seed0_flex_8gpu4x4.toml

    .venv/bin/python experiments/debug_balrog/launch_eai.py \
        experiments/textworld_flex_grid/rl_eai_full_context_seed0_flex_8gpu4x4.toml \
        --dry-run

    .venv/bin/python experiments/debug_balrog/launch_eai.py \
        experiments/textworld_flex_grid/rl_eai_full_context_seed0_flex_8gpu4x4.toml \
        --status

Grid launch / relaunch sweep:
    for cfg in $(find experiments/textworld_flex_grid -maxdepth 1 -name '*.toml' | sort); do
        .venv/bin/python experiments/debug_balrog/launch_eai.py "$cfg"
    done

The launch sweep is also the relaunch command. For each config, the launcher:
    - reads `[wandb].name`;
    - checks `jobs/<wandb.name>.json`;
    - skips the job if EAI reports RUNNING, QUEUED, QUEUING, SUCCEEDED, or
      COMPLETED;
    - submits a new EAI job and overwrites the job record for other states
      such as FAILED, INTERRUPTED, CANCELLED, or UNKNOWN.

Grid status sweep:
    for cfg in $(find experiments/textworld_flex_grid -maxdepth 1 -name '*.toml' | sort); do
        .venv/bin/python experiments/debug_balrog/launch_eai.py "$cfg" --status
    done
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

os.environ.setdefault("EAI_PROFILE", "yul201")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # kv-eviction root
JOBS_DIR = SCRIPT_DIR / "jobs"

IMAGE = "registry.toolkit-sp.yul201.service-now.com/snow.shared/ui_copilot_playwright:latest"
HOME_DATA = "snow.home.xinghan_lu:/home/toolkit:rw"
DATA_MOUNTS = [
    "snow.research.mmteb.xinghan_scratch:/scratch",
]
DEFAULT_GPU_COUNT = 8
GPU_MEM = 80
CPU = 64
DEFAULT_MEM = 256
DISTILL_MEM = 768
WORKDIR = "/home/toolkit/emi_dir/kv-eviction"
VENV_DIR = f"{WORKDIR}/.venv"

ALIVE_STATES = {"RUNNING", "QUEUED", "QUEUING"}
SKIP_RELAUNCH_STATES = ALIVE_STATES | {"SUCCEEDED", "COMPLETED"}
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_account():
    result = subprocess.run(
        ["eai", "account", "get", "--no-header", "--field", "fullName"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Failed to get EAI account: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def get_job_state(job_id):
    result = subprocess.run(
        ["eai", "job", "get", job_id, "--no-header", "--field", "state"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return "UNKNOWN"
    return result.stdout.strip().upper()


def color_state(state):
    if state in ("RUNNING", "SUCCEEDED", "COMPLETED"):
        return f"{GREEN}{state}{RESET}"
    if state in ("QUEUED", "QUEUING"):
        return f"{YELLOW}{state}{RESET}"
    return f"{RED}{state}{RESET}"


def load_job_info(run_name):
    job_file = JOBS_DIR / f"{run_name}.json"
    if job_file.exists():
        return json.loads(job_file.read_text())
    return None


def save_job_info(run_name, job_id, config_rel):
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    info = {
        "job_id": job_id,
        "config": config_rel,
        "run_name": run_name,
        "launched_at": datetime.now().isoformat(),
    }
    (JOBS_DIR / f"{run_name}.json").write_text(json.dumps(info, indent=2) + "\n")


def compact_job_slug(run_name):
    slug = run_name.lower()
    for old, new in (
        ("full-context", "full-ctx"),
        ("markovian", "mk"),
        ("eviction", "evict"),
        ("textworld", "tw"),
        ("turns", "t"),
        ("stride", "s"),
        ("flexmask", "fm"),
        ("distill0p01", "d001"),
    ):
        slug = slug.replace(old, new)
    return re.sub(r"[^a-z0-9]+", "_", slug).strip("_")


def is_distill_run(run_name):
    slug = run_name.lower()
    return "distill" in slug or "d001" in slug


def memory_gb_for_run(run_name):
    return DISTILL_MEM if is_distill_run(run_name) else DEFAULT_MEM


def extra_env_for_run(run_name):
    env = []
    if is_distill_run(run_name):
        env.append("KVE_MEM_TRACE=1")
    if "phase4debug" in run_name.lower():
        env.extend(
            [
                "KVE_PHASE4_PIN_HIT_RATE_LIMIT=1",
                "KVE_TRACE_PHASE4_PIN=1",
                "KVE_TRACE_PHASE4_SCHED=1",
            ]
        )
    if "kldebug" in run_name.lower():
        env.extend(
            [
                "KVE_CALL_MISMATCH_TOP=20",
                "KVE_CALL_MISMATCH_THRESHOLD=0.005",
                "KVE_TOP_MISMATCH_TOKENS=20",
                "KVE_TOP_MISMATCH_SCOPE=loss_mask",
            ]
        )
    return env


def launch_job(run_name, config_rel, account, gpu_count, mem_gb, dry_run=False):
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    job_name = f"vllm_server_id_{compact_job_slug(run_name)}_{timestamp}"
    print(f"  EAI name: {job_name}")
    entrypoint = (
        "if [ -z \"${WANDB_API_KEY:-}\" ] && [ -f .local_artifacts/.wandb_api_key ]; "
        "then export WANDB_API_KEY=\"$(tr -d '\\r\\n' < .local_artifacts/.wandb_api_key)\"; fi; "
        "export WANDB_MODE=\"${WANDB_MODE:-online}\"; "
        f"source .venv/bin/activate && exec rl @ {shlex.quote(config_rel)}"
    )

    cmd = [
        "eai", "job", "new",
        "--account", account,
        "--restartable",
        "--name", job_name,
        "--image", IMAGE,
        "--gpu", str(gpu_count),
        "--gpu-mem", str(GPU_MEM),
        "--cpu", str(CPU),
        "--mem", str(mem_gb),
        "--data", HOME_DATA,
    ]
    for mount in DATA_MOUNTS:
        cmd += ["--data", mount]
    cmd += [
        "--env", "HOME=/home/toolkit",
        "--env", f"VIRTUAL_ENV={VENV_DIR}",
        "--env", f"PATH={VENV_DIR}/bin:/home/toolkit/.local/bin:/usr/local/bin:/usr/bin:/bin",
        # litellm requires OPENAI_API_KEY to be non-empty even when
        # talking to a local vLLM (which ignores the value). tau2-bench
        # user simulator routes through litellm.
        "--env", f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
    ]
    for env_value in extra_env_for_run(run_name):
        cmd += ["--env", env_value]
    cmd += [
        "--tag", "kv-eviction",
        "--tag", run_name,
        "--workdir", WORKDIR,
        "--field", "id",
        "--no-header",
        "--", "bash", "-lc", entrypoint,
    ]

    if dry_run:
        print(f"  {shlex.join(cmd)}")
        return None

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  {RED}ERROR:{RESET} {result.stderr.strip()}")
        return None

    job_id = result.stdout.strip()
    if not job_id:
        print(f"  {RED}ERROR:{RESET} no job ID returned")
        return None
    return job_id


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="TOML filename, repo-relative path, or absolute path")
    parser.add_argument("--dry-run", action="store_true", help="Print command without submitting")
    parser.add_argument("--status", action="store_true", help="Check job status only")
    args = parser.parse_args()

    # Accept absolute paths, repo-relative paths, or bare filenames in SCRIPT_DIR.
    raw = Path(args.config)
    if raw.is_absolute():
        config_path = raw
    elif (REPO_DIR / raw).exists():
        config_path = REPO_DIR / raw
    else:
        config_path = SCRIPT_DIR / raw

    config_path = config_path.resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    run_name = cfg.get("wandb", {}).get("name") or config_path.stem.replace("_", "-")
    config_rel = str(config_path.relative_to(REPO_DIR))
    try:
        gpu_count = int(
            cfg.get("deployment", {}).get("gpus_per_node", DEFAULT_GPU_COUNT)
        )
    except (TypeError, ValueError):
        gpu_count = DEFAULT_GPU_COUNT

    account = get_account()
    info = load_job_info(run_name)

    if args.status:
        if info:
            state = get_job_state(info["job_id"])
            print(f"  {color_state(state):>20}  {run_name}  (job {info['job_id']})")
        else:
            print(f"  No job recorded for {run_name}")
        return

    if info:
        state = get_job_state(info["job_id"])
        print(f"  {color_state(state):>20}  {run_name}  (job {info['job_id']})")
        if state in SKIP_RELAUNCH_STATES:
            print("  Job does not need relaunch — skipping. Use --status to check.")
            return

    print(f"{BOLD}Launching {run_name} — {account}{RESET}")
    print(f"  Config:  {config_rel}")
    print(f"  GPUs:    {gpu_count}x H100-{GPU_MEM}GB")
    mem_gb = memory_gb_for_run(run_name)
    print(f"  Mem:     {mem_gb}GB")
    print(f"  Workdir: {WORKDIR}")

    job_id = launch_job(
        run_name, config_rel, account, gpu_count, mem_gb, dry_run=args.dry_run,
    )
    if job_id:
        save_job_info(run_name, job_id, config_rel)
        print(f"  -> job {job_id}")


if __name__ == "__main__":
    main()
