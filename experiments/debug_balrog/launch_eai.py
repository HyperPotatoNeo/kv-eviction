#!/usr/bin/env python3
"""Launch any RL TOML config on EAI with 8 H100-80GB GPUs.

    python experiments/debug_balrog/launch_eai.py rl_full.toml
    python experiments/debug_balrog/launch_eai.py rl_no_eviction.toml
    python experiments/debug_balrog/launch_eai.py rl_full.toml --dry-run
    python experiments/debug_balrog/launch_eai.py rl_full.toml --status
    python experiments/debug_balrog/launch_eai.py experiments/compaction_rgmix/rl.toml
    python experiments/debug_balrog/launch_eai.py /absolute/path/to/rl.toml
"""

import argparse
import json
import os
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
HOME_DATA = "snow.research.adea.emiliano_home:/home/toolkit"
DATA_MOUNTS = [
    "snow.research.ui_assist.data:/mnt/ui_assist/data:ro",
    "snow.research.ui_assist.data:/mnt/ui_assist/data_rw",
    "snow.research.adea.data:/mnt/adea/data:ro",
    "snow.research.adea.data:/mnt/adea/data_rw",
]
GPU_COUNT = 8
GPU_MEM = 80
CPU = 64
MEM = 256
WORKDIR = "/home/toolkit/kv-eviction"

ALIVE_STATES = {"RUNNING", "QUEUED", "QUEUING"}
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
    if state == "RUNNING":
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


def launch_job(run_name, config_rel, account, dry_run=False):
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    job_name = f"{run_name.replace('-', '_')}_{timestamp}"
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    cmd = [
        "eai", "job", "new",
        "--account", account,
        "--restartable",
        "--name", job_name,
        "--image", IMAGE,
        "--gpu", str(GPU_COUNT),
        "--gpu-mem", str(GPU_MEM),
        "--cpu", str(CPU),
        "--mem", str(MEM),
        "--data", HOME_DATA,
    ]
    for mount in DATA_MOUNTS:
        cmd += ["--data", mount]
    cmd += [
        "--env", "HOME=/home/toolkit",
        "--env", "PATH=/home/toolkit/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "--env", f"WANDB_API_KEY={wandb_key}",
        # litellm requires OPENAI_API_KEY to be non-empty even when
        # talking to a local vLLM (which ignores the value). tau2-bench
        # user simulator routes through litellm.
        "--env", f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
        "--tag", "kv-eviction",
        "--tag", run_name,
        "--workdir", WORKDIR,
        "--field", "id",
        "--no-header",
        "--", "uv", "run", "rl", "@", config_rel,
    ]

    if dry_run:
        print(f"  {' '.join(cmd)}")
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
    parser.add_argument("config", help="TOML filename in experiments/debug_balrog/ (e.g. rl_full.toml)")
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
        if state in ALIVE_STATES:
            print("  Job is alive — skipping relaunch. Use --status to check.")
            return

    print(f"{BOLD}Launching {run_name} — {account}{RESET}")
    print(f"  Config:  {config_rel}")
    print(f"  GPUs:    {GPU_COUNT}x H100-{GPU_MEM}GB")
    print(f"  Workdir: {WORKDIR}")

    job_id = launch_job(run_name, config_rel, account, dry_run=args.dry_run)
    if job_id:
        save_job_info(run_name, job_id, config_rel)
        print(f"  -> job {job_id}")


if __name__ == "__main__":
    main()
