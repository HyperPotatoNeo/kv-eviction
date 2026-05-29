#!/usr/bin/env python3
"""Build TextWorld experiment reports from local configs, logs, and W&B.

The tracker is intentionally human-owned. This script can initialize it from
local TOMLs, then resolves status/metrics/failure classes into generated CSV,
SVG, and Markdown artifacts. W&B history is optional and cached locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - this repo's venv has pyyaml.
    raise SystemExit("PyYAML is required: install pyyaml or run inside .venv") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKER = REPO_ROOT / "experiments/_reports/tracker.yaml"
DEFAULT_OUT = REPO_ROOT / "experiments/_reports/reports/textworld_main.md"
DEFAULT_PLOTS_DIR = REPO_ROOT / "experiments/_reports/reports/plots"
DEFAULT_CACHE_DIR = REPO_ROOT / "experiments/_reports/cache"
DEFAULT_JOBS_DIR = REPO_ROOT / "experiments/debug_balrog/jobs"
DEFAULT_CONFIG_ROOT = REPO_ROOT / "experiments/_local_jobs"
DEFAULT_WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "laurent-charlin")
DEFAULT_GPU_COUNT = 8

METRIC_KEYS = {
    "success_rate": "eval/textworld-env/success_rate",
    "hard_success_rate": "eval/textworld-env/hard_success_rate",
    "hard_success_count": "eval/textworld-env/hard_success_count",
    "avg_score": "eval/textworld-env/avg@1",
    "train_reward": "reward/textworld-env/mean",
    "step_time_s": "time/step",
}

FAILURE_PATTERNS = [
    ("checkpoint_corrupt", re.compile(r"metadata is None|CheckpointException", re.I)),
    ("trainer_sigkill", re.compile(r"Signal 9|exitcode\s*:\s*-9|failed \(exitcode: -9\)", re.I)),
    ("cuda_oom", re.compile(r"CUDA out of memory|OutOfMemoryError", re.I)),
    ("phase4_kv_mismatch", re.compile(r"PHASE4.*(mismatch|expected)|retained KV|PIN-MISS", re.I)),
    ("inference_died", re.compile(r"VLLM_DP_Coordinator.*died|Executor failed|RuntimeError: cancelled", re.I)),
    ("http_read_error", re.compile(r"httpx\.ReadError|httpcore\.ReadError", re.I)),
]


@dataclass
class ResolvedRun:
    run_name: str
    family: str
    variant: str
    config: str
    scratch_dir: str
    wandb_project: str
    wandb_id: str
    eai_job_id: str
    turns: int | None
    stride: int | None
    seed: int | None
    max_steps: int | None
    status: str
    result_use: str
    relaunch: str
    failure_class: str
    failure_source: str
    latest_step: int | None
    success_rate: float | None
    hard_success_rate: float | None
    hard_success_count: float | None
    avg_score: float | None
    train_reward: float | None
    notes: str


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict in {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, width=120)


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def infer_family(config_path: Path, run_name: str) -> str:
    text = f"{config_path.as_posix()} {run_name}".lower()
    if "distill" in text or "-d001-" in text:
        return "self_distill"
    if "full_context" in text or "full-context" in text:
        return "full_context"
    if "markovian" in text:
        return "markovian"
    if "evict" in text or "eviction" in text or "flexmask" in text:
        return "kv_eviction"
    return "debug"


def infer_variant(config_path: Path, run_name: str, family: str) -> str:
    text = f"{config_path.name} {run_name}".lower()
    if "phase4debug" in text:
        return "phase4debug"
    if "fixkl" in text:
        return "fixkl"
    if "distill" in text or family == "self_distill":
        return "distill0p01" if "0p01" in text or "d001" in text else "distill"
    if "flexmask" in text:
        return "flexmask"
    if "flex" in text:
        return "flex"
    return family


def infer_turn_stride_seed(run_name: str, config_path: Path) -> tuple[int | None, int | None, int | None]:
    text = f"{run_name} {config_path.name}"
    turns = None
    stride = None
    seed = None
    for pattern in (r"turns?(\d+)", r"\bt(\d+)\b"):
        match = re.search(pattern, text)
        if match:
            turns = int(match.group(1))
            break
    for pattern in (r"stride(\d+)", r"\bs(\d+)\b"):
        match = re.search(pattern, text)
        if match:
            stride = int(match.group(1))
            break
    match = re.search(r"seed(\d+)", text)
    if match:
        seed = int(match.group(1))
    return turns, stride, seed


def load_job_records(jobs_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not jobs_dir.exists():
        return records
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_name = data.get("run_name") or path.stem
        records[str(run_name)] = data
    return records


def config_to_run(path: Path, job_records: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    cfg = read_toml(path)
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict) or not wandb_cfg.get("name"):
        return None
    run_name = str(wandb_cfg["name"])
    family = infer_family(path, run_name)
    turns, stride, seed = infer_turn_stride_seed(run_name, path)
    job = job_records.get(run_name, {})
    return {
        "run_name": run_name,
        "family": family,
        "variant": infer_variant(path, run_name, family),
        "turns": turns,
        "stride": stride,
        "seed": seed,
        "config": relpath(path),
        "scratch_dir": str(cfg.get("output_dir", "")),
        "wandb_project": str(wandb_cfg.get("project", "")),
        "wandb_id": str(wandb_cfg.get("id", run_name)),
        "eai_job_id": str(job.get("job_id", "")),
        "result_use": "auto",
        "relaunch": "auto",
        "notes": "",
    }


def is_paper_config(path: Path, run: dict[str, Any]) -> bool:
    name = path.name
    family = str(run["family"])
    run_name = str(run["run_name"])
    if family == "full_context":
        return "8gpu4x4" in name and "full-context" in run_name
    if family == "markovian":
        return "8gpu4x4" in name and "markovian-textworld" in run_name
    if family == "kv_eviction":
        return "8gpu4x4" in name and "distill" not in name and "tw-evict-" in run_name
    if family == "self_distill":
        return "distill0p01_8gpu4x4" in name and "tw-evict-" in run_name
    return False


def iter_config_paths(config_root: Path) -> list[Path]:
    families = ["full_context", "markovian", "kv_eviction", "self_distill"]
    paths: list[Path] = []
    for family in families:
        paths.extend(sorted((config_root / family).glob("*.toml")))
    return paths


def init_tracker(tracker: Path, config_root: Path, jobs_dir: Path, force: bool) -> None:
    if tracker.exists() and not force:
        raise SystemExit(f"{tracker} already exists; use --force to overwrite")
    job_records = load_job_records(jobs_dir)
    runs = []
    seen: set[str] = set()
    for path in iter_config_paths(config_root):
        run = config_to_run(path, job_records)
        if run is None:
            continue
        if not is_paper_config(path, run):
            continue
        key = run["run_name"]
        if key in seen:
            continue
        seen.add(key)
        runs.append(run)
    data = {
        "paper": {
            "title": "TextWorld eviction comparison",
            "dataset": "textworld_cooking_mix",
            "wandb_entity": DEFAULT_WANDB_ENTITY,
            "metric_keys": METRIC_KEYS,
            "families": {
                "full_context": "Full-context baseline.",
                "markovian": "Markovian thinker baseline without KV eviction.",
                "kv_eviction": "Turn-based KV eviction / flexmask runs.",
                "self_distill": "KV eviction plus self-distillation.",
            },
            "runs": sorted(runs, key=lambda r: (r["family"], r.get("turns") or 0, r.get("stride") or 0, r["run_name"])),
        }
    }
    write_yaml(tracker, data)
    print(f"Wrote {tracker} with {len(runs)} paper runs")


def tracker_runs(data: dict[str, Any]) -> list[dict[str, Any]]:
    paper = data.get("paper", {})
    if not isinstance(paper, dict):
        raise ValueError("Tracker must contain a top-level 'paper' dict")
    runs = paper.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("Tracker 'paper.runs' must be a list")
    return [r for r in runs if isinstance(r, dict)]


def read_text_limited(path: Path, max_bytes: int = 8_000_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(max(0, size - max_bytes))
        return f.read().decode("utf-8", errors="ignore")


def classify_failure(scratch_dir: Path) -> tuple[str, str]:
    logs = {
        "trainer.log": read_text_limited(scratch_dir / "logs/trainer.log"),
        "orchestrator.log": read_text_limited(scratch_dir / "logs/orchestrator.log"),
        "inference.log": read_text_limited(scratch_dir / "logs/inference.log", max_bytes=4_000_000),
    }
    # Prefer root-cause sources over downstream shutdown noise.
    for source in ("trainer.log", "orchestrator.log", "inference.log"):
        text = logs[source]
        for failure_class, pattern in FAILURE_PATTERNS:
            if pattern.search(text):
                return failure_class, source
    return "none", ""


def latest_step_from_dirs(scratch_dir: Path) -> int | None:
    steps: list[int] = []
    for root in [
        scratch_dir / "checkpoints",
        scratch_dir / "run_default/checkpoints",
        scratch_dir / "run_default/broadcasts",
    ]:
        if not root.exists():
            continue
        for child in root.iterdir():
            match = re.match(r"step_(\d+)$", child.name)
            if match:
                steps.append(int(match.group(1)))
    return max(steps) if steps else None


def summary_paths(scratch_dir: Path) -> list[Path]:
    roots = [
        scratch_dir / "run_default/wandb",
        scratch_dir / "wandb",
    ]
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("run-*/files/wandb-summary.json"):
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    return sorted(paths, key=lambda p: p.stat().st_mtime if p.exists() else 0)


def wandb_binary_paths(scratch_dir: Path) -> list[Path]:
    roots = [
        scratch_dir / "run_default/wandb",
        scratch_dir / "wandb",
    ]
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("run-*/run-*.wandb"):
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    return sorted(paths, key=lambda p: p.stat().st_mtime if p.exists() else 0)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def normalize_metric_row(run: dict[str, Any], source: str, source_path: str, data: dict[str, Any]) -> dict[str, Any] | None:
    if not any(key in data for key in METRIC_KEYS.values()):
        return None
    step = as_int(data.get("progress/ckpt_step")) or as_int(data.get("step")) or as_int(data.get("_step"))
    row = {
        "run_name": run.get("run_name", ""),
        "family": run.get("family", ""),
        "variant": run.get("variant", ""),
        "turns": run.get("turns"),
        "stride": run.get("stride"),
        "seed": run.get("seed"),
        "step": step,
        "source": source,
        "source_path": source_path,
    }
    for short_key, wandb_key in METRIC_KEYS.items():
        row[short_key] = as_float(data.get(wandb_key))
    return row


def extract_local_metrics(run: dict[str, Any]) -> list[dict[str, Any]]:
    scratch = Path(str(run.get("scratch_dir", "")))
    rows: list[dict[str, Any]] = []
    for path in summary_paths(scratch):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row = normalize_metric_row(run, "local_summary", relpath(path), data)
        if row is None:
            continue
        rows.append(row)
    return rows


def extract_local_wandb_history(run: dict[str, Any]) -> list[dict[str, Any]]:
    scratch = Path(str(run.get("scratch_dir", "")))
    paths = wandb_binary_paths(scratch)
    if not paths:
        return []
    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
    except ImportError:
        return []

    wanted = {"_step", "step", "progress/ckpt_step", *METRIC_KEYS.values()}
    rows: list[dict[str, Any]] = []
    for path in paths:
        store = DataStore()
        try:
            store.open_for_scan(str(path))
            while True:
                data = store.scan_data()
                if data is None:
                    break
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                if record.WhichOneof("record_type") != "history":
                    continue
                history: dict[str, Any] = {}
                for item in record.history.item:
                    key = "/".join(item.nested_key)
                    if key not in wanted:
                        continue
                    try:
                        history[key] = json.loads(item.value_json)
                    except json.JSONDecodeError:
                        history[key] = item.value_json
                row = normalize_metric_row(run, "local_wandb_history", relpath(path), history)
                if row is not None:
                    rows.append(row)
        except Exception:
            continue
        finally:
            store.close()
    return rows


def extract_wandb_metrics(
    run: dict[str, Any],
    entity: str,
    cache_dir: Path,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    project = str(run.get("wandb_project", ""))
    run_id = str(run.get("wandb_id") or run.get("run_name") or "")
    run_name = str(run.get("run_name") or run_id)
    if not entity or not project or not run_id:
        return []

    cache_path = cache_dir / "wandb_history" / f"{sanitize_filename(run_name)}.jsonl"
    if cache_path.exists() and not refresh:
        return load_jsonl(cache_path)

    try:
        import wandb
    except ImportError:
        return []

    rows: list[dict[str, Any]] = []
    try:
        api = wandb.Api()
        wb_run = api.run(f"{entity}/{project}/{run_id}")
        source_path = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
        history_keys = ["_step", "step", "progress/ckpt_step", *METRIC_KEYS.values()]
        for item in wb_run.scan_history(keys=history_keys, page_size=1000):
            row = normalize_metric_row(run, "wandb", source_path, item)
            if row is not None:
                rows.append(row)
    except Exception as exc:  # W&B failures should not block local-log reporting.
        rows = [
            {
                "run_name": run_name,
                "family": run.get("family", ""),
                "variant": run.get("variant", ""),
                "turns": run.get("turns"),
                "stride": run.get("stride"),
                "seed": run.get("seed"),
                "step": None,
                "source": "wandb_api_error",
                "source_path": f"{type(exc).__name__}: {exc}",
                "success_rate": None,
                "hard_success_rate": None,
                "hard_success_count": None,
                "avg_score": None,
                "train_reward": None,
                "step_time_s": None,
            }
        ]
    write_jsonl(cache_path, rows)
    return rows


def extract_metrics(
    run: dict[str, Any],
    use_wandb: bool = False,
    wandb_entity: str = "",
    cache_dir: Path | None = None,
    refresh_wandb: bool = False,
) -> list[dict[str, Any]]:
    rows = extract_local_metrics(run)
    rows.extend(extract_local_wandb_history(run))
    if use_wandb and cache_dir is not None:
        rows.extend(extract_wandb_metrics(run, wandb_entity, cache_dir, refresh=refresh_wandb))
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("run_name"),
            row.get("step"),
            row.get("source"),
            row.get("success_rate"),
            row.get("hard_success_rate"),
            row.get("hard_success_count"),
            row.get("avg_score"),
            row.get("train_reward"),
            row.get("step_time_s"),
        )
        deduped[key] = row
    return list(deduped.values())


def latest_metric(metrics: list[dict[str, Any]], key: str) -> float | None:
    candidates = [m for m in metrics if m.get(key) is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda m: (m.get("step") is not None, m.get("step") or -1, m.get("source_path", "")))
    return as_float(candidates[-1].get(key))


def latest_metric_step(metrics: list[dict[str, Any]]) -> int | None:
    steps = [as_int(m.get("step")) for m in metrics]
    steps = [s for s in steps if s is not None]
    return max(steps) if steps else None


def resolve_status(failure_class: str, latest_step: int | None, max_steps: int | None, scratch_dir: Path) -> str:
    if failure_class != "none":
        return "failed"
    if max_steps is not None and latest_step is not None and latest_step >= max_steps:
        return "complete"
    logs = scratch_dir / "logs"
    if logs.exists():
        newest = max((p.stat().st_mtime for p in logs.glob("*.log")), default=0)
        if newest and (time.time() - newest) < 2 * 3600:
            return "running"
        return "partial"
    return "unknown"


def resolve_result_use(raw: str, family: str, status: str, failure_class: str, has_metric: bool) -> str:
    if raw and raw != "auto":
        return raw
    if family == "debug":
        return "debug"
    if status == "complete" and has_metric:
        return "candidate"
    if status in {"running", "partial", "unknown"} and has_metric:
        return "candidate"
    if status == "failed" and has_metric and failure_class not in {"checkpoint_corrupt", "phase4_kv_mismatch"}:
        return "partial"
    return "exclude"


def resolve_relaunch(raw: str, status: str, failure_class: str) -> str:
    if raw and raw != "auto":
        return raw
    if status != "failed":
        return "no"
    if failure_class in {"trainer_sigkill", "checkpoint_corrupt", "phase4_kv_mismatch", "cuda_oom"}:
        return "yes"
    return "maybe"


def resolve_run(
    run: dict[str, Any],
    use_wandb: bool = False,
    wandb_entity: str = "",
    cache_dir: Path | None = None,
    refresh_wandb: bool = False,
) -> tuple[ResolvedRun, list[dict[str, Any]]]:
    config = Path(str(run.get("config", "")))
    if not config.is_absolute():
        config = REPO_ROOT / config
    cfg = read_toml(config) if config.exists() else {}
    scratch = Path(str(run.get("scratch_dir") or cfg.get("output_dir") or ""))
    metrics = extract_metrics(
        run,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        cache_dir=cache_dir,
        refresh_wandb=refresh_wandb,
    )
    failure_class, failure_source = classify_failure(scratch)
    metric_step = latest_metric_step(metrics)
    dir_step = latest_step_from_dirs(scratch)
    latest_step = max([s for s in [metric_step, dir_step] if s is not None], default=None)
    max_steps = as_int(run.get("max_steps")) or as_int(cfg.get("max_steps"))
    status = resolve_status(failure_class, latest_step, max_steps, scratch)
    success = latest_metric(metrics, "success_rate")
    hard_success = latest_metric(metrics, "hard_success_rate")
    hard_count = latest_metric(metrics, "hard_success_count")
    avg_score = latest_metric(metrics, "avg_score")
    train_reward = latest_metric(metrics, "train_reward")
    has_metric = success is not None or hard_success is not None
    result_use = resolve_result_use(str(run.get("result_use", "auto")), str(run.get("family", "")), status, failure_class, has_metric)
    relaunch = resolve_relaunch(str(run.get("relaunch", "auto")), status, failure_class)
    return (
        ResolvedRun(
            run_name=str(run.get("run_name", "")),
            family=str(run.get("family", "")),
            variant=str(run.get("variant", "")),
            config=str(run.get("config", "")),
            scratch_dir=str(scratch),
            wandb_project=str(run.get("wandb_project", "")),
            wandb_id=str(run.get("wandb_id", run.get("run_name", ""))),
            eai_job_id=str(run.get("eai_job_id", "")),
            turns=as_int(run.get("turns")),
            stride=as_int(run.get("stride")),
            seed=as_int(run.get("seed")),
            max_steps=max_steps,
            status=status,
            result_use=result_use,
            relaunch=relaunch,
            failure_class=failure_class,
            failure_source=failure_source,
            latest_step=latest_step,
            success_rate=success,
            hard_success_rate=hard_success,
            hard_success_count=hard_count,
            avg_score=avg_score,
            train_reward=train_reward,
            notes=str(run.get("notes", "")),
        ),
        metrics,
    )


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def row_dict(run: ResolvedRun) -> dict[str, Any]:
    return {
        "family": run.family,
        "run_name": run.run_name,
        "variant": run.variant,
        "turns": run.turns,
        "stride": run.stride,
        "seed": run.seed,
        "status": run.status,
        "result_use": run.result_use,
        "relaunch": run.relaunch,
        "failure_class": run.failure_class,
        "failure_source": run.failure_source,
        "latest_step": run.latest_step,
        "success_rate": run.success_rate,
        "hard_success_rate": run.hard_success_rate,
        "hard_success_count": run.hard_success_count,
        "avg_score": run.avg_score,
        "train_reward": run.train_reward,
        "scratch_dir": run.scratch_dir,
        "config": run.config,
        "eai_job_id": run.eai_job_id,
        "notes": run.notes,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


FAMILY_COLORS = {
    "full_context": "#2563eb",
    "markovian": "#059669",
    "kv_eviction": "#d97706",
    "self_distill": "#7c3aed",
    "debug": "#6b7280",
}

FAMILY_LABELS = {
    "full_context": "Full",
    "markovian": "Markov",
    "kv_eviction": "Evict",
    "self_distill": "Distill",
    "debug": "Debug",
}


def short_run_label(run: ResolvedRun) -> str:
    if run.family == "full_context":
        return "full context"
    turn = run.turns if run.turns is not None else "-"
    stride = run.stride if run.stride is not None else "-"
    return f"{run.family} t{turn} s{stride}"


def family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family)


def line_style(run: ResolvedRun) -> tuple[str, str]:
    if run.result_use in {"primary", "candidate"}:
        return "0.95", ""
    if run.result_use == "partial":
        return "0.70", ' stroke-dasharray="6 4"'
    return "0.42", ' stroke-dasharray="2 4"'


def metric_series_by_run(
    runs: list[ResolvedRun],
    metric_rows: list[dict[str, Any]],
    metric: str,
    include_excluded: bool = False,
) -> dict[str, list[tuple[int, float]]]:
    run_by_name = {r.run_name: r for r in runs}
    grouped: dict[str, dict[float, float]] = defaultdict(dict)
    for row in metric_rows:
        run_name = str(row.get("run_name", ""))
        run = run_by_name.get(run_name)
        if run is None:
            continue
        if run.result_use == "exclude" and not include_excluded:
            continue
        step = as_int(row.get("step"))
        value = as_float(row.get(metric))
        if step is None or value is None:
            continue
        grouped[run_name][step] = value
    return {name: sorted(points.items()) for name, points in grouped.items() if points}


def metric_series_by_gpu_hours(
    runs: list[ResolvedRun],
    metric_rows: list[dict[str, Any]],
    metric: str,
    gpus: int = DEFAULT_GPU_COUNT,
    include_excluded: bool = False,
) -> dict[str, list[tuple[float, float]]]:
    run_by_name = {r.run_name: r for r in runs}
    rows_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        run_name = str(row.get("run_name", ""))
        run = run_by_name.get(run_name)
        if run is None:
            continue
        if run.result_use == "exclude" and not include_excluded:
            continue
        if as_int(row.get("step")) is None:
            continue
        if as_float(row.get(metric)) is None and as_float(row.get("step_time_s")) is None:
            continue
        rows_by_run[run_name].append(row)

    series: dict[str, list[tuple[float, float]]] = {}
    for run_name, rows in rows_by_run.items():
        step_times: dict[int, float] = {}
        values: dict[int, float] = {}
        for row in sorted(rows, key=lambda r: (as_int(r.get("step")) or -1, str(r.get("source_path", "")))):
            step = as_int(row.get("step"))
            if step is None:
                continue
            step_time = as_float(row.get("step_time_s"))
            if step_time is not None and step_time >= 0 and step not in step_times:
                step_times[step] = step_time
            value = as_float(row.get(metric))
            if value is not None:
                values[step] = value

        cumulative_seconds = 0.0
        points: list[tuple[float, float]] = []
        for step in sorted(set(step_times) | set(values)):
            if step in step_times:
                cumulative_seconds += step_times[step]
            if step in values:
                gpu_hours = cumulative_seconds * gpus / 3600.0
                points.append((gpu_hours, values[step]))
        if points:
            series[run_name] = points
    return series


def smooth_series(points: list[tuple[float, float]], window: int = 10) -> list[tuple[float, float]]:
    if window <= 1 or len(points) <= 1:
        return points
    smoothed: list[tuple[float, float]] = []
    values: list[float] = []
    for step, value in points:
        values.append(value)
        current = values[-window:]
        smoothed.append((step, sum(current) / len(current)))
    return smoothed


def write_bar_svg(path: Path, runs: list[ResolvedRun], metric: str, title: str) -> None:
    values = [(r, getattr(r, metric)) for r in runs if getattr(r, metric) is not None and r.result_use != "exclude"]
    values.sort(key=lambda item: (item[0].family, item[0].turns or 0, item[0].stride or 0, item[0].run_name))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="80">'
            '<rect width="100%" height="100%" fill="#ffffff"/>'
            '<text x="20" y="45" font-family="sans-serif" font-size="16">No metric data found.</text></svg>\n',
            encoding="utf-8",
        )
        return
    left = 310
    right = 40
    top = 54
    row_h = 24
    width = 960
    bar_w = width - left - right
    height = top + len(values) * row_h + 40
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="20" y="28" font-family="sans-serif" font-size="20" font-weight="700">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top - 14}" x2="{left + bar_w}" y2="{top - 14}" stroke="#d1d5db"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = left + tick * bar_w
        lines.append(f'<line x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{height - 24}" stroke="#eef2f7"/>')
        lines.append(f'<text x="{x - 8:.1f}" y="{height - 8}" font-family="sans-serif" font-size="11" fill="#4b5563">{tick:.2g}</text>')
    for i, (run, value) in enumerate(values):
        y = top + i * row_h
        label = f"{short_run_label(run)} {run.variant}"
        bar_len = max(0.0, min(1.0, float(value))) * bar_w
        opacity = "1.0" if run.result_use in {"primary", "candidate"} else "0.45"
        dash = ' stroke-dasharray="4 2"' if run.result_use == "partial" else ""
        lines.append(f'<text x="20" y="{y + 14}" font-family="sans-serif" font-size="12">{svg_escape(label)}</text>')
        lines.append(
            f'<rect x="{left}" y="{y}" width="{bar_len:.1f}" height="16" '
            f'fill="{FAMILY_COLORS.get(run.family, "#6b7280")}" opacity="{opacity}"{dash}/>'
        )
        lines.append(f'<text x="{left + bar_len + 6:.1f}" y="{y + 13}" font-family="sans-serif" font-size="12">{value:.3f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_line_svg(path: Path, runs: list[ResolvedRun], metric_rows: list[dict[str, Any]], metric: str, title: str) -> None:
    run_by_name = {r.run_name: r for r in runs}
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row in metric_rows:
        run_name = str(row.get("run_name", ""))
        run = run_by_name.get(run_name)
        if run is None or run.result_use == "exclude":
            continue
        step = as_int(row.get("step"))
        value = as_float(row.get(metric))
        if step is None or value is None:
            continue
        grouped[run_name][step] = value

    series = [
        (run_by_name[name], sorted(points.items()))
        for name, points in grouped.items()
        if name in run_by_name and points
    ]
    series.sort(key=lambda item: (item[0].family, item[0].turns or 0, item[0].stride or 0, item[0].run_name))

    path.parent.mkdir(parents=True, exist_ok=True)
    if not series:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="80">'
            '<rect width="100%" height="100%" fill="#ffffff"/>'
            '<text x="20" y="45" font-family="sans-serif" font-size="16">No metric history found.</text></svg>\n',
            encoding="utf-8",
        )
        return

    x_max = max(step for _, points in series for step, _ in points)
    y_max = max(1.0, max(value for _, points in series for _, value in points))
    left = 72
    right = 275
    top = 52
    bottom = 52
    width = 1100
    plot_h = 330
    legend_h = max(0, len(series) * 18 - plot_h)
    height = top + plot_h + bottom + legend_h
    plot_w = width - left - right

    def sx(step: int) -> float:
        return left + (step / max(1, x_max)) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value / y_max) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="20" y="28" font-family="sans-serif" font-size="20" font-weight="700">{svg_escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#d1d5db"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(tick * y_max)
        label = tick * y_max
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eef2f7"/>')
        lines.append(f'<text x="24" y="{y + 4:.1f}" font-family="sans-serif" font-size="11" fill="#4b5563">{label:.2g}</text>')
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = left + tick * plot_w
        step = int(round(tick * x_max))
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        lines.append(f'<text x="{x - 12:.1f}" y="{top + plot_h + 22}" font-family="sans-serif" font-size="11" fill="#4b5563">{step}</text>')
    lines.append(f'<text x="{left + plot_w / 2 - 24:.1f}" y="{height - 12}" font-family="sans-serif" font-size="12" fill="#374151">step</text>')

    for run, points in series:
        if len(points) == 1:
            x = sx(points[0][0])
            y = sy(points[0][1])
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{FAMILY_COLORS.get(run.family, "#6b7280")}" />'
            )
            continue
        point_text = " ".join(f"{sx(step):.1f},{sy(value):.1f}" for step, value in points)
        dash = ' stroke-dasharray="5 3"' if run.result_use == "partial" else ""
        opacity = "0.55" if run.result_use == "partial" else "0.95"
        lines.append(
            f'<polyline points="{point_text}" fill="none" stroke="{FAMILY_COLORS.get(run.family, "#6b7280")}" '
            f'stroke-width="2" opacity="{opacity}"{dash}/>'
        )

    legend_x = left + plot_w + 22
    for i, (run, points) in enumerate(series):
        y = top + 14 + i * 18
        dash = ' stroke-dasharray="5 3"' if run.result_use == "partial" else ""
        label = f"{short_run_label(run)} latest={points[-1][1]:.3f} step={points[-1][0]}"
        lines.append(
            f'<line x1="{legend_x}" y1="{y - 4}" x2="{legend_x + 18}" y2="{y - 4}" '
            f'stroke="{FAMILY_COLORS.get(run.family, "#6b7280")}" stroke-width="2"{dash}/>'
        )
        lines.append(f'<text x="{legend_x + 24}" y="{y}" font-family="sans-serif" font-size="11">{svg_escape(label)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_facet_svg(
    path: Path,
    runs: list[ResolvedRun],
    metric_rows: list[dict[str, Any]],
    metric: str,
    title: str,
    smooth_window: int = 1,
    x_axis: str = "step",
    gpus: int = DEFAULT_GPU_COUNT,
) -> None:
    runs_sorted = sorted(runs, key=lambda r: (r.family, r.turns or 0, r.stride or 0, r.run_name))
    non_full = [r for r in runs_sorted if r.family != "full_context"]
    full_context = [r for r in runs_sorted if r.family == "full_context"]
    turns = sorted({r.turns for r in non_full if r.turns is not None})
    strides = sorted({r.stride for r in non_full if r.stride is not None})
    if x_axis == "gpu_hours":
        series_by_run = metric_series_by_gpu_hours(runs_sorted, metric_rows, metric, gpus=gpus, include_excluded=True)
    else:
        series_by_run = metric_series_by_run(runs_sorted, metric_rows, metric, include_excluded=True)
    if smooth_window > 1:
        series_by_run = {name: smooth_series(points, smooth_window) for name, points in series_by_run.items()}

    path.parent.mkdir(parents=True, exist_ok=True)
    if not turns or not strides or not series_by_run:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="90">'
            '<rect width="100%" height="100%" fill="#ffffff"/>'
            '<text x="20" y="50" font-family="sans-serif" font-size="16">No faceted metric data found.</text></svg>\n',
            encoding="utf-8",
        )
        return

    all_points = [point for points in series_by_run.values() for point in points]
    x_max = max(x for x, _ in all_points)
    y_max = max(1.0, max(value for _, value in all_points))
    cell_w = 500
    cell_h = 250
    left = 58
    top = 114 if smooth_window > 1 else 98
    right = 36
    bottom = 38
    width = left + len(strides) * cell_w + right
    height = top + len(turns) * cell_h + bottom
    plot_left_pad = 44
    plot_top_pad = 42
    plot_w = 270
    plot_h = 154

    def sx(cell_x: int, x_value: float) -> float:
        return cell_x + plot_left_pad + (x_value / max(1.0, x_max)) * plot_w

    def sy(cell_y: int, value: float) -> float:
        return cell_y + plot_top_pad + plot_h - (value / y_max) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="20" y="30" font-family="sans-serif" font-size="22" font-weight="700" fill="#111827">{svg_escape(title)}</text>',
        f'<text x="20" y="54" font-family="sans-serif" font-size="12" fill="#4b5563">{svg_escape("Rows are max_turns, columns are stride. Full context is repeated in every panel. Dotted lines are excluded runs; dashed lines are partial runs.")}</text>',
    ]
    if smooth_window > 1:
        lines.append(
            f'<text x="20" y="70" font-family="sans-serif" font-size="12" fill="#4b5563">Training reward shown as trailing running average, window={smooth_window} logged points.</text>'
        )
    if x_axis == "gpu_hours":
        lines.append(
            f'<text x="430" y="70" font-family="sans-serif" font-size="12" fill="#4b5563">x-axis is cumulative GPU-hours from time/step using {gpus} GPUs.</text>'
        )

    legend_x = 20
    legend_y = 92 if smooth_window > 1 else 78
    for family in ["full_context", "markovian", "kv_eviction", "self_distill"]:
        color = FAMILY_COLORS[family]
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 18}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(
            f'<text x="{legend_x + 24}" y="{legend_y + 4}" font-family="sans-serif" font-size="12" fill="#111827">{family_label(family)}</text>'
        )
        legend_x += 110

    for col, stride in enumerate(strides):
        x = left + col * cell_w + plot_left_pad + plot_w / 2 - 30
        lines.append(
            f'<text x="{x:.1f}" y="{top - 18}" font-family="sans-serif" font-size="13" font-weight="700" fill="#111827">stride={stride}</text>'
        )
    for row, turn in enumerate(turns):
        y = top + row * cell_h + plot_top_pad + plot_h / 2
        lines.append(
            f'<text x="14" y="{y:.1f}" font-family="sans-serif" font-size="13" font-weight="700" fill="#111827" transform="rotate(-90 14 {y:.1f})">max_turns={turn}</text>'
        )

    for row, turn in enumerate(turns):
        for col, stride in enumerate(strides):
            cell_x = left + col * cell_w
            cell_y = top + row * cell_h
            plot_x = cell_x + plot_left_pad
            plot_y = cell_y + plot_top_pad
            lines.append(
                f'<rect x="{cell_x + 8}" y="{cell_y + 6}" width="{cell_w - 16}" height="{cell_h - 14}" fill="#ffffff" stroke="#d1d5db"/>'
            )
            lines.append(
                f'<text x="{cell_x + 18}" y="{cell_y + 26}" font-family="sans-serif" font-size="12" fill="#374151">t={turn}, s={stride}</text>'
            )
            lines.append(
                f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#e5e7eb"/>'
            )

            for tick in [0.0, 0.5, 1.0]:
                y = sy(cell_y, tick * y_max)
                lines.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#f3f4f6"/>')
                lines.append(
                    f'<text x="{plot_x - 30}" y="{y + 4:.1f}" font-family="sans-serif" font-size="10" fill="#6b7280">{tick * y_max:.2g}</text>'
                )
            for tick in [0.0, 0.5, 1.0]:
                x = plot_x + tick * plot_w
                x_value = tick * x_max
                tick_label = f"{x_value:.1f}" if x_axis == "gpu_hours" else str(int(round(x_value)))
                lines.append(f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_y + plot_h}" stroke="#f9fafb"/>')
                lines.append(
                    f'<text x="{x - 11:.1f}" y="{plot_y + plot_h + 16}" font-family="sans-serif" font-size="10" fill="#6b7280">{tick_label}</text>'
                )

            cell_runs = [
                *full_context,
                *[r for r in non_full if r.turns == turn and r.stride == stride],
            ]
            cell_runs = [r for r in cell_runs if r.run_name in series_by_run]
            cell_runs.sort(key=lambda r: (0 if r.family == "full_context" else 1, r.family, r.run_name))
            label_x = plot_x + plot_w + 18
            label_y = plot_y + 12
            if not cell_runs:
                lines.append(
                    f'<text x="{plot_x + 35}" y="{plot_y + 82}" font-family="sans-serif" font-size="12" fill="#9ca3af">no data</text>'
                )
                continue

            for i, run in enumerate(cell_runs):
                points = series_by_run[run.run_name]
                color = FAMILY_COLORS.get(run.family, "#6b7280")
                opacity, dash = line_style(run)
                if len(points) == 1:
                    x = sx(cell_x, points[0][0])
                    y = sy(cell_y, points[0][1])
                    lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" opacity="{opacity}"/>')
                else:
                    point_text = " ".join(f"{sx(cell_x, x_value):.1f},{sy(cell_y, value):.1f}" for x_value, value in points)
                    lines.append(
                        f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2" opacity="{opacity}"{dash}/>'
                    )
                latest_x, latest_value = points[-1]
                latest_label = f"gpu h {latest_x:.1f}" if x_axis == "gpu_hours" else f"step {int(round(latest_x))}"
                ly = label_y + i * 31
                lines.append(
                    f'<line x1="{label_x}" y1="{ly - 4}" x2="{label_x + 16}" y2="{ly - 4}" stroke="{color}" stroke-width="2" opacity="{opacity}"{dash}/>'
                )
                lines.append(
                    f'<text x="{label_x + 22}" y="{ly}" font-family="sans-serif" font-size="10.5" fill="#111827">{svg_escape(family_label(run.family))} {latest_value:.3f}</text>'
                )
                lines.append(
                    f'<text x="{label_x + 22}" y="{ly + 11}" font-family="sans-serif" font-size="9" fill="#6b7280">{svg_escape(latest_label)}</text>'
                )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(rows: list[ResolvedRun], wandb_entity: str = "") -> str:
    header = [
        "family",
        "run",
        "t",
        "s",
        "status",
        "use",
        "relaunch",
        "step",
        "success",
        "hard",
        "train",
        "failure",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for run in rows:
        run_cell = f"`{run.run_name}`"
        if wandb_entity and run.wandb_project and run.wandb_id:
            url = f"https://wandb.ai/{wandb_entity}/{run.wandb_project}/runs/{run.wandb_id}"
            run_cell = f"[`{run.run_name}`]({url})"
        lines.append(
            "| "
            + " | ".join(
                [
                    run.family,
                    run_cell,
                    fmt(run.turns),
                    fmt(run.stride),
                    run.status,
                    run.result_use,
                    run.relaunch,
                    fmt(run.latest_step),
                    fmt(run.success_rate),
                    fmt(run.hard_success_rate),
                    fmt(run.train_reward),
                    run.failure_class,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_markdown(
    path: Path,
    runs: list[ResolvedRun],
    metric_rows: list[dict[str, Any]],
    plots_dir: Path,
    use_wandb: bool = False,
    wandb_entity: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    runs_sorted = sorted(runs, key=lambda r: (r.family, r.turns or 0, r.stride or 0, r.run_name))
    status_counts = Counter(r.status for r in runs_sorted)
    failure_counts = Counter(r.failure_class for r in runs_sorted)
    result_counts = Counter(r.result_use for r in runs_sorted)
    source_counts = Counter(str(row.get("source", "unknown")) for row in metric_rows)
    relaunch = [r for r in runs_sorted if r.relaunch == "yes"]
    primary = [r for r in runs_sorted if r.result_use in {"primary", "candidate"}]
    partial = [r for r in runs_sorted if r.result_use == "partial"]
    excluded = [r for r in runs_sorted if r.result_use in {"exclude", "debug"}]
    train_gpu_grid = plots_dir / "train_reward_gpu_hours_grid.svg"
    success_grid = plots_dir / "success_rate_grid.svg"
    hard_grid = plots_dir / "hard_success_rate_grid.svg"
    train_grid = plots_dir / "train_reward_grid.svg"
    success_bar = plots_dir / "success_rate_latest.svg"
    hard_bar = plots_dir / "hard_success_rate_latest.svg"
    train_bar = plots_dir / "train_reward_latest.svg"
    write_facet_svg(
        train_gpu_grid,
        runs_sorted,
        metric_rows,
        "train_reward",
        "Training Reward Running Average By GPU Hours",
        smooth_window=10,
        x_axis="gpu_hours",
        gpus=DEFAULT_GPU_COUNT,
    )
    write_facet_svg(
        train_grid,
        runs_sorted,
        metric_rows,
        "train_reward",
        "Training Reward Running Average By Max Turns And Stride",
        smooth_window=10,
    )
    write_facet_svg(success_grid, runs_sorted, metric_rows, "success_rate", "Eval Success Rate By Max Turns And Stride")
    write_facet_svg(hard_grid, runs_sorted, metric_rows, "hard_success_rate", "Eval Hard Success Rate By Max Turns And Stride")
    write_bar_svg(success_bar, runs_sorted, "success_rate", "Latest Success Rate")
    write_bar_svg(hard_bar, runs_sorted, "hard_success_rate", "Latest Hard Success Rate")
    write_bar_svg(train_bar, runs_sorted, "train_reward", "Latest Training Reward")

    def counts_text(counter: Counter[str]) -> str:
        return ", ".join(f"{key}={value}" for key, value in sorted(counter.items())) or "-"

    rel_train_gpu_grid = os.path.relpath(train_gpu_grid, path.parent)
    rel_success_grid = os.path.relpath(success_grid, path.parent)
    rel_hard_grid = os.path.relpath(hard_grid, path.parent)
    rel_train_grid = os.path.relpath(train_grid, path.parent)
    rel_success_bar = os.path.relpath(success_bar, path.parent)
    rel_hard_bar = os.path.relpath(hard_bar, path.parent)
    rel_train_bar = os.path.relpath(train_bar, path.parent)
    source_text = "local scratch logs, local W&B summaries, and local `.wandb` histories"
    if use_wandb and source_counts.get("wandb", 0):
        source_text += f", plus W&B API history under `{wandb_entity}`"
    elif use_wandb:
        source_text += f". W&B API access under `{wandb_entity}` was attempted but returned no remote history rows"
    warnings = []
    if source_counts.get("wandb_api_error", 0):
        warnings.append(
            f"- W&B API warnings: {source_counts['wandb_api_error']} remote lookups failed with the current credentials; local `.wandb` histories are still included."
        )
    lines = [
        "# TextWorld Experiment Report",
        "",
        f"Generated from `experiments/_reports/tracker.yaml`, {source_text}.",
        "",
        "## Summary",
        "",
        f"- Runs: {len(runs_sorted)}",
        f"- Metric rows: {len(metric_rows)}",
        f"- Metric sources: {counts_text(source_counts)}",
        f"- Status: {counts_text(status_counts)}",
        f"- Result use: {counts_text(result_counts)}",
        f"- Failure classes: {counts_text(failure_counts)}",
        *warnings,
        "",
        "## Plots",
        "",
        "Faceted plots are grouped by `max_turns` rows and `stride` columns. Full context is repeated in every panel.",
        f"`GPU-hours = cumulative(time/step seconds) * {DEFAULT_GPU_COUNT} / 3600`.",
        "",
        f"![Training reward by GPU hours]({rel_train_gpu_grid})",
        "",
        f"![Training reward by step]({rel_train_grid})",
        "",
        f"![Eval success rate grid]({rel_success_grid})",
        "",
        f"![Eval hard success rate grid]({rel_hard_grid})",
        "",
        f"![Latest training reward]({rel_train_bar})",
        "",
        f"![Latest success rate]({rel_success_bar})",
        "",
        f"![Latest hard success rate]({rel_hard_bar})",
        "",
        "## Candidate Results",
        "",
        markdown_table(primary, wandb_entity) if primary else "_No candidate runs found._",
        "",
        "## Partial Runs",
        "",
        markdown_table(partial, wandb_entity) if partial else "_No partial runs found._",
        "",
        "## Relaunch Queue",
        "",
        markdown_table(relaunch, wandb_entity) if relaunch else "_No runs currently marked for relaunch._",
        "",
        "## Excluded Or Debug Runs",
        "",
        markdown_table(excluded, wandb_entity) if excluded else "_No excluded/debug runs found._",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(
    tracker: Path,
    out: Path,
    plots_dir: Path,
    cache_dir: Path,
    use_wandb: bool = False,
    wandb_entity: str = "",
    refresh_wandb: bool = False,
) -> None:
    data = read_yaml(tracker)
    paper = data.get("paper", {})
    if not isinstance(paper, dict):
        paper = {}
    wandb_entity = wandb_entity or str(paper.get("wandb_entity") or DEFAULT_WANDB_ENTITY)
    runs: list[ResolvedRun] = []
    metric_rows: list[dict[str, Any]] = []
    for raw in tracker_runs(data):
        resolved, metrics = resolve_run(
            raw,
            use_wandb=use_wandb,
            wandb_entity=wandb_entity,
            cache_dir=cache_dir,
            refresh_wandb=refresh_wandb,
        )
        runs.append(resolved)
        metric_rows.extend(metrics)
    write_csv(cache_dir / "runs.csv", [row_dict(r) for r in runs])
    write_csv(cache_dir / "metrics.csv", metric_rows)
    write_markdown(out, runs, metric_rows, plots_dir, use_wandb=use_wandb, wandb_entity=wandb_entity)
    print(f"Wrote {out}")
    print(f"Wrote {cache_dir / 'runs.csv'}")
    print(f"Wrote {cache_dir / 'metrics.csv'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument("--init-tracker", action="store_true", help="Initialize tracker.yaml from local TOMLs")
    parser.add_argument("--force", action="store_true", help="Overwrite tracker when used with --init-tracker")
    parser.add_argument("--wandb", action="store_true", help="Pull W&B API history and cache it under --cache-dir")
    parser.add_argument("--wandb-entity", default="", help=f"W&B entity; defaults to tracker paper.wandb_entity or {DEFAULT_WANDB_ENTITY}")
    parser.add_argument("--refresh-wandb", action="store_true", help="Refresh cached W&B history")
    args = parser.parse_args(argv)

    if args.init_tracker:
        init_tracker(args.tracker, args.config_root, args.jobs_dir, args.force)
        return 0
    build_report(
        args.tracker,
        args.out,
        args.plots_dir,
        args.cache_dir,
        use_wandb=args.wandb,
        wandb_entity=args.wandb_entity,
        refresh_wandb=args.refresh_wandb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
