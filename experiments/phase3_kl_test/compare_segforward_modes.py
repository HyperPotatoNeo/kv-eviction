#!/usr/bin/env python3
"""Compare legacy vs per-segment-backward modes of segmented_forward.

Hypothesis: phase3_kl_test validated the LEGACY path (1.23x ratio,
noise). Smoke #4 uses the per-segment-backward path via the trainer
dispatch and sees ~50x higher Mismatch KL vs the full-context baseline.

This script runs BOTH modes on the same input and captures per-segment
owned logits position-by-position, then compares them element-wise.
Truncated to the first N compaction events to fit in a 40 GB A100.

Usage (inside podman container with kv-eviction venv activated):
    python compare_segforward_modes.py --num-events 3 --sample-idx 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, "/pscratch/sd/s/siddart2/kv-eviction/src")
from kv_eviction.segmented_forward import segmented_forward  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BLOCK_SIZE = 16
COMPACTION_STRIDE = 512
ROLLOUT_PATH = Path(
    "/pscratch/sd/s/siddart2/kv-eviction/experiments/phase3_kl_test/results/rollouts_compaction.json"
)


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def truncate_sample(sample: dict, num_events: int) -> dict:
    """Keep only the first num_events compaction events and truncate the
    completion to just past the last retained boundary plus a small tail."""
    events = sample["compaction_events"][:num_events]
    if events:
        last_boundary = events[-1]["num_output_tokens_at_compaction"]
        # Tail of one stride/4 tokens so the last segment is non-trivial
        tail = COMPACTION_STRIDE // 4
        new_comp_len = min(last_boundary + tail, len(sample["completion_ids"]))
    else:
        new_comp_len = min(2 * COMPACTION_STRIDE, len(sample["completion_ids"]))
    return {
        "prompt_ids": sample["prompt_ids"],
        "completion_ids": sample["completion_ids"][:new_comp_len],
        "inference_logprobs": sample["inference_logprobs"][:new_comp_len],
        "compaction_events": events,
    }


def run_and_capture_legacy(
    model,
    input_ids,
    position_ids,
    segment_boundaries,
    prompt_len,
    prompt_aligned_len,
    temperature,
) -> dict[int, torch.Tensor]:
    """Legacy mode (loss_fn=None). Rather than rely on the final full
    torch.cat (which OOMs for 16k samples on 40 GB), we monkey-patch
    torch.Tensor.__iadd__ on a holder object — no, simpler: we call
    segmented_forward with loss_fn=None and then rip the per-segment
    pieces out of all_logits_pieces before the final cat.

    Actually, the cleanest way: segmented_forward ONLY cats when
    loss_fn is None. So we let it cat and just move the result to CPU
    immediately, slice out the positions we need, then drop the full
    tensor. The final cat is the memory bottleneck; torch.no_grad
    avoids activation retention but the logits tensor itself is still
    ~5 GB at 16k. Truncated samples (num_events=3) fit comfortably.

    Returns: dict mapping global_logit_pos -> logits_at_that_pos [vocab]
    """
    with torch.no_grad():
        out = segmented_forward(
            model=model,
            input_ids=input_ids,
            position_ids=position_ids,
            segment_boundaries=segment_boundaries,
            prompt_len=prompt_len,
            prompt_aligned_len=prompt_aligned_len,
            stride=COMPACTION_STRIDE,
            temperature=temperature,
            max_forward_passes=None,
            activation_checkpointing=False,
            loss_fn=None,
        )
        logits = out["logits"]  # [1, seq_len, vocab]
        # Move to CPU and drop GPU copy ASAP
        captured = {}
        for pos in range(logits.shape[1]):
            captured[pos] = logits[0, pos, :].detach().to("cpu").float()
        del logits, out
        torch.cuda.empty_cache()
    return captured


def run_and_capture_psb(
    model,
    input_ids,
    position_ids,
    segment_boundaries,
    prompt_len,
    prompt_aligned_len,
    temperature,
    dummy_scalar,
) -> dict[int, torch.Tensor]:
    """Per-segment backward mode. The loss_fn closure captures
    per-segment owned logits directly, no final cat needed.

    Note: per-segment backward mode requires grad to be ENABLED for
    the forward (so window_loss.backward() can run). We disable it
    via inference_mode? No, inference_mode breaks backward. Instead
    we let the forward build a graph, call the zero-loss .backward()
    inside segmented_forward, and never accumulate any real gradient.
    """
    captured: dict[int, torch.Tensor] = {}

    def _capture_loss_fn(seg_logits, full_logit_start, full_logit_end):
        # seg_logits: [1, N, vocab] — the segment's owned, temperature-
        # scaled, boundary-trimmed logits. Each position i in
        # seg_logits[0] corresponds to global logit position
        # (full_logit_start + i). Store a detached CPU copy.
        n_owned = seg_logits.shape[1]
        for i in range(n_owned):
            global_pos = full_logit_start + i
            captured[global_pos] = seg_logits[0, i, :].detach().to("cpu").float()
        # Return a zero scalar routed through the dummy_scalar leaf so
        # segmented_forward's window_loss.backward() has a path to
        # something that requires_grad. All model params have
        # requires_grad=False, so backward only writes grad to
        # dummy_scalar.grad (size 1 scalar), consuming near-zero memory.
        return dummy_scalar + seg_logits.sum().detach() * 0.0

    # Clear any existing param grads; we're about to run backward
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    # Also clear grad from any dummy scalar we created previously
    # (backward accumulates into .grad; on a 40 GB GPU we can't afford
    # to leave them allocated between segments).

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=segment_boundaries,
        prompt_len=prompt_len,
        prompt_aligned_len=prompt_aligned_len,
        stride=COMPACTION_STRIDE,
        temperature=temperature,
        max_forward_passes=None,
        activation_checkpointing=False,
        loss_fn=_capture_loss_fn,
        bptt_segments=1,  # M3 semantics, matches smoke #4
    )
    torch.cuda.empty_cache()
    return captured


def compute_segment_owned_ranges(
    segment_boundaries: list[int],
    prompt_len: int,
    seq_len: int,
) -> list[tuple[int, int]]:
    """Mirror segmented_forward's owned-range construction.
    Returns list of (owned_start, owned_end) for each segment — the
    global logit positions each segment owns. Matches the per-segment
    metric aggregation boundary that prime-rl's trainer uses for
    compaction samples.
    """
    # Reconstruct seg_ranges the same way segmented_forward does
    seg_ranges: list[tuple[int, int]] = []
    prev_boundary = 0
    for i, boundary in enumerate(segment_boundaries):
        if i == 0:
            seg_start = 0
        else:
            seg_start = prompt_len + prev_boundary - 1
        seg_end = min(prompt_len + boundary, seq_len)
        if seg_start < seg_end:
            seg_ranges.append((seg_start, seg_end))
        prev_boundary = boundary
    last_covered = prompt_len + segment_boundaries[-1]
    if last_covered < seq_len:
        seg_ranges.append((last_covered - 1, seq_len))

    # Convert to owned ranges (non-final drops last logit, final keeps all)
    owned = []
    for i, (s, e) in enumerate(seg_ranges):
        is_last = i == len(seg_ranges) - 1
        if is_last:
            owned.append((s, e))
        else:
            owned.append((s, e - 1))
    return owned


def compare(
    legacy: dict[int, torch.Tensor],
    psb: dict[int, torch.Tensor],
    prompt_len: int,
    completion_ids: list[int],
    inference_logprobs: list[float],
    segment_boundaries: list[int] | None = None,
    seq_len: int | None = None,
) -> dict:
    """Compare legacy and per-segment-backward captured logits at
    positions both modes own. Also compute each mode's logprobs against
    inference."""
    # Positions both modes cover
    common = sorted(set(legacy.keys()) & set(psb.keys()))
    # Restrict to completion positions (logits that predict a completion
    # token, i.e., logit at position P predicts token at position P+1
    # and P+1 is a completion token iff P+1 >= prompt_len iff P >= prompt_len-1)
    common = [p for p in common if p >= prompt_len - 1]
    log(f"  common positions in completion range: {len(common)}")

    logit_diffs = []
    lsm_diffs = []
    sampled_tok_diffs = []
    legacy_vs_inf = []
    psb_vs_inf = []
    # Signed log_ratios for mismatch_kl computation. mismatch_kl matches
    # prime-rl's trainer metric: mismatch_kl = exp(log_ratio) - log_ratio - 1
    # where log_ratio = trainer_logprob - inference_logprob.
    legacy_log_ratios: list[float] = []
    # Also track by position so we can re-group by segment below.
    pos_to_log_ratio: dict[int, float] = {}

    for pos in common:
        legacy_logits = legacy[pos]  # [vocab]
        psb_logits = psb[pos]  # [vocab]

        # Raw logit diff
        logit_diffs.append(float((legacy_logits - psb_logits).abs().max().item()))

        # log_softmax diff (what actually matters for policy logprobs)
        legacy_lsm = torch.log_softmax(legacy_logits, dim=-1)
        psb_lsm = torch.log_softmax(psb_logits, dim=-1)
        lsm_diffs.append(float((legacy_lsm - psb_lsm).abs().max().item()))

        # For the sampled token at position (pos + 1), compare logprobs
        comp_idx = (pos + 1) - prompt_len
        if 0 <= comp_idx < len(completion_ids):
            tok = completion_ids[comp_idx]
            legacy_lp = float(legacy_lsm[tok].item())
            psb_lp = float(psb_lsm[tok].item())
            inf_lp = float(inference_logprobs[comp_idx])
            sampled_tok_diffs.append(abs(legacy_lp - psb_lp))
            legacy_vs_inf.append(abs(legacy_lp - inf_lp))
            psb_vs_inf.append(abs(psb_lp - inf_lp))
            # Signed log_ratio for mismatch_kl
            legacy_log_ratios.append(legacy_lp - inf_lp)
            pos_to_log_ratio[pos] = legacy_lp - inf_lp

    def stats(xs):
        if not xs:
            return {"n": 0, "mean": None, "max": None, "median": None}
        import statistics
        return {
            "n": len(xs),
            "mean": sum(xs) / len(xs),
            "max": max(xs),
            "median": statistics.median(xs),
        }

    # Mismatch KL: exp(log_ratio) - log_ratio - 1. Matches prime-rl's
    # trainer/rl/loss.py:139 definition. This is what smoke #4 reports
    # as "Mismatch KL" in its per-step trainer summary.
    import math
    mismatch_kl_values = [math.exp(r) - r - 1 for r in legacy_log_ratios]

    # Per-segment aggregation: simulate what the trainer's segmented
    # dispatch does. For each segment:
    #   1. compute_loss is called with the segment's token range only
    #   2. _safe_mean(mismatch_kl, loss_mask) returns a scalar = token-
    #      weighted mean WITHIN the segment
    #   3. the per-segment scalar is accumulated into a shape-[n_segments]
    #      tensor
    #   4. downstream compute_stats does torch.cat + .mean() — an
    #      UNWEIGHTED mean across segments
    # If smoke #4's reported 0.0395 matches this unweighted aggregation
    # while the token-weighted mean is much smaller, the 72x gap is a
    # metric-aggregation artifact not a correctness bug.
    per_segment_mismatch_kl_means: list[float] = []
    if segment_boundaries is not None and seq_len is not None:
        owned = compute_segment_owned_ranges(segment_boundaries, prompt_len, seq_len)
        log(f"  reconstructed {len(owned)} segment owned ranges")
        for (o_start, o_end) in owned:
            # Collect log_ratios for positions in this segment's owned
            # range AND in the loss region (target token >= prompt_len).
            seg_log_ratios = []
            for p in range(o_start, o_end):
                comp_idx = (p + 1) - prompt_len
                if comp_idx < 0 or comp_idx >= len(completion_ids):
                    continue
                if p not in pos_to_log_ratio:
                    continue
                seg_log_ratios.append(pos_to_log_ratio[p])
            if seg_log_ratios:
                seg_mismatch_kl = [math.exp(r) - r - 1 for r in seg_log_ratios]
                per_segment_mismatch_kl_means.append(
                    sum(seg_mismatch_kl) / len(seg_mismatch_kl)
                )
            else:
                per_segment_mismatch_kl_means.append(0.0)

    def log_ratio_percentiles(values):
        if not values:
            return {}
        sorted_abs = sorted(abs(v) for v in values)
        n = len(sorted_abs)
        return {
            "p50": sorted_abs[n // 2],
            "p90": sorted_abs[int(n * 0.9)],
            "p99": sorted_abs[int(n * 0.99)],
            "p999": sorted_abs[min(n - 1, int(n * 0.999))],
            "max": sorted_abs[-1],
        }

    return {
        "n_common": len(common),
        "raw_logit_diff": stats(logit_diffs),
        "logsoftmax_diff_max_over_vocab": stats(lsm_diffs),
        "sampled_token_logprob_diff": stats(sampled_tok_diffs),
        "legacy_vs_inference": stats(legacy_vs_inf),
        "psb_vs_inference": stats(psb_vs_inf),
        "mismatch_kl_mean_token_weighted": sum(mismatch_kl_values) / len(mismatch_kl_values) if mismatch_kl_values else None,
        "mismatch_kl_max": max(mismatch_kl_values) if mismatch_kl_values else None,
        "mismatch_kl_mean_per_segment_unweighted": (
            sum(per_segment_mismatch_kl_means) / len(per_segment_mismatch_kl_means)
            if per_segment_mismatch_kl_means else None
        ),
        "per_segment_mismatch_kl_means": per_segment_mismatch_kl_means,
        "log_ratio_abs_percentiles": log_ratio_percentiles(legacy_log_ratios),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Sample index (-1 to run all samples)")
    parser.add_argument("--num-events", type=int, default=3,
                        help="Truncate to first N compaction events")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    log(f"Loading {MODEL} on {device} (bf16 flash_attention_2)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    # eval() avoids dropout/bn surprises; we never call optimizer.step
    # so training mode wasn't needed.
    model.eval()
    # Freeze ALL model params so per-segment backward doesn't accumulate
    # .grad tensors (which on a 4B model = 8 GB per copy). We'll add a
    # single dummy scalar that DOES require grad and route the zero-loss
    # through it so segmented_forward's window_loss.backward() succeeds.
    for p in model.parameters():
        p.requires_grad_(False)
    dummy_scalar = torch.zeros((), device=device, requires_grad=True)
    log(f"Model loaded in {time.time()-t0:.0f}s")
    log(f"GPU total mem: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    rollouts = json.loads(ROLLOUT_PATH.read_text())
    sample_idxs = (
        list(range(len(rollouts["samples"])))
        if args.sample_idx < 0
        else [args.sample_idx]
    )

    all_results = []
    for idx in sample_idxs:
        log(f"========== Sample {idx} ==========")
        sample = truncate_sample(rollouts["samples"][idx], args.num_events)
        log(
            f"Sample {idx} truncated: prompt={len(sample['prompt_ids'])}, "
            f"completion={len(sample['completion_ids'])}, events={len(sample['compaction_events'])}"
        )

        prompt_len = len(sample["prompt_ids"])
        seq_len = prompt_len + len(sample["completion_ids"])
        input_ids = torch.tensor(
            [sample["prompt_ids"] + sample["completion_ids"]], dtype=torch.long, device=device,
        )
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        temperature = torch.ones(1, seq_len, device=device, dtype=torch.bfloat16)
        prompt_aligned_len = ((prompt_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        segment_boundaries = [
            int(e["num_output_tokens_at_compaction"]) for e in sample["compaction_events"]
        ]

        log("=== Running LEGACY mode ===")
        t0 = time.time()
        legacy = run_and_capture_legacy(
            model, input_ids, position_ids, segment_boundaries,
            prompt_len, prompt_aligned_len, temperature,
        )
        log(f"  legacy done in {time.time()-t0:.1f}s, captured {len(legacy)} positions")
        torch.cuda.reset_peak_memory_stats()

        log("=== Running PER-SEGMENT BACKWARD mode ===")
        t0 = time.time()
        psb = run_and_capture_psb(
            model, input_ids, position_ids, segment_boundaries,
            prompt_len, prompt_aligned_len, temperature, dummy_scalar,
        )
        log(f"  psb done in {time.time()-t0:.1f}s, captured {len(psb)} positions")

        log("=== Comparing ===")
        result = compare(
            legacy, psb, prompt_len, sample["completion_ids"],
            sample["inference_logprobs"],
            segment_boundaries=segment_boundaries,
            seq_len=seq_len,
        )
        result["sample_idx"] = idx
        all_results.append(result)
        # Free GPU memory between samples
        del legacy, psb, input_ids, position_ids, temperature
        torch.cuda.empty_cache()

    # Keep backward-compat: if single sample, present it as the "current" result
    result = all_results[0] if len(all_results) == 1 else {
        "n_samples": len(all_results),
        "per_sample": all_results,
        # Aggregate across samples
        "aggregate_mismatch_kl_mean_token_weighted": sum(
            r["mismatch_kl_mean_token_weighted"] or 0.0 for r in all_results
        ) / len(all_results),
        "aggregate_mismatch_kl_mean_per_segment_unweighted": sum(
            r["mismatch_kl_mean_per_segment_unweighted"] or 0.0 for r in all_results
        ) / len(all_results),
        "aggregate_mismatch_kl_max": max(r["mismatch_kl_max"] or 0.0 for r in all_results),
    }

    if len(all_results) == 1:
        r = all_results[0]
        log(f"n_common_positions: {r['n_common']}")
        log(f"raw_logit_diff:              mean={r['raw_logit_diff']['mean']:.6f}  max={r['raw_logit_diff']['max']:.6f}")
        log(f"logsoftmax_diff (max over vocab per pos): mean={r['logsoftmax_diff_max_over_vocab']['mean']:.6f}  max={r['logsoftmax_diff_max_over_vocab']['max']:.6f}")
        log(f"sampled_token_logprob_diff:  mean={r['sampled_token_logprob_diff']['mean']:.6f}  max={r['sampled_token_logprob_diff']['max']:.6f}")
        log(f"legacy vs inference:         mean={r['legacy_vs_inference']['mean']:.6f}  max={r['legacy_vs_inference']['max']:.6f}")
        log(f"psb    vs inference:         mean={r['psb_vs_inference']['mean']:.6f}  max={r['psb_vs_inference']['max']:.6f}")
        log(f"mismatch_kl mean (token-weighted):          {r['mismatch_kl_mean_token_weighted']:.6f}")
        log(f"mismatch_kl mean (per-segment, UNWEIGHTED): {r['mismatch_kl_mean_per_segment_unweighted']:.6f}")
        log(f"mismatch_kl max (single position):          {r['mismatch_kl_max']:.6f}")
    else:
        log("=== Per-sample summary ===")
        log(f"{'idx':>4}  {'tok_weighted':>14}  {'per_seg_unweighted':>18}  {'max_pos':>10}  {'lr_p99':>10}  {'lr_max':>10}")
        for r in all_results:
            log(
                f"{r['sample_idx']:>4}  {r['mismatch_kl_mean_token_weighted']:>14.6f}  "
                f"{r['mismatch_kl_mean_per_segment_unweighted']:>18.6f}  "
                f"{r['mismatch_kl_max']:>10.6f}  "
                f"{r['log_ratio_abs_percentiles']['p99']:>10.4f}  "
                f"{r['log_ratio_abs_percentiles']['max']:>10.4f}"
            )
        log("")
        log("=== Aggregate across samples ===")
        log(f"  aggregate mismatch_kl mean (token-weighted):          {result['aggregate_mismatch_kl_mean_token_weighted']:.6f}")
        log(f"  aggregate mismatch_kl mean (per-segment, UNWEIGHTED): {result['aggregate_mismatch_kl_mean_per_segment_unweighted']:.6f}")
        log(f"  aggregate mismatch_kl max:                            {result['aggregate_mismatch_kl_max']:.6f}")
    log(f"|log_ratio| percentiles: p50={result['log_ratio_abs_percentiles'].get('p50', 0):.4f} "
        f"p90={result['log_ratio_abs_percentiles'].get('p90', 0):.4f} "
        f"p99={result['log_ratio_abs_percentiles'].get('p99', 0):.4f} "
        f"p999={result['log_ratio_abs_percentiles'].get('p999', 0):.4f} "
        f"max={result['log_ratio_abs_percentiles'].get('max', 0):.4f}")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        log(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
