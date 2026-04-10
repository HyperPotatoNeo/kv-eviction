#!/usr/bin/env python3
"""Pre-prod probe: does prime-rl's checkpoint_wrapper corrupt DynamicCache
under use_cache=True + past_key_values?

The concern:
  - checkpoint_wrapper defaults to CheckpointImpl.NO_REENTRANT.
  - Non-reentrant checkpoint re-runs the wrapped function during backward
    using saved tensor hooks. Non-tensor inputs (DynamicCache is a Python
    object) are captured by closure, not snapshotted.
  - During a forward pass with use_cache=True, each block does
    `past_key_values.update(K_new, V_new, layer_idx)`. The first time a
    given layer_idx is seen, update() appends a new slot. Subsequent
    calls (e.g., during checkpoint re-run of that block in backward)
    take the "slot already exists" path and CONCATENATE K/V along the
    seq dim, doubling the slot's length.
  - If this fires, the re-run's attention sees a KV cache ~2x the
    original length, producing wrong output → wrong recomputed
    activations → wrong gradients. Crucially: no loud error, just
    quietly wrong policy gradients.

The probe:
  1. Baseline path: load Qwen3-4B, no checkpoint_wrapper anywhere.
     Forward one sequence with past_key_values=None + use_cache=True.
     Backward a plain sum loss, record per-param grads.
  2. AC path: clone the same model, apply checkpoint_wrapper to every
     decoder layer, same forward/backward, record per-param grads.
  3. Compare: flatten all grads into a single vector each, compute
     the relative L2 error and the elementwise max abs diff.

Interpretation:
  - Matching (relative error < ~1e-3): the concern is dismissed.
    Either HF's DynamicCache has protection against double-update
    (e.g., position-based deduplication), or non-reentrant checkpoint
    handles the stateful-input case via a mechanism I haven't spotted.
  - Diverging (relative error >> kernel noise): the concern is real.
    Smoke #3 needs `trainer.model.ac.freq = 0` for the first run, and
    we need to design a fix for the AC+use_cache interaction.

Uses a short input (seq_len ~1k) for speed; the concern doesn't depend
on sequence length. Runs on 1 GPU, ~1-2 minutes.
"""

from __future__ import annotations

import time

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SEQ_LEN = 1024


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run(model: torch.nn.Module, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward with use_cache=True, backward a sum loss. Return flat grads."""
    model.zero_grad(set_to_none=True)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    # use_cache=True + past_key_values=None => each block appends to a fresh
    # cache inside the top-level forward. That's exactly the problematic
    # pattern: each layer_idx starts empty, gets populated once during
    # forward, then (if AC is on) the checkpoint's backward re-run would
    # call update() again on an already-populated slot.
    out = model(
        input_ids=input_ids,
        position_ids=position_ids,
        use_cache=True,
    )
    logits = out.logits if hasattr(out, "logits") else out["logits"]
    loss = logits.float().sum()
    loss.backward()

    grads: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        grads[name] = p.grad.detach().clone().float()
    return {"loss": loss.detach().clone(), "grads": grads}


def flatten_grads(grads: dict[str, torch.Tensor], order: list[str]) -> torch.Tensor:
    return torch.cat([grads[n].reshape(-1) for n in order])


def main() -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    log(f"Loading {MODEL} (bf16, flash_attention_2)")
    t0 = time.time()
    model_a = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    model_a.train()
    log(f"Model A loaded in {time.time()-t0:.1f}s")

    # Use a deterministic input so any divergence between the two runs is
    # attributable to the AC-vs-noAC difference, not RNG.
    torch.manual_seed(42)
    input_ids = torch.randint(
        low=0, high=model_a.config.vocab_size, size=(1, SEQ_LEN), device=device
    )

    log("Run 1: BASELINE (no checkpoint_wrapper, use_cache=True)")
    result_a = run(model_a, input_ids)
    log(f"  loss = {result_a['loss'].item():.4f}")

    # Free model A params that we don't need (keep grads for comparison).
    log("Loading model B (same weights, with checkpoint_wrapper per block)")
    t0 = time.time()
    model_b = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    model_b.train()
    # Apply checkpoint_wrapper to every decoder layer, matching prime-rl's
    # pattern in trainer/model.py:648. Default impl is NO_REENTRANT.
    language_model = model_b.model  # Qwen3Model
    for layer_name, block in list(language_model.layers.named_children()):
        wrapped = checkpoint_wrapper(block, preserve_rng_state=False)
        language_model.layers.register_module(layer_name, wrapped)
    log(f"Model B wrapped in {time.time()-t0:.1f}s")

    log("Run 2: AC (checkpoint_wrapper per block, use_cache=True)")
    try:
        result_b = run(model_b, input_ids)
        log(f"  loss = {result_b['loss'].item():.4f}")
    except Exception as e:
        log(f"  RUN FAILED: {type(e).__name__}: {e}")
        log("CONCLUSION: AC + use_cache=True LOUDLY FAILS (not silently).")
        log("This means prime-rl AC + segmented_forward cannot co-exist unless")
        log("we either disable AC for compaction runs or patch the re-run path.")
        raise

    # Compare grads
    # Both models should have identical named_parameters because
    # checkpoint_wrapper preserves the underlying block's parameters (it
    # just wraps the forward). Verify the names match.
    common = [n for n in result_a["grads"] if n in result_b["grads"]]
    only_a = sorted(set(result_a["grads"]) - set(result_b["grads"]))
    only_b = sorted(set(result_b["grads"]) - set(result_a["grads"]))
    log(f"Params with grads: baseline={len(result_a['grads'])}, AC={len(result_b['grads'])}, common={len(common)}")
    if only_a:
        log(f"  only in baseline (first 5): {only_a[:5]}")
    if only_b:
        log(f"  only in AC (first 5): {only_b[:5]}")

    # Some param names may have a "_checkpoint_wrapped_module." prefix in
    # the AC model — checkpoint_wrapper wraps the block as a submodule.
    # Build a mapping by stripping that prefix for the AC model.
    AC_PREFIX = "_checkpoint_wrapped_module."
    def strip_ac(name: str) -> str:
        return name.replace(AC_PREFIX, "")

    baseline_keys = {n: n for n in result_a["grads"]}
    ac_keys = {strip_ac(n): n for n in result_b["grads"]}
    matched = []
    for stripped, baseline_name in baseline_keys.items():
        if stripped in ac_keys:
            matched.append((baseline_name, ac_keys[stripped]))

    log(f"Matched {len(matched)} params after stripping '{AC_PREFIX}' from AC side")
    if len(matched) != len(result_a["grads"]):
        unmatched = sorted(set(baseline_keys) - {b for b, _ in matched})
        log(f"  UNMATCHED baseline params ({len(unmatched)}): {unmatched[:5]}")

    # Compute diffs
    rel_errors = []
    max_abs_diffs = []
    worst_params = []
    for baseline_name, ac_name in matched:
        g_a = result_a["grads"][baseline_name]
        g_b = result_b["grads"][ac_name]
        if g_a.shape != g_b.shape:
            log(f"  SHAPE MISMATCH on {baseline_name}: {g_a.shape} vs {g_b.shape}")
            continue
        diff = (g_a - g_b).abs()
        max_abs = float(diff.max().item())
        denom = float(g_a.abs().max().item()) + 1e-12
        rel = max_abs / denom
        rel_errors.append(rel)
        max_abs_diffs.append(max_abs)
        worst_params.append((rel, baseline_name))

    worst_params.sort(reverse=True)

    log("=" * 60)
    log("GRADIENT COMPARISON: baseline (no AC) vs AC (checkpoint_wrapper)")
    log("=" * 60)
    log(f"  Matched params:                    {len(matched)}")
    log(f"  Loss identical:                    "
        f"{abs(result_a['loss'].item() - result_b['loss'].item()) < 1e-4}")
    if rel_errors:
        import statistics
        log(f"  Relative error max:                {max(rel_errors):.3e}")
        log(f"  Relative error mean:               {statistics.mean(rel_errors):.3e}")
        log(f"  Relative error median:             {statistics.median(rel_errors):.3e}")
        log(f"  Max elementwise abs diff:          {max(max_abs_diffs):.3e}")
        log("  Top 5 params by relative error:")
        for rel, name in worst_params[:5]:
            log(f"    {rel:.3e}  {name}")

    log("=" * 60)
    if rel_errors and max(rel_errors) < 1e-2:
        log("VERDICT: gradients MATCH within noise. Cache-mutation concern DISMISSED.")
        log("prime-rl's checkpoint_wrapper + use_cache=True appears safe for")
        log("segmented_forward production use.")
    elif rel_errors:
        log(f"VERDICT: gradients DIVERGE (worst rel error {max(rel_errors):.3e}).")
        log("This suggests the DynamicCache mutation concern is real. Smoke #3")
        log("should run with trainer.model.ac.freq=0 for the first pass, and")
        log("we need a fix for the AC+use_cache interaction.")
    log("=" * 60)


if __name__ == "__main__":
    main()
