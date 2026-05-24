"""Debugpy-friendly standalone version of a TextWorld compaction run.

Runs a short TextWorld cooking episode against a turn-mode-compaction vLLM
engine, with debugpy wired up so you can attach from VS Code / Cursor and
set breakpoints in vllm/v1/core/sched/scheduler.py.

Mirrors experiments/debug_balrog/compaction_debug.py, but swaps BALROG's
BabyAI env for the TextWorld cooking env defined in
experiments/textworld_env/textworld_env.py. The compaction knobs below
mirror experiments/compaction_textworld/inference.toml (turn-based:
max_turns=10, stride=3, window=4096, stride=512, protected_prefix=-1).

Usage:
    # Default: waits for a debugger to attach before doing anything.
    python experiments/textworld_env/compaction_debug.py

    # The process will print "Waiting for debugger on 127.0.0.1:5678..."
    # and BLOCK until you attach. In VS Code / Cursor, use "Python: Attach"
    # pointing at localhost:5678. Then set breakpoints in:
    #   vllm/vllm/v1/core/sched/scheduler.py
    #     - _scan_new_turn_boundaries   (turn tracking)
    #     - _turn_mode_effective_prompt
    #     - _should_compact             (trigger check)
    #     - _compact_request            (eviction range + event emission)

    # Env var cheatsheet:
    #   VLLM_DEBUGPY_OFF=1          skip debugpy entirely (script runs headless)
    #   VLLM_DEBUGPY_PORT=5678      override port (default 5678)
    #   VLLM_DEBUGPY_NOWAIT=1       listen but don't block; attach whenever
    #   KVE_ASSERT_NO_PHASE4_REFILL=1
    #                               fail inside vLLM if a Phase4 continuation
    #                               would re-prefill retained KV tokens
    #   MAX_TURNS=20                how many game turns to run
    #   PAD=1                       pad messages so <|im_end|> lands on
    #                               block boundary (matches notebook default)
    #   DATASET=/path/to/textworld_cooking_mix    override dataset dir
    #   GAME_IDX=0                  which game in metadata.json to play

Critical: we force VLLM_ENABLE_V1_MULTIPROCESSING=0 below so the
EngineCore runs in the MAIN process. Otherwise the scheduler lives in a
spawned subprocess and debugpy listening in this process will never see
it. This slows engine init a bit but is the only way to make attach-
and-breakpoint work without a second listener in the subprocess.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Must be set BEFORE importing vllm. Keeps the scheduler in-process so
# debugpy in this process can actually hit breakpoints inside it.
# ---------------------------------------------------------------------------
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_COMPACTION_DEBUG_TOKENS", "1")
# Phase 2/4 verification logs: [PREFIX-HIT] per request at admission,
# [EVICT-PRE-DECODE] + [EVICT-DONE] when admission compaction fires.
os.environ.setdefault("VLLM_COMPACTION_VERBOSE", "1")
os.environ.setdefault("KVE_TRACE_PHASE4_PREFIX_HIT", "1")


# def _maybe_start_debugpy() -> None:
#     if os.environ.get("VLLM_DEBUGPY_OFF", "0") == "1":
#         return
#     try:
#         import debugpy  # type: ignore
#     except ImportError:
#         print(
#             "debugpy not installed. Run: uv pip install debugpy "
#             "(or set VLLM_DEBUGPY_OFF=1 to skip)",
#             file=sys.stderr,
#         )
#         sys.exit(1)

#     port = int(os.environ.get("VLLM_DEBUGPY_PORT", "5678"))
#     debugpy.listen(("127.0.0.1", port))
#     if os.environ.get("VLLM_DEBUGPY_NOWAIT", "0") == "1":
#         print(f"[debugpy] listening on 127.0.0.1:{port} (not waiting)")
#         return
#     print(
#         f"[debugpy] waiting for debugger on 127.0.0.1:{port} ... "
#         "(attach from VS Code / Cursor 'Python: Attach')",
#         flush=True,
#     )
#     debugpy.wait_for_client()
#     print("[debugpy] client attached, continuing")


# _maybe_start_debugpy()


# ---------------------------------------------------------------------------
# Compaction config knobs (mirror experiments/compaction_textworld/inference.toml)
# Turn-based eviction: max_turns=10 / stride=3, with block-FIFO fallback
# at window=4096 / stride=512 / block_size=16.
# ---------------------------------------------------------------------------
PAD = os.environ.get("PAD", "1") == "1"
AUTO_PAD_FINISH = os.environ.get("AUTO_PAD_FINISH", "1") == "1"
# Phase 2 prefix-caching smoke: set PREFIX_CACHE=1 to verify the
# hash-chain rebuild on eviction (plans/prefix_caching_compaction.md).
# Default off to match the historical compaction config.
PREFIX_CACHE = os.environ.get("PREFIX_CACHE", "1") == "1"
# Phase 4: assemble the next-turn prompt as [vLLM_kv_state + new_user_fragment]
# instead of [full_chat_history]. Requires PREFIX_CACHE=1 + Phase 2 rehash;
# this is what actually realizes the prefix-cache hit on the kept window.
PHASE4 = True
BLOCK_SIZE = 16
PAD_FILLER_ID: int | None = None
MAX_TURNS = int(os.environ.get("MAX_TURNS", "30"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2000"))
N_SAMPLES = int(os.environ.get("N_SAMPLES", "1"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
BASE_SEED = int(os.environ.get("SEED", "0"))
COMPACTION_MAX_TURNS = int(os.environ.get("COMPACTION_MAX_TURNS", "3"))
DATASET_PATH = os.environ.get(
    "DATASET",
    "/scratch/epp/textworld_cooking_mix",
)
GAME_IDX = int(os.environ.get("GAME_IDX", "0"))
OUT_PATH = os.environ.get(
    "OUT_PATH",
    "/home/toolkit/emi_dir/kv-eviction/experiments/textworld_env/out_turn_debug.txt",
)
SUMMARY_OUT_PATH = os.environ.get(
    "SUMMARY_OUT_PATH",
    "/home/toolkit/emi_dir/kv-eviction/experiments/textworld_env/out_turn_debug_summary.txt",
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    max_model_len=16384,
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    enable_prefix_caching=PREFIX_CACHE,
    block_size=BLOCK_SIZE,
    # Block-FIFO fallback (fires if KV exceeds window before turn-mode
    # accumulates max_turns completed turns).
    compaction_window_size=4096,
    compaction_stride=512,
    # -1 = auto-detect system prompt end from first <|im_end|>.
    compaction_protected_prefix_tokens=-1,
    # Turn-mode: evict 1 oldest completed turn whenever max_turns live.
    compaction_max_turns=COMPACTION_MAX_TURNS,
    compaction_eviction_turn_stride=1,
    compaction_turn_end_token_id=None,
    # With PAD=True in the chat() helper, every <|im_end|> is padded so
    # that the next message's first token lands on a block boundary.
    # This flag tells the turn-mode planner to snap evict_end UP to that
    # boundary (instead of the default inward snap), eliminating the
    # orphan tail-of-last-evicted-turn that otherwise survives in KV.
    compaction_assume_aligned_turn_boundaries=PAD,
    # Match the production Phase 4 configs: after the assistant finishes,
    # keep the request alive for one filler-token prefill step when needed
    # so the next Phase 4 prompt can hit a full final block.
    compaction_block_aligned_finish=AUTO_PAD_FINISH,
    compaction_filler_token_id=151643,
    async_scheduling=False,
)
tokenizer = llm.get_tokenizer()
print(
    f"loaded (turn-mode, PAD={PAD}, PREFIX_CACHE={PREFIX_CACHE}, "
    f"PHASE4={PHASE4}, AUTO_PAD_FINISH={AUTO_PAD_FINISH}, "
    f"COMPACTION_MAX_TURNS={COMPACTION_MAX_TURNS})"
)


# ---------------------------------------------------------------------------
# Chat helper (no tools — TextWorld uses plain <action> XML in content)
# ---------------------------------------------------------------------------


def _filler_token_id() -> int:
    if PAD_FILLER_ID is not None:
        return int(PAD_FILLER_ID)
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    enc = tokenizer.encode(" ", add_special_tokens=False)
    if not enc:
        eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        assert isinstance(eot, int) and eot >= 0, "no usable filler token"
        return eot
    return int(enc[-1])


def _render_padded(messages, im_end_id, block_size, filler_id):
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    raw = tokenizer.encode(rendered, add_special_tokens=False)
    bad = [(i, type(t).__name__) for i, t in enumerate(raw) if not isinstance(t, int)]
    if bad:
        raise TypeError(f"non-int tokens in encode output: first 5={bad[:5]}")

    out, pads = [], []
    for tok in raw:
        out.append(tok)
        if tok == im_end_id:
            remainder = len(out) % block_size
            n = (block_size - remainder) % block_size
            if n:
                out.extend([filler_id] * n)
            pads.append(n)
    return raw, out, pads


def _sampling_params(
    seed: int,
    phase4_expected_cached_tokens: int = 0,
) -> SamplingParams:
    extra_args = None
    if phase4_expected_cached_tokens > 0:
        extra_args = {
            "kve_phase4_expected_cached_tokens": int(
                phase4_expected_cached_tokens
            )
        }
    return SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        seed=seed,
        logprobs=1,
        extra_args=extra_args,
    )


def _build_full_chat_prompt_ids(messages, show_pad_summary=True, label=""):
    if PAD:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assert isinstance(im_end_id, int) and im_end_id >= 0, (
            f"<|im_end|> not in tokenizer (got {im_end_id!r})"
        )
        filler_id = _filler_token_id()
        raw, padded, pads = _render_padded(
            messages, im_end_id, BLOCK_SIZE, filler_id
        )
        if show_pad_summary:
            print(
                f"  {label}[pad] raw={len(raw)} -> padded={len(padded)} "
                f"(+{len(padded)-len(raw)}); per-im_end pads={pads} "
                f"(filler_id={filler_id})"
            )
        return padded

    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    bad = [(i, type(t).__name__) for i, t in enumerate(ids) if not isinstance(t, int)]
    if bad:
        raise TypeError(f"non-int tokens in encode output: first 5={bad[:5]}")
    return ids


def chat(messages, max_tokens=512, temperature=1.0, seed=0, show_pad_summary=True):
    sp = SamplingParams(
        max_tokens=max_tokens, temperature=temperature, seed=seed, logprobs=1
    )

    if PAD:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assert isinstance(im_end_id, int) and im_end_id >= 0, (
            f"<|im_end|> not in tokenizer (got {im_end_id!r})"
        )
        filler_id = _filler_token_id()
        raw, padded, pads = _render_padded(
            messages, im_end_id, BLOCK_SIZE, filler_id
        )
        if show_pad_summary:
            print(
                f"  [pad] raw={len(raw)} -> padded={len(padded)} "
                f"(+{len(padded)-len(raw)}); per-im_end pads={pads} "
                f"(filler_id={filler_id})"
            )
        outs = llm.generate(
            prompts=[{"prompt_token_ids": padded}],
            sampling_params=sp,
            use_tqdm=False,
        )
    else:
        outs = llm.chat(messages=messages, sampling_params=sp, use_tqdm=False)

    out = outs[0]
    text = out.outputs[0].text
    return text, out


# ---------------------------------------------------------------------------
# Phase 4: incremental prompt assembly using vLLM's post-compaction kept view.
#
# Instead of re-rendering the full chat history each turn (which forces a
# full re-prefill because compaction has rotated old turns out of cache),
# we send the NEW user message appended to the canonical vLLM KV state:
#
#     submitted_prompt = [kept_token_ids_from_last_event]
#                      + [assistant_output_tokens_from_last_turn]
#                      + [<|im_start|>user\n{obs}<|im_end|>\n<|im_start|>assistant\n]
#                      (with block-aligning fillers after the new im_end)
#
# With Phase 2 hash rebuild in vLLM, the kept_token_ids portion hashes
# against rebuilt block entries — so we expect a near-100% prefix-cache
# hit on everything except the new user fragment.
# ---------------------------------------------------------------------------

# Qwen3 chat-template fragment for "new user message + asst generation prompt".
# Matches the suffix that tokenizer.apply_chat_template would produce when
# extending a chat by one user turn with add_generation_prompt=True.
_NEW_USER_FRAGMENT = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def _pad_after_im_end(tokens, start_offset, im_end_id, block_size, filler_id):
    """Append tokens one at a time onto `start_offset` and insert
    block-aligning fillers after each <|im_end|>. Mirrors `_render_padded`'s
    pad logic but starts from a running offset rather than 0.

    Returns (padded_tokens, total_pad_count).
    """
    out = []
    running = start_offset
    total_pad = 0
    for tok in tokens:
        out.append(tok)
        running += 1
        if tok == im_end_id:
            remainder = running % block_size
            n = (block_size - remainder) % block_size
            if n:
                out.extend([filler_id] * n)
                running += n
                total_pad += n
    return out, total_pad


def _build_phase4_prompt_ids(
    prev_state_tokens,
    new_user_content,
    show_summary=True,
    label="",
):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    filler_id = _filler_token_id()

    fragment_text = _NEW_USER_FRAGMENT.format(content=new_user_content)
    fragment_ids = tokenizer.encode(fragment_text, add_special_tokens=False)

    if PAD:
        padded_fragment, total_pad = _pad_after_im_end(
            fragment_ids, len(prev_state_tokens),
            im_end_id, BLOCK_SIZE, filler_id,
        )
    else:
        padded_fragment, total_pad = fragment_ids, 0

    submitted = list(prev_state_tokens) + padded_fragment
    if show_summary:
        print(
            f"  {label}[phase4] prev_state={len(prev_state_tokens)} "
            f"fragment_raw={len(fragment_ids)} "
            f"fragment_padded={len(padded_fragment)} "
            f"(+{total_pad}) total={len(submitted)}"
        )
    return submitted


def chat_phase4(
    prev_state_tokens,
    new_user_content,
    max_tokens=512,
    temperature=1.0,
    seed=0,
    show_summary=True,
):
    """Submit a turn as [prev_state_tokens + new_user_fragment].

    prev_state_tokens is whatever vLLM physically has in KV after the
    previous turn — i.e. last event's kept_token_ids extended by that
    turn's assistant output (or the full submitted padded prompt if no
    compaction fired that turn).
    """
    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        logprobs=1,
        extra_args={
            "kve_phase4_expected_cached_tokens": len(prev_state_tokens)
        },
    )
    submitted = _build_phase4_prompt_ids(
        prev_state_tokens,
        new_user_content,
        show_summary=show_summary,
    )

    outs = llm.generate(
        prompts=[{"prompt_token_ids": submitted}],
        sampling_params=sp,
        use_tqdm=False,
    )
    out = outs[0]
    return out.outputs[0].text, out, submitted


def derive_next_prev_state(out, submitted_prompt_ids):
    """Compute the post-turn vLLM KV state from a RequestOutput.

    If compaction fired during this turn, the last event's kept_token_ids
    captures what physically survived after the final eviction; everything
    sampled after that (the assistant output) is appended.

    If no compaction fired, vLLM still has the full submitted prompt plus
    the asst output in KV.

    With PAD=True, the asst's trailing <|im_end|> is followed by filler
    tokens to the next block boundary, so the next-turn user fragment's
    <|im_start|>user lands block-aligned. This mirrors
    `_update_phase4_state_from_response` in src/kv_eviction/env.py:
    vLLM's auto-pad emits filler only, with no chat-template separator.
    """
    events = getattr(out, "compaction_events", None) or []
    if events:
        kept = list(getattr(events[-1], "kept_token_ids", []))
        # Defensive: pre-Phase-1 servers won't emit kept_token_ids; in that
        # case fall back to the submitted prompt (no real Phase 4 hit, but
        # at least the script doesn't crash).
        if not kept:
            kept = list(submitted_prompt_ids)
    else:
        kept = list(submitted_prompt_ids)
    asst_tokens = list(out.outputs[0].token_ids)
    state = kept + asst_tokens
    if PAD:
        filler_id = _filler_token_id()
        remainder = len(state) % BLOCK_SIZE
        n = (BLOCK_SIZE - remainder) % BLOCK_SIZE
        if n:
            state.extend([filler_id] * n)
    return state


# ---------------------------------------------------------------------------
# TextWorld loop
# ---------------------------------------------------------------------------
# Reuse helpers from the textworld env module so we don't duplicate the
# parser-lock / game-file-resolution logic.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textworld_env import (  # noqa: E402
    SYSTEM_PROMPT,
    _resolve_game_files,
    _start_game,
)


_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)


def reset_env(dataset_path=DATASET_PATH, game_idx=GAME_IDX):
    ds_path = Path(dataset_path)
    with open(ds_path / "metadata.json") as f:
        meta = json.load(f)
    game_files = _resolve_game_files(ds_path, meta["game_files"])
    max_scores = meta["max_scores"]
    game_file = game_files[game_idx]
    max_score = max_scores[game_idx]
    env, game_state = _start_game(game_file)
    return env, game_state, max_score, game_file


def parse_action(text):
    m = _ACTION_RE.search(text)
    if not m:
        return None
    action = m.group(1).strip()
    return action or None


@dataclass
class RolloutState:
    sample_idx: int
    env: Any
    max_score: int
    game_file: Any
    conv: list[dict[str, str]]
    score: int = 0
    prev_state_tokens: list[int] | None = None
    done: bool = False
    cumul_events: int = 0
    cumul_tokens_evicted: int = 0
    bad_prefix_hits: int = 0
    timings: list[tuple[int, float, int, int, int]] = field(default_factory=list)


def _format_event(i, ev):
    lines = [
        f"  event #{i}: evicted={ev.tokens_evicted} "
        f"offset_after={ev.position_offset_after} "
        f"prompt_tokens={getattr(ev, 'num_prompt_tokens', '?')} "
        f"last_turn={getattr(ev, 'last_turn_evicted', '?')} "
        f"turns_evicted_after={getattr(ev, 'num_turns_evicted_after', '?')}"
    ]
    ids = getattr(ev, "evicted_token_ids", None)
    if ids:
        try:
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
        except Exception as e:
            decoded = f"<decode-failed: {e}>"
        lines.append(f"  evicted-text ({len(decoded)} chars):")
        lines.append("  " + "-" * 70)
        for ln in (decoded.splitlines() or [decoded]):
            lines.append(f"  | {ln}")
        lines.append("  " + "-" * 70)
    return "\n".join(lines)


def main_multi_sample():
    """Run several same-game rollouts through one batched vLLM engine.

    This stresses the failure mode where Phase4 continuations from
    near-identical TextWorld traces share prefix-cache entries. Each sample
    owns an independent TextWorld env, but all samples use the same GAME_IDX.
    """
    samples: list[RolloutState] = []
    for sample_idx in range(N_SAMPLES):
        env, game_state, max_score, game_file = reset_env(
            dataset_path=DATASET_PATH, game_idx=GAME_IDX
        )
        samples.append(
            RolloutState(
                sample_idx=sample_idx,
                env=env,
                max_score=max_score,
                game_file=game_file,
                conv=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": game_state.feedback},
                ],
            )
        )

    bad_samples: set[int] = set()
    total_bad_prefix_hits = 0

    with open(OUT_PATH, "w") as f, open(SUMMARY_OUT_PATH, "w") as fs:
        f.write(
            f"# TextWorld multi-sample compaction trace "
            f"(PAD={PAD} PREFIX_CACHE={PREFIX_CACHE} PHASE4={PHASE4})\n"
        )
        f.write(
            f"# dataset={DATASET_PATH} game_idx={GAME_IDX} "
            f"n_samples={N_SAMPLES} max_turns={MAX_TURNS} "
            f"max_tokens={MAX_TOKENS} temperature={TEMPERATURE} "
            f"seed={BASE_SEED}\n"
        )
        fs.write(
            "# sample turn events prompt output cached hit_pct "
            "phase4_expected_cached phase4_refill_delta score done batch_s\n"
        )
        f.flush()
        fs.flush()

        try:
            for turn in range(MAX_TURNS):
                active = [sample for sample in samples if not sample.done]
                if not active:
                    break

                prompts = []
                params = []
                submitted_by_sample: dict[int, list[int]] = {}
                expected_by_sample: dict[int, int] = {}

                print(
                    f"turn {turn}: batching {len(active)} same-game samples...",
                    flush=True,
                )
                for sample in active:
                    label = f"[s{sample.sample_idx}] "
                    if PHASE4 and sample.prev_state_tokens is not None:
                        expected = len(sample.prev_state_tokens)
                        submitted = _build_phase4_prompt_ids(
                            sample.prev_state_tokens,
                            sample.conv[-1]["content"],
                            show_summary=True,
                            label=label,
                        )
                    else:
                        expected = 0
                        submitted = _build_full_chat_prompt_ids(
                            sample.conv,
                            show_pad_summary=True,
                            label=label,
                        )
                    prompts.append({"prompt_token_ids": submitted})
                    params.append(
                        _sampling_params(
                            BASE_SEED + sample.sample_idx * 100000 + turn,
                            phase4_expected_cached_tokens=expected,
                        )
                    )
                    submitted_by_sample[sample.sample_idx] = submitted
                    expected_by_sample[sample.sample_idx] = expected

                t0 = time.perf_counter()
                outs = llm.generate(
                    prompts=prompts,
                    sampling_params=params,
                    use_tqdm=False,
                )
                batch_seconds = time.perf_counter() - t0

                for sample, out in zip(active, outs, strict=True):
                    text = out.outputs[0].text
                    action = parse_action(text)
                    events = getattr(out, "compaction_events", None) or []
                    prompt_len = (
                        len(out.prompt_token_ids) if out.prompt_token_ids else 0
                    )
                    cached = getattr(out, "num_cached_tokens", None) or 0
                    hit_pct = (100.0 * cached / prompt_len) if prompt_len else 0.0
                    output_len = len(out.outputs[0].token_ids)
                    expected = expected_by_sample[sample.sample_idx]
                    refill_delta = max(0, expected - cached)
                    if refill_delta > 0:
                        sample.bad_prefix_hits += 1
                        total_bad_prefix_hits += 1
                        bad_samples.add(sample.sample_idx)

                    sample.timings.append(
                        (turn, batch_seconds, prompt_len, output_len, cached)
                    )
                    tokens_evicted_this_turn = sum(
                        int(getattr(ev, "tokens_evicted", 0)) for ev in events
                    )
                    sample.cumul_events += len(events)
                    sample.cumul_tokens_evicted += tokens_evicted_this_turn

                    print(
                        f"  s{sample.sample_idx}: prompt={prompt_len} "
                        f"cached={cached} ({hit_pct:.1f}%) "
                        f"expected={expected} delta={refill_delta} "
                        f"events={len(events)}"
                    )
                    fs.write(
                        f"sample={sample.sample_idx:02d} turn={turn:02d} "
                        f"events={len(events)} prompt={prompt_len} "
                        f"output={output_len} cached={cached} "
                        f"hit_pct={hit_pct:.1f} "
                        f"phase4_expected_cached={expected} "
                        f"phase4_refill_delta={refill_delta} "
                        f"tokens_evicted={tokens_evicted_this_turn} "
                        f"cumul_events={sample.cumul_events} "
                        f"cumul_tokens_evicted={sample.cumul_tokens_evicted} "
                        f"score={sample.score}/{sample.max_score} "
                        f"done={int(sample.done)} "
                        f"batch_s={batch_seconds:.3f}\n"
                    )

                    f.write("=" * 80 + "\n")
                    f.write(f"SAMPLE {sample.sample_idx} TURN {turn}\n")
                    f.write("=" * 80 + "\n")
                    f.write(
                        f"prompt_tokens={prompt_len} output_tokens={output_len} "
                        f"cached_tokens={cached} ({hit_pct:.1f}%) "
                        f"phase4_expected_cached={expected} "
                        f"phase4_refill_delta={refill_delta} "
                        f"events={len(events)} batch_seconds={batch_seconds:.3f}\n\n"
                    )
                    if events:
                        f.write(f"COMPACTION EVENTS ({len(events)}):\n")
                        for i, ev in enumerate(events):
                            f.write(_format_event(i, ev) + "\n")
                        f.write("\n")
                    f.write("ASSISTANT:\n")
                    f.write(text + "\n\n")
                    raw = tokenizer.decode(
                        out.outputs[0].token_ids, skip_special_tokens=False
                    )
                    f.write("ASSISTANT (raw, skip_special_tokens=False):\n")
                    f.write(raw + "\n\n")
                    f.write(f"PARSED ACTION: {action!r}\n\n")

                    if PHASE4:
                        sample.prev_state_tokens = derive_next_prev_state(
                            out, submitted_by_sample[sample.sample_idx]
                        )
                    sample.conv.append({"role": "assistant", "content": text})

                    if action is None:
                        action = "look"
                        f.write("USER (no <action>, falling back to 'look'):\n\n")

                    try:
                        new_state, new_score, done = sample.env.step(str(action))
                        obs = new_state.feedback
                    except Exception as e:
                        err = f"Error stepping env: {e}"
                        sample.conv.append({"role": "user", "content": err})
                        f.write("USER (env error):\n")
                        f.write(err + "\n\n")
                        print(f"  s{sample.sample_idx}: step error: {e}")
                        continue

                    reward_delta = new_score - sample.score
                    sample.score = new_score
                    sample.done = bool(done)
                    f.write(
                        f"SCORE={sample.score}/{sample.max_score} "
                        f"(+{reward_delta}) done={done}\n\n"
                    )
                    f.write("USER (obs):\n")
                    f.write(obs + "\n\n")

                    if done:
                        sample.conv.append({
                            "role": "user",
                            "content": (
                                "Game Over! Final score: "
                                f"{sample.score}/{sample.max_score}"
                            ),
                        })
                    else:
                        sample.conv.append({"role": "user", "content": obs})
                    f.flush()
                fs.flush()

        finally:
            for sample in samples:
                try:
                    sample.env.close()
                except Exception:
                    pass

        fs.write(
            f"\n# total_bad_prefix_hits={total_bad_prefix_hits} "
            f"affected_samples={len(bad_samples)}/{N_SAMPLES} "
            f"bad_sample_ids={sorted(bad_samples)}\n"
        )
        fs.flush()

    print(
        f"\nmulti-sample trace written to {OUT_PATH}; "
        f"summary={SUMMARY_OUT_PATH}; "
        f"bad_prefix_hits={total_bad_prefix_hits} "
        f"affected_samples={len(bad_samples)}/{N_SAMPLES}"
    )


def main():
    if N_SAMPLES > 1:
        main_multi_sample()
        return

    env, game_state, max_score, game_file = reset_env(
        dataset_path=DATASET_PATH, game_idx=GAME_IDX
    )
    initial_obs = game_state.feedback

    conv = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_obs},
    ]

    score = 0

    # Phase 4 carry state: vLLM's canonical KV-state token sequence after
    # the previous turn. Initialized on turn 0 from the full first
    # submitted prompt (kept = entire submitted prompt when no compaction
    # has fired yet).
    prev_state_tokens: list[int] | None = None

    # Per-turn timing for cached-vs-uncached benchmarking. Each entry:
    # (turn, gen_seconds, prompt_tokens, output_tokens, cached_tokens).
    turn_timings: list[tuple[int, float, int, int, int]] = []
    expected_phase4_cached = 0

    with open(OUT_PATH, "w") as f, open(SUMMARY_OUT_PATH, "w") as fs:
        f.write(
            f"# TextWorld turn-mode compaction trace "
            f"(PAD={PAD} PREFIX_CACHE={PREFIX_CACHE} PHASE4={PHASE4})\n"
        )
        f.write(
            f"# dataset={DATASET_PATH} game_idx={GAME_IDX} "
            f"game_file={game_file} max_score={max_score} "
            f"max_turns={MAX_TURNS}\n"
        )
        # One-line-per-turn compaction summary log. Format:
        #   turn=N events=X cumul=Y prompt=A output=B cached=C(%) \
        #     last_evicted=Z turns_evicted_after=W tokens_evicted_this_turn=T
        fs.write(
            "# turn-mode compaction summary "
            f"(max_turns=3 stride=1 game_idx={GAME_IDX})\n"
        )
        fs.write(
            "# events = NEW events this turn (cumulative count is in cumul). "
            "tokens_evicted_this_turn sums tokens_evicted across new events.\n"
        )
        fs.flush()
        cumul_events = 0
        cumul_tokens_evicted = 0
        f.write("=" * 80 + "\n")
        f.write("SYSTEM PROMPT\n")
        f.write("=" * 80 + "\n")
        f.write(SYSTEM_PROMPT + "\n\n")
        f.write("=" * 80 + "\n")
        f.write("INITIAL USER (obs)\n")
        f.write("=" * 80 + "\n")
        f.write(initial_obs + "\n\n")
        f.flush()

        try:
            for turn in range(MAX_TURNS):
                print(f"turn {turn}...", end=" ", flush=True)

                t0 = time.perf_counter()
                if PHASE4 and prev_state_tokens is not None:
                    # Submit only the new user message after the carried
                    # post-compaction KV state. conv[-1] is the latest
                    # appended user obs (set at end of previous iteration).
                    new_user_content = conv[-1]["content"]
                    expected_phase4_cached = len(prev_state_tokens)
                    text, out, submitted_ids = chat_phase4(
                        prev_state_tokens,
                        new_user_content,
                        max_tokens=MAX_TOKENS,
                        temperature=0.0,
                        seed=turn,
                        show_summary=True,
                    )
                else:
                    expected_phase4_cached = 0
                    text, out = chat(
                        conv,
                        max_tokens=MAX_TOKENS,
                        temperature=0.0,
                        seed=turn,
                        show_pad_summary=True,
                    )
                    submitted_ids = (
                        list(out.prompt_token_ids) if out.prompt_token_ids else []
                    )
                gen_seconds = time.perf_counter() - t0
                events = getattr(out, "compaction_events", None) or []
                action = parse_action(text)

                f.write("=" * 80 + "\n")
                f.write(f"TURN {turn}\n")
                f.write("=" * 80 + "\n")
                prompt_len = (
                    len(out.prompt_token_ids) if out.prompt_token_ids else 0
                )
                # Phase 2 reporting: num_cached_tokens is set when prefix
                # caching is on. Hit rate = cached / prompt_len.
                cached = getattr(out, "num_cached_tokens", None) or 0
                hit_pct = (100.0 * cached / prompt_len) if prompt_len else 0.0
                output_len = len(out.outputs[0].token_ids)
                turn_timings.append(
                    (turn, gen_seconds, prompt_len, output_len, cached)
                )
                f.write(
                    f"prompt_tokens(scheduled)={prompt_len} "
                    f"output_tokens={output_len} "
                    f"cached_tokens={cached} ({hit_pct:.1f}%) "
                    f"phase4_expected_cached={expected_phase4_cached} "
                    f"phase4_refill_delta="
                    f"{max(0, expected_phase4_cached - cached)} "
                    f"events={len(events)} "
                    f"gen_seconds={gen_seconds:.3f}\n\n"
                )
                print(
                    f"  prompt={prompt_len} cached={cached} ({hit_pct:.1f}%) "
                    f"expected={expected_phase4_cached} "
                    f"delta={max(0, expected_phase4_cached - cached)} "
                    f"events={len(events)} t={gen_seconds:.3f}s"
                )

                if events:
                    f.write(f"COMPACTION EVENTS ({len(events)}):\n")
                    for i, ev in enumerate(events):
                        f.write(_format_event(i, ev) + "\n")
                    f.write("\n")

                # Summary log: one line per turn. Each chat() submits a new
                # vLLM request, so `events` here is the per-request list for
                # THIS turn (not cumulative across turns).
                tokens_evicted_this_turn = sum(
                    int(getattr(ev, "tokens_evicted", 0)) for ev in events
                )
                cumul_events += len(events)
                cumul_tokens_evicted += tokens_evicted_this_turn
                last_ev = events[-1] if events else None
                last_turn_evicted = (
                    getattr(last_ev, "last_turn_evicted", -1) if last_ev else -1
                )
                turns_evicted_after = (
                    getattr(last_ev, "num_turns_evicted_after", 0) if last_ev else 0
                )
                fs.write(
                    f"turn={turn:2d} events={len(events)} "
                    f"prompt={prompt_len} output={output_len} "
                    f"cached={cached} ({hit_pct:5.1f}%) "
                    f"phase4_expected_cached={expected_phase4_cached} "
                    f"phase4_refill_delta="
                    f"{max(0, expected_phase4_cached - cached)} "
                    f"tokens_evicted={tokens_evicted_this_turn} "
                    f"last_turn_evicted={last_turn_evicted} "
                    f"turns_evicted_after={turns_evicted_after} "
                    f"cumul_events={cumul_events} "
                    f"cumul_tokens_evicted={cumul_tokens_evicted}\n"
                )
                # Decoded evicted text per event (requires
                # VLLM_COMPACTION_DEBUG_TOKENS=1 — set at top of file).
                for i, ev in enumerate(events):
                    ids = list(getattr(ev, "evicted_token_ids", []) or [])
                    if not ids:
                        continue
                    try:
                        decoded = tokenizer.decode(ids, skip_special_tokens=False)
                    except Exception as e:
                        decoded = f"<decode-failed: {e}>"
                    fs.write(f"  evicted[event {i}] ({len(ids)} tok):\n")
                    for ln in (decoded.splitlines() or [decoded]):
                        fs.write(f"    | {ln}\n")
                # Decoded assistant output for this turn.
                try:
                    out_decoded = tokenizer.decode(
                        out.outputs[0].token_ids, skip_special_tokens=False
                    )
                except Exception as e:
                    out_decoded = f"<decode-failed: {e}>"
                fs.write(f"  output ({output_len} tok):\n")
                for ln in (out_decoded.splitlines() or [out_decoded]):
                    fs.write(f"    | {ln}\n")
                fs.write("\n")
                fs.flush()

                f.write("ASSISTANT:\n")
                f.write(text + "\n\n")
                raw = tokenizer.decode(
                    out.outputs[0].token_ids, skip_special_tokens=False
                )
                f.write("ASSISTANT (raw, skip_special_tokens=False):\n")
                f.write(raw + "\n\n")
                f.write(f"PARSED ACTION: {action!r}\n\n")

                # Phase 4 carry: derive post-turn KV state for the next
                # iteration. derive_next_prev_state uses the LAST event's
                # kept_token_ids when compaction fired, else the full
                # submitted prompt, then appends asst output. This is
                # what vLLM physically has in cache going into turn N+1.
                if PHASE4:
                    prev_state_tokens = derive_next_prev_state(out, submitted_ids)

                conv.append({"role": "assistant", "content": text})

                if action is None:
                    # Matches textworld_env.py fallback: use "look" instead
                    # of raising, so we exercise the compaction loop even
                    # when the model doesn't emit a well-formed <action>.
                    f.write("USER (no <action>, falling back to 'look'):\n\n")
                    action = "look"

                try:
                    new_state, new_score, done = env.step(str(action))
                    obs = new_state.feedback
                except Exception as e:
                    err = f"Error stepping env: {e}"
                    f.write("USER (env error):\n")
                    f.write(err + "\n\n")
                    f.flush()
                    conv.append({"role": "user", "content": err})
                    print(f"(step error: {e})")
                    continue

                reward_delta = new_score - score
                score = new_score

                f.write(
                    f"SCORE={score}/{max_score} (+{reward_delta}) "
                    f"done={done}\n\n"
                )
                f.write("USER (obs):\n")
                f.write(obs + "\n\n")
                f.flush()

                if done:
                    conv.append({
                        "role": "user",
                        "content": f"Game Over! Final score: {score}/{max_score}",
                    })
                    f.write("=" * 80 + "\n")
                    f.write(f"EPISODE FINISHED: score={score}/{max_score}\n")
                    f.write("=" * 80 + "\n")
                    print(f"episode finished (score={score}/{max_score})")
                    break

                conv.append({"role": "user", "content": obs})
                print(
                    f"action={action!r} score={score}/{max_score} "
                    f"(+{reward_delta})"
                )
        finally:
            try:
                env.close()
            except Exception:
                pass

        # ---------------------------------------------------------------
        # Timing summary (for cached-vs-uncached benchmarking)
        # ---------------------------------------------------------------
        if turn_timings:
            total_gen = sum(t[1] for t in turn_timings)
            total_prompt = sum(t[2] for t in turn_timings)
            total_output = sum(t[3] for t in turn_timings)
            total_cached = sum(t[4] for t in turn_timings)
            n = len(turn_timings)
            hit_overall = (100.0 * total_cached / total_prompt) if total_prompt else 0.0
            tok_per_s = (total_prompt + total_output) / total_gen if total_gen else 0.0

            # Per-token timings (ms). Multiple denominators because each
            # answers a different question:
            #   - per_total_tok: overall throughput inverse (lower = faster).
            #   - per_prompt_tok: prefill-side cost; should drop with caching.
            #   - per_output_tok: decode-side cost; caching shouldn't move this.
            #   - per_uncached_prompt_tok: prefill cost amortized only over
            #     tokens that actually had to be prefilled (prompt - cached).
            uncached_prompt = max(total_prompt - total_cached, 0)
            ms_per_total = 1000.0 * total_gen / max(total_prompt + total_output, 1)
            ms_per_prompt = 1000.0 * total_gen / max(total_prompt, 1)
            ms_per_output = 1000.0 * total_gen / max(total_output, 1)
            ms_per_uncached_prompt = (
                1000.0 * total_gen / uncached_prompt if uncached_prompt > 0 else float("nan")
            )

            f.write("=" * 80 + "\n")
            f.write(
                f"TIMING SUMMARY (PREFIX_CACHE={PREFIX_CACHE} PAD={PAD} "
                f"PHASE4={PHASE4})\n"
            )
            f.write("=" * 80 + "\n")
            f.write(
                f"turns={n} total_gen_s={total_gen:.3f} "
                f"avg_per_turn_s={total_gen / n:.3f}\n"
                f"total_prompt_tokens={total_prompt} "
                f"total_output_tokens={total_output} "
                f"total_cached={total_cached} ({hit_overall:.1f}%)\n"
                f"throughput_tokens_per_s={tok_per_s:.1f}\n"
                f"ms_per_total_tok={ms_per_total:.3f} "
                f"ms_per_prompt_tok={ms_per_prompt:.3f} "
                f"ms_per_output_tok={ms_per_output:.3f} "
                f"ms_per_uncached_prompt_tok={ms_per_uncached_prompt:.3f}\n\n"
            )
            f.write("per-turn (turn, gen_s, prompt, output, cached, hit%):\n")
            for tt in turn_timings:
                turn_i, gs, pl, ol, c = tt
                hp = (100.0 * c / pl) if pl else 0.0
                f.write(
                    f"  turn={turn_i:3d} t={gs:7.3f}s "
                    f"prompt={pl:6d} output={ol:5d} "
                    f"cached={c:6d} ({hp:5.1f}%)\n"
                )

            print(
                f"\n=== TIMING SUMMARY (PREFIX_CACHE={PREFIX_CACHE}) ===\n"
                f"turns={n} total_gen={total_gen:.3f}s "
                f"avg/turn={total_gen / n:.3f}s\n"
                f"prompt_toks={total_prompt} output_toks={total_output} "
                f"cached={total_cached} ({hit_overall:.1f}%)\n"
                f"throughput={tok_per_s:.1f} tok/s\n"
                f"ms/total_tok={ms_per_total:.3f} "
                f"ms/prompt_tok={ms_per_prompt:.3f} "
                f"ms/output_tok={ms_per_output:.3f} "
                f"ms/uncached_prompt_tok={ms_per_uncached_prompt:.3f}"
            )

    print(f"\nfull trace written to {OUT_PATH} (PAD={PAD})")
    print(f"size: {os.path.getsize(OUT_PATH)} bytes")


if __name__ == "__main__":
    main()
