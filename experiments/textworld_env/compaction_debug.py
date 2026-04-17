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
from pathlib import Path

# ---------------------------------------------------------------------------
# Must be set BEFORE importing vllm. Keeps the scheduler in-process so
# debugpy in this process can actually hit breakpoints inside it.
# ---------------------------------------------------------------------------
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_COMPACTION_DEBUG_TOKENS", "1")


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
BLOCK_SIZE = 16
PAD_FILLER_ID: int | None = None
MAX_TURNS = int(os.environ.get("MAX_TURNS", "30"))
DATASET_PATH = os.environ.get(
    "DATASET",
    "/home/toolkit/kv-eviction/textworld_cooking_mix",
)
GAME_IDX = int(os.environ.get("GAME_IDX", "500"))
OUT_PATH = os.environ.get(
    "OUT_PATH",
    "/home/toolkit/kv-eviction/experiments/textworld_env/out_turn_debug.txt",
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
    enable_prefix_caching=False,
    block_size=BLOCK_SIZE,
    # Block-FIFO fallback (fires if KV exceeds window before turn-mode
    # accumulates max_turns completed turns).
    compaction_window_size=4096,
    compaction_stride=512,
    # -1 = auto-detect system prompt end from first <|im_end|>.
    compaction_protected_prefix_tokens=-1,
    # Turn-mode: 2-1 aggressive — evict 1 oldest completed turn whenever ≥2 live.
    compaction_max_turns=2,
    compaction_eviction_turn_stride=1,
    compaction_turn_end_token_id=None,
    # With PAD=True in the chat() helper, every <|im_end|> is padded so
    # that the next message's first token lands on a block boundary.
    # This flag tells the turn-mode planner to snap evict_end UP to that
    # boundary (instead of the default inward snap), eliminating the
    # orphan tail-of-last-evicted-turn that otherwise survives in KV.
    compaction_assume_aligned_turn_boundaries=PAD,
    async_scheduling=False,
)
tokenizer = llm.get_tokenizer()
print(f"loaded (turn-mode, PAD={PAD})")


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


def chat(messages, max_tokens=512, temperature=1.0, seed=0, show_pad_summary=True):
    sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed)

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


def main():
    env, game_state, max_score, game_file = reset_env(
        dataset_path=DATASET_PATH, game_idx=GAME_IDX
    )
    initial_obs = game_state.feedback

    conv = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_obs},
    ]

    score = 0

    with open(OUT_PATH, "w") as f:
        f.write(f"# TextWorld turn-mode compaction trace (PAD={PAD})\n")
        f.write(
            f"# dataset={DATASET_PATH} game_idx={GAME_IDX} "
            f"game_file={game_file} max_score={max_score} "
            f"max_turns={MAX_TURNS}\n"
        )
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

                text, out = chat(
                    conv,
                    max_tokens=2000,
                    temperature=1.0,
                    seed=turn,
                    show_pad_summary=True,
                )
                events = getattr(out, "compaction_events", None) or []
                action = parse_action(text)

                f.write("=" * 80 + "\n")
                f.write(f"TURN {turn}\n")
                f.write("=" * 80 + "\n")
                f.write(
                    f"prompt_tokens(scheduled)="
                    f"{len(out.prompt_token_ids) if out.prompt_token_ids else '?'} "
                    f"output_tokens={len(out.outputs[0].token_ids)} "
                    f"events={len(events)}\n\n"
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

    print(f"\nfull trace written to {OUT_PATH} (PAD={PAD})")
    print(f"size: {os.path.getsize(OUT_PATH)} bytes")


if __name__ == "__main__":
    main()
