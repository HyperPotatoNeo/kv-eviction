"""Debugpy-friendly standalone version of compaction_test.ipynb.

Runs a short BabyAI episode against a turn-mode-compaction vLLM engine,
with debugpy wired up so you can attach from VS Code / Cursor and set
breakpoints in vllm/v1/core/sched/scheduler.py.

Usage:
    # Default: waits for a debugger to attach before doing anything.
    python experiments/debug_balrog/compaction_debug.py

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
    #   MAX_TURNS=10                how many BabyAI turns to run
    #   PAD=1                       pad messages so <|im_end|> lands on
    #                               block boundary (matches notebook default)

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
# Compaction config knobs (mirror the notebook)
# ---------------------------------------------------------------------------
PAD = os.environ.get("PAD", "1") == "1"
BLOCK_SIZE = 16
PAD_FILLER_ID: int | None = None
MAX_TURNS = int(os.environ.get("MAX_TURNS", "20"))
TASK = os.environ.get("TASK", "BabyAI-MixedTrainLocal-v0/putnext")
SEED = int(os.environ.get("SEED", "0"))
OUT_PATH = os.environ.get(
    "OUT_PATH",
    "/home/toolkit/kv-eviction/experiments/debug_balrog/out_turn_debug.txt",
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    max_model_len=8192,
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    enable_prefix_caching=False,
    block_size=BLOCK_SIZE,
    compaction_window_size=1024,
    compaction_stride=256,
    compaction_protected_prefix_tokens=0,
    compaction_max_turns=4,
    compaction_eviction_turn_stride=2,
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
# Chat helper + tools
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "take_action",
            "description": "Take an action in the BabyAI environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": (
                            "Action to take (e.g. 'forward', 'left', "
                            "'right', 'pickup', 'drop', 'toggle', 'done')."
                        ),
                    }
                },
                "required": ["action"],
            },
        },
    }
]


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


def _render_padded(messages, tools, im_end_id, block_size, filler_id):
    rendered = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, tokenize=False
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


def chat(messages, tools=None, max_tokens=512, temperature=1.0, seed=0,
         show_pad_summary=True):
    sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed)

    if PAD:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assert isinstance(im_end_id, int) and im_end_id >= 0, (
            f"<|im_end|> not in tokenizer (got {im_end_id!r})"
        )
        filler_id = _filler_token_id()
        raw, padded, pads = _render_padded(
            messages, tools, im_end_id, BLOCK_SIZE, filler_id
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
        outs = llm.chat(
            messages=messages, tools=tools, sampling_params=sp, use_tqdm=False
        )

    out = outs[0]
    text = out.outputs[0].text
    return text, out


# ---------------------------------------------------------------------------
# BALROG loop
# ---------------------------------------------------------------------------
from balrog.environments import make_env  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

_balrog_cfg = OmegaConf.load(
    "/home/toolkit/kv-eviction/.venv/lib/python3.12/site-packages/balrog/config/config.yaml"
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def reset_env(task=TASK, seed=SEED):
    env = make_env("babyai", task, _balrog_cfg)
    obs, _ = env.reset(seed=seed)
    return env, obs


def obs_text(obs):
    if isinstance(obs, dict) and "text" in obs and "long_term_context" in obs["text"]:
        return obs["text"]["long_term_context"]
    return str(obs)


def parse_action(text):
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        blob = json.loads(m.group(1))
        return blob.get("arguments", {}).get("action")
    except Exception:
        return None


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
    env, obs = reset_env(task=TASK, seed=SEED)
    system_prompt = env.get_instruction_prompt(instructions=obs["mission"])

    conv = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": obs_text(obs)},
    ]

    with open(OUT_PATH, "w") as f:
        f.write(f"# BALROG turn-mode compaction trace (PAD={PAD})\n")
        f.write(f"# task: {TASK}, seed={SEED}, max_turns={MAX_TURNS}\n")
        f.write("=" * 80 + "\n")
        f.write("SYSTEM PROMPT\n")
        f.write("=" * 80 + "\n")
        f.write(system_prompt + "\n\n")
        f.write("=" * 80 + "\n")
        f.write("INITIAL USER (obs)\n")
        f.write("=" * 80 + "\n")
        f.write(obs_text(obs) + "\n\n")
        f.flush()

        for turn in range(MAX_TURNS):
            print(f"turn {turn}...", end=" ", flush=True)

            text, out = chat(
                conv,
                tools=TOOLS,
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
                err = (
                    "Error: No valid tool_call found. Emit a <tool_call> "
                    "with take_action."
                )
                f.write("USER (env error):\n")
                f.write(err + "\n\n")
                f.flush()
                conv.append({"role": "user", "content": err})
                print("(no action)")
                continue

            try:
                valid = env.check_action_validity(action)
                obs, reward, term, trunc, info = env.step(valid)
            except Exception as e:
                err = f"Error stepping env: {e}"
                f.write("USER (env error):\n")
                f.write(err + "\n\n")
                f.flush()
                conv.append({"role": "user", "content": err})
                print(f"(step error: {e})")
                continue

            f.write(f"REWARD={reward} done={term or trunc}\n\n")
            next_obs_txt = obs_text(obs)
            f.write("USER (obs):\n")
            f.write(next_obs_txt + "\n\n")
            f.flush()

            conv.append({"role": "user", "content": next_obs_txt})
            print(f"action={action!r} reward={reward}")
            if term or trunc:
                f.write("=" * 80 + "\n")
                f.write("EPISODE FINISHED\n")
                f.write("=" * 80 + "\n")
                print("episode finished")
                break

    print(f"\nfull trace written to {OUT_PATH} (PAD={PAD})")
    print(f"size: {os.path.getsize(OUT_PATH)} bytes")


if __name__ == "__main__":
    main()
