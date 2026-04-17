"""Training-capable verifiers wrapper around Sierra's original tau-bench.

Unlike `tau2-bench` / `tau3-bench` (eval-only), `tau-bench` ships a real
retail train split, so we can RL-train on it directly:

  domain="retail"  -> 500 train tasks  (dataset)
                       115 test tasks  (eval_dataset)
  domain="airline" ->  50 test tasks   (dataset aliased = eval_dataset)

The env id is `tau-bench-train` (verifiers maps this to the installed module
`tau_bench_train`, i.e. this file).

User simulator routing
----------------------
`tau_bench.envs.user.LLMUserSimulationEnv` calls `litellm.completion(...)`
directly, with no hook for `api_base` / `api_key`. At import time we
monkey-patch the module-level `completion` symbol (idempotent, sentinel-
guarded) so calls inherit `api_base` / `api_key` from env vars, which the
wrapper sets from `user_base_url` / `user_api_key_var`. This routes the
user simulator at a local vLLM endpoint without forking tau-bench.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Optional, TypeVar

import verifiers as vf
from datasets import Dataset

import tau_bench.envs.user as _tau_user
from tau_bench.envs.airline.env import MockAirlineDomainEnv
from tau_bench.envs.retail.env import MockRetailDomainEnv
from tau_bench.types import RESPOND_ACTION_NAME, Action

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Monkey-patch: route tau-bench user simulator to a local vLLM endpoint.
# ---------------------------------------------------------------------------
_PATCH_SENTINEL = "__kv_eviction_patched__"


def _apply_user_completion_patch() -> None:
    orig = _tau_user.completion
    if getattr(orig, _PATCH_SENTINEL, False):
        return

    def _patched(*args: Any, **kwargs: Any) -> Any:
        base = os.environ.get("TAUBENCH_USER_BASE_URL")
        key_var = os.environ.get("TAUBENCH_USER_API_KEY_VAR", "OPENAI_API_KEY")
        key = os.environ.get(key_var) or os.environ.get("TAUBENCH_USER_API_KEY")
        if base:
            kwargs.setdefault("api_base", base)
        if key:
            kwargs.setdefault("api_key", key)
        return orig(*args, **kwargs)

    setattr(_patched, _PATCH_SENTINEL, True)
    _tau_user.completion = _patched


_apply_user_completion_patch()


# ---------------------------------------------------------------------------
# Domain helpers.
# ---------------------------------------------------------------------------
def _make_probe_env(domain: str, task_split: str):
    """Construct a domain env with the human user strategy (no API calls)
    purely to harvest metadata: tasks, tools_info, wiki. Never reset/step on
    this env; it's discarded after metadata extraction.

    We pass `task_index=0` to sidestep an upstream bug in
    `tau_bench.envs.base.Env.__init__` (line 69), which does
    `random.randint(0, len(tasks))` — inclusive on both ends, so it can
    return `len(tasks)` which crashes the next line `tasks[self.task_index]`.
    """
    if domain == "retail":
        return MockRetailDomainEnv(user_strategy="human", task_split=task_split, task_index=0)
    if domain == "airline":
        return MockAirlineDomainEnv(user_strategy="human", task_split="test", task_index=0)
    raise ValueError(f"Unknown domain: {domain}")


def _make_rollout_env(
    domain: str,
    task_split: str,
    user_model: str,
    user_provider: str,
):
    """Construct a domain env with LLM user strategy for a live rollout.
    Must be called after `_apply_user_completion_patch` so the user sim's
    construction-time `litellm.completion` call is routed to the local vLLM.

    `task_index=0` sidesteps the upstream off-by-one in `Env.__init__`
    (see `_make_probe_env` docstring). `setup_state` overrides with the
    per-rollout index via `tau_env.reset(task_index=...)`.
    """
    if domain == "retail":
        return MockRetailDomainEnv(
            user_strategy="llm",
            user_model=user_model,
            user_provider=user_provider,
            task_split=task_split,
            task_index=0,
        )
    if domain == "airline":
        return MockAirlineDomainEnv(
            user_strategy="llm",
            user_model=user_model,
            user_provider=user_provider,
            task_split="test",
            task_index=0,
        )
    raise ValueError(f"Unknown domain: {domain}")


# ---------------------------------------------------------------------------
# Env.
# ---------------------------------------------------------------------------
class TauBenchTrainEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        domain: str = "retail",
        user_model: str = "gpt-4o",
        user_provider: str = "openai",
        user_base_url: Optional[str] = None,
        user_api_key_var: str = "OPENAI_API_KEY",
        max_num_steps: int = 30,
        max_workers: int = 128,
        max_turns: int = -1,
        **kwargs: Any,
    ):
        if domain not in ("retail", "airline"):
            raise ValueError(f"Unknown domain: {domain!r}; expected 'retail' or 'airline'")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.domain = domain
        self.user_model = user_model
        self.user_provider = user_provider
        self.max_num_steps = max_num_steps
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tau-bench-train",
        )

        # Env var plumbing for the litellm monkey-patch. Set before any
        # rollout MockDomainEnv construction so the eager user-sim
        # completion call in LLMUserSimulationEnv.__init__ is routed.
        if user_base_url:
            os.environ["TAUBENCH_USER_BASE_URL"] = user_base_url
        if user_api_key_var:
            os.environ["TAUBENCH_USER_API_KEY_VAR"] = user_api_key_var

        # --- Probe envs for metadata ---------------------------------------
        if domain == "retail":
            train_probe = _make_probe_env("retail", "train")
            eval_probe = _make_probe_env("retail", "test")
        else:  # airline has no train split
            train_probe = _make_probe_env("airline", "test")
            eval_probe = train_probe

        system_prompt = train_probe.wiki

        tool_defs = [
            vf.Tool(
                name=t["function"]["name"],
                description=t["function"].get("description", ""),
                parameters=t["function"].get("parameters", {}),
                strict=False,
            )
            for t in train_probe.tools_info
        ]

        train_split_name = "train" if domain == "retail" else "test"

        def _rows(tasks, split_name: str) -> list[dict]:
            return [
                {
                    "prompt": [{"role": "system", "content": system_prompt}],
                    "info": {
                        "task_index": i,
                        "domain": domain,
                        "task_split": split_name,
                    },
                }
                for i in range(len(tasks))
            ]

        dataset = Dataset.from_list(_rows(train_probe.tasks, train_split_name))
        eval_dataset = Dataset.from_list(_rows(eval_probe.tasks, "test"))

        rubric = vf.Rubric(funcs=[self._reward], weights=[1.0])

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric,
            tool_defs=tool_defs,
            max_turns=max_turns,
            message_type="chat",
            **kwargs,
        )

    # -----------------------------------------------------------------
    # Async helpers.
    # -----------------------------------------------------------------
    async def _run_in_thread(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool, partial(func, *args, **kwargs)
        )

    # -----------------------------------------------------------------
    # Lifecycle.
    # -----------------------------------------------------------------
    async def setup_state(self, state: vf.State) -> vf.State:
        info = state["info"]
        task_index = int(info["task_index"])
        task_split = info["task_split"]

        tau_env = await self._run_in_thread(
            _make_rollout_env,
            self.domain,
            task_split,
            self.user_model,
            self.user_provider,
        )
        reset_res = await self._run_in_thread(tau_env.reset, task_index=task_index)

        state["tau_env"] = tau_env
        state["done"] = False
        state["tau_reward"] = 0.0
        state["step_count"] = 0

        # Append the user simulator's opening paraphrase as the first user turn.
        state["prompt"].append(
            vf.UserMessage(content=reset_res.observation or "")
        )
        return state

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        tau_env = state["tau_env"]
        assert isinstance(messages, list)
        last = messages[-1]

        tool_calls = getattr(last, "tool_calls", None) or []
        if tool_calls:
            responses: list[vf.Message] = []
            for tc in tool_calls:
                try:
                    kwargs_parsed = json.loads(tc.arguments or "{}")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid tool-call arguments: {e}")
                    kwargs_parsed = {}
                action = Action(name=tc.name, kwargs=kwargs_parsed)
                env_resp = await self._run_in_thread(tau_env.step, action)
                responses.append(
                    vf.ToolMessage(
                        tool_call_id=tc.id,
                        content=str(env_resp.observation or ""),
                    )
                )
                state["step_count"] += 1
                state["tau_reward"] = float(env_resp.reward)
                if env_resp.done:
                    state["done"] = True
                    break
            return responses

        # Plain assistant content → route to user simulator via respond action.
        content = getattr(last, "content", "") or ""
        if isinstance(content, list):
            content = "".join(
                getattr(p, "text", "") for p in content if getattr(p, "type", "") == "text"
            )
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
        env_resp = await self._run_in_thread(tau_env.step, action)
        state["step_count"] += 1
        state["tau_reward"] = float(env_resp.reward)
        if env_resp.done:
            state["done"] = True
        return [vf.UserMessage(content=str(env_resp.observation or ""))]

    # -----------------------------------------------------------------
    # Stop predicates.
    # -----------------------------------------------------------------
    @vf.stop
    async def task_done(self, state: vf.State) -> bool:
        return bool(state.get("done", False))

    @vf.stop
    async def max_steps_reached(self, state: vf.State) -> bool:
        return int(state.get("step_count", 0)) >= self.max_num_steps

    # -----------------------------------------------------------------
    # Cleanup.
    # -----------------------------------------------------------------
    @vf.cleanup
    async def cleanup_rollout(self, state: vf.State) -> None:
        state.pop("tau_env", None)

    # -----------------------------------------------------------------
    # Reward.
    # -----------------------------------------------------------------
    async def _reward(self, state: vf.State, **kwargs: Any) -> float:
        # tau-bench's step() calls calculate_reward() when done=True and stores
        # the reward in env_resp.reward (0 or 1). We already tracked it in
        # state["tau_reward"]. If the rollout ended without done (e.g. max
        # turns reached), fall back to recomputing via calculate_reward.
        reward = state.get("tau_reward", 0.0)
        if reward == 0.0 and state.get("done") is False:
            tau_env = state.get("tau_env")
            if tau_env is not None:
                try:
                    res = await self._run_in_thread(tau_env.calculate_reward)
                    reward = float(res.reward)
                except Exception as e:
                    self.logger.warning(f"calculate_reward failed: {e}")
        return float(reward)


def load_environment(
    domain: str = "retail",
    user_model: str = "gpt-4o",
    user_provider: str = "openai",
    user_base_url: Optional[str] = None,
    user_api_key_var: str = "OPENAI_API_KEY",
    max_num_steps: int = 30,
    max_workers: int = 128,
    max_turns: int = -1,
    **kwargs: Any,
) -> vf.Environment:
    return TauBenchTrainEnv(
        domain=domain,
        user_model=user_model,
        user_provider=user_provider,
        user_base_url=user_base_url,
        user_api_key_var=user_api_key_var,
        max_num_steps=max_num_steps,
        max_workers=max_workers,
        max_turns=max_turns,
        **kwargs,
    )
