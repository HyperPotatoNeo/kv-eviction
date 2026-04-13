# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the AsyncOpenAI.chat.completions.create interceptor
installed by `kv_eviction.env._install_message_padding_interceptor`.

Replaces the underlying `orig_create` with an AsyncMock that records the
kwargs it received, then invokes the installed wrapper and asserts the
wrapper has:
  - not touched kwargs when padding is disabled,
  - injected `extra_body={"prompt_token_ids": ...}` when padding is
    enabled,
  - merged into (rather than clobbered) a pre-existing `extra_body`,
  - stashed the padded ids onto the returned response object so
    Patch #1 can forward them to the verifiers Response.

No HTTP, no real model, no tokenizer: a stub tokenizer with a fixed
return value is enough to exercise the branching logic.
"""

from types import SimpleNamespace

import pytest

import kv_eviction.env as env


class _StubTok:
    pad_token_id = 100

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False
    ):
        return "TOK:"

    def encode(self, s, add_special_tokens=False):
        # raw: 3 tokens before an <|im_end|>; block_size=4 means 0 padding
        # (body=3, target=3).
        return [1, 2, 3, 999]

    def convert_tokens_to_ids(self, tok):
        return 999 if tok == "<|im_end|>" else -1


@pytest.fixture
def stub_response():
    # Pydantic-style stand-in; we only care that setattr works for
    # `prompt_token_ids` forwarding.
    return SimpleNamespace(id="resp-0", prompt_token_ids=None)


@pytest.fixture(autouse=True)
def reset_padding_config():
    yield
    env._padding_config = None


def _get_patched_create():
    from openai.resources.chat.completions.completions import AsyncCompletions

    return AsyncCompletions.create


def test_padding_disabled_is_passthrough(stub_response):
    import asyncio
    from unittest.mock import patch

    env._padding_config = None
    called_with = {}

    async def fake_orig(self, *args, **kwargs):
        called_with.update(kwargs)
        return stub_response

    # Reinstall interceptor on top of fake_orig.
    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create

        async def run():
            return await patched(
                self=None,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
            )

        asyncio.run(run())

    assert "extra_body" not in called_with
    assert stub_response.prompt_token_ids is None


def test_padding_enabled_injects_extra_body(stub_response):
    import asyncio
    from unittest.mock import patch

    called_with = {}

    async def fake_orig(self, *args, **kwargs):
        called_with.update(kwargs)
        return stub_response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
        )

        async def run():
            return await patched(
                self=None,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
            )

        asyncio.run(run())

    assert "extra_body" in called_with
    # raw=[1,2,3,999] body=3 tokens before <|im_end|> at block_size=4
    # -> no padding needed, prompt_token_ids == raw.
    assert called_with["extra_body"]["prompt_token_ids"] == [1, 2, 3, 999]
    # Forward-stash on response.
    assert stub_response.prompt_token_ids == [1, 2, 3, 999]


def test_padding_merges_preexisting_extra_body(stub_response):
    import asyncio
    from unittest.mock import patch

    called_with = {}

    async def fake_orig(self, *args, **kwargs):
        called_with.update(kwargs)
        return stub_response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
        )

        async def run():
            return await patched(
                self=None,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                extra_body={"some_other_flag": True},
            )

        asyncio.run(run())

    assert called_with["extra_body"]["some_other_flag"] is True
    assert called_with["extra_body"]["prompt_token_ids"] == [1, 2, 3, 999]
