"""OpenAI-compatible chat backend for the agent harness.

Many providers with free tiers (Groq, Google AI Studio's Gemini endpoint,
OpenRouter, Cerebras) speak the OpenAI chat-completions protocol. This module
lets the harness use them without changing its agent loop: `OpenAIChatClient`
has the same `messages.create(...)` interface as the Anthropic client and
returns responses in the same shape (content blocks of type `text` and
`tool_use`, a `usage` record and a `stop_reason`).

Standard library only (urllib), so no extra dependency. The API key is read
from the environment variable named by the provider preset and is never
written to disk or logged.

Free tiers are rate limited per minute and per day. The client spaces
requests (`min_interval_s`) and retries rate-limit and server errors with
backoff, honouring `Retry-After`. When retries run out it raises
`ProviderError`; the harness then stops the attempt and labels it
`api_error`, and the summary leaves such attempts out of the pass rate
(they say nothing about the agent).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    min_interval_s: float  # spacing between requests, from the free tier's per-minute limit


PROVIDERS = {
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", 2.5),
    "gemini": Provider(
        "gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", 6.5
    ),
    "openrouter": Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", 3.5),
    "cerebras": Provider("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", 2.5),
}
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
INVALID_ARGS = "printf 'the tool call had invalid JSON arguments\\n' >&2; exit 2"


class ProviderError(RuntimeError):
    """The provider refused or kept failing; nothing about the agent."""


class _Block(SimpleNamespace):
    def model_dump(self, **_: Any) -> dict[str, Any]:
        return dict(vars(self))


# ---------------------------------------------------------------- conversion


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic tool definitions -> OpenAI function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    return "\n\n".join(b["text"] for b in system if b.get("type") == "text")


def _user_messages(content: Any) -> list[dict[str, Any]]:
    """One Anthropic user turn -> OpenAI messages: tool results become `tool`
    messages (one per call), any text becomes a user message."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    out: list[dict[str, Any]] = []
    texts = []
    for block in content:
        if block.get("type") == "tool_result":
            body = block.get("content", "")
            if not isinstance(body, str):
                body = "".join(b.get("text", "") for b in body)
            out.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": body})
        elif block.get("type") == "text":
            texts.append(block["text"])
    if texts:
        out.append({"role": "user", "content": "\n".join(texts)})
    return out


def _assistant_message(content: list[dict[str, Any]]) -> dict[str, Any]:
    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    calls = [
        {
            "id": b["id"],
            "type": "function",
            "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))},
        }
        for b in content
        if b.get("type") == "tool_use"
    ]
    msg: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        msg["tool_calls"] = calls
    return msg


def to_openai_messages(system: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-format conversation (as the harness keeps it) -> OpenAI chat
    messages. Cache-control markers are Anthropic-only and are dropped."""
    out = [{"role": "system", "content": _system_text(system)}]
    for m in messages:
        if m["role"] == "user":
            out.extend(_user_messages(m["content"]))
        else:
            content = m["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            out.append(_assistant_message(content))
    return out


def _tool_block(call: dict[str, Any], index: int) -> _Block:
    fn = call.get("function", {})
    raw = fn.get("arguments") or "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        if not isinstance(args, dict):
            raise ValueError
    except ValueError:
        # Hand the agent an error it can see instead of silently running nothing.
        args = {"command": INVALID_ARGS}
    return _Block(
        type="tool_use",
        id=call.get("id") or f"call_{index}",
        name=fn.get("name", "bash"),
        input=args,
    )


STOP = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}


def from_openai_response(data: dict[str, Any]) -> SimpleNamespace:
    """OpenAI chat-completion JSON -> an Anthropic-shaped response."""
    if not data.get("choices"):
        raise ProviderError(f"no choices in response: {str(data)[:300]}")
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    blocks: list[_Block] = []
    if msg.get("content"):
        blocks.append(_Block(type="text", text=msg["content"]))
    blocks += [_tool_block(c, i) for i, c in enumerate(msg.get("tool_calls") or [])]
    u = data.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    usage = SimpleNamespace(
        input_tokens=(u.get("prompt_tokens") or 0) - cached,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=cached,
        output_tokens=u.get("completion_tokens") or 0,
    )
    reason = choice.get("finish_reason") or "stop"
    return SimpleNamespace(content=blocks, usage=usage, stop_reason=STOP.get(reason, reason))


# ---------------------------------------------------------------- transport


class OpenAIChatClient:
    """Drop-in for `anthropic.Anthropic()` in the harness's agent loop."""

    def __init__(
        self,
        provider: Provider,
        api_key: str,
        *,
        min_interval_s: float | None = None,
        max_retries: int = 6,
        timeout_s: float = 300.0,
    ) -> None:
        self.provider = provider
        self._key = api_key
        self.min_interval_s = provider.min_interval_s if min_interval_s is None else min_interval_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._last = 0.0
        self.messages = self

    @classmethod
    def from_env(cls, name: str, **kw: Any) -> OpenAIChatClient:
        provider = PROVIDERS[name]
        key = os.environ.get(provider.key_env)
        if not key:
            raise ProviderError(f"set {provider.key_env} in the environment")
        return cls(provider, key, **kw)

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: Any,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> SimpleNamespace:
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": to_openai_messages(system, messages),
            "tools": to_openai_tools(tools),
            "tool_choice": "auto",
        }
        return from_openai_response(self._post("/chat/completions", body))

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self.provider.base_url.rstrip("/") + path
        payload = json.dumps(body).encode()
        delay = 5.0
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            req = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                    "User-Agent": "bioeng-termbench-agent-eval",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code not in RETRY_STATUS or attempt == self.max_retries:
                    raise ProviderError(
                        f"HTTP {e.code} from {self.provider.name}: {detail}"
                    ) from None
                retry_after = e.headers.get("Retry-After") if e.headers else None
                time.sleep(_retry_delay(retry_after, delay))
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise ProviderError(f"{self.provider.name} unreachable: {e}") from None
                time.sleep(delay)
            delay = min(delay * 2, 120.0)
        raise ProviderError("unreachable")  # pragma: no cover


def _retry_delay(header: str | None, fallback: float) -> float:
    try:
        return min(max(float(header), 1.0), 300.0) if header else fallback
    except ValueError:
        return fallback
