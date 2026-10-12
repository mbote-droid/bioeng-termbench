"""The OpenAI-compatible backend: message conversion both ways, retries and
rate-limit handling, and a full agent loop through it against a faked
provider. No network, no API key."""

import argparse
import importlib.util
import io
import json
import sys
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent_eval"))
oc = importlib.import_module("openai_compat")
spec = importlib.util.spec_from_file_location("harness", ROOT / "agent_eval/harness.py")
harness = importlib.util.module_from_spec(spec)
sys.modules["harness"] = harness
spec.loader.exec_module(harness)


def _args(**kw):
    base = dict(max_turns=10, max_tokens=100, command_timeout=60.0, token_budget=0)
    return argparse.Namespace(**(base | kw))


HISTORY = [
    {"role": "user", "content": "fix the package"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "c1", "name": "bash", "input": {"command": "ls"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "c1",
                "content": "pkmodel\n[exit 0]",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    },
]


def test_conversation_converts_to_openai_messages() -> None:
    system = [{"type": "text", "text": "be careful", "cache_control": {"type": "ephemeral"}}]
    msgs = oc.to_openai_messages(system, HISTORY)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[0]["content"] == "be careful"
    call = msgs[2]["tool_calls"][0]
    assert call["id"] == "c1" and json.loads(call["function"]["arguments"]) == {"command": "ls"}
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "pkmodel\n[exit 0]"}
    assert "cache_control" not in json.dumps(msgs)


def test_tools_convert_to_functions() -> None:
    (tool,) = oc.to_openai_tools([harness.BASH_TOOL])
    assert tool["type"] == "function" and tool["function"]["name"] == "bash"
    assert tool["function"]["parameters"]["required"] == ["command"]


def test_response_converts_to_anthropic_shape() -> None:
    data = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "Running tests.",
                    "tool_calls": [
                        {
                            "id": "x",
                            "function": {"name": "bash", "arguments": '{"command": "pytest"}'},
                        },
                        {"function": {"name": "bash", "arguments": "not json"}},
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 800},
        },
    }
    r = oc.from_openai_response(data)
    assert r.stop_reason == "tool_use"
    text, good, bad = r.content
    assert text.type == "text" and text.text == "Running tests."
    assert good.type == "tool_use" and good.id == "x" and good.input == {"command": "pytest"}
    assert bad.id == "call_1" and bad.input == {"command": oc.INVALID_ARGS}
    assert (r.usage.input_tokens, r.usage.cache_read_input_tokens, r.usage.output_tokens) == (
        200,
        800,
        50,
    )
    assert oc.from_openai_response(
        {"choices": [{"finish_reason": "stop", "message": {"content": "done"}}]}
    ).stop_reason == ("end_turn")


def test_invalid_arguments_reach_the_agent_as_an_error(tmp_path) -> None:
    out = harness.run_bash(oc.INVALID_ARGS, tmp_path, 10)
    assert "invalid JSON arguments" in out and "[exit 2]" in out


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("u", code, "err", headers, io.BytesIO(b'{"error": "x"}'))


def test_rate_limits_are_retried_honouring_retry_after(monkeypatch) -> None:
    replies = [_http_error(429, "7"), _http_error(503), _Resp(b'{"choices": []}')]

    def fake_urlopen(req, timeout):
        assert req.headers["Authorization"] == "Bearer k"
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    sleeps = []
    monkeypatch.setattr(oc.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(oc.time, "sleep", sleeps.append)
    client = oc.OpenAIChatClient(oc.PROVIDERS["groq"], "k", min_interval_s=0)
    assert client._post("/chat/completions", {}) == {"choices": []}
    assert sleeps == [7.0, 10.0]  # Retry-After first, then backoff (5 s doubled)


def test_auth_errors_are_not_retried(monkeypatch) -> None:
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise _http_error(401)

    monkeypatch.setattr(oc.urllib.request, "urlopen", fake_urlopen)
    client = oc.OpenAIChatClient(oc.PROVIDERS["groq"], "k", min_interval_s=0)
    with pytest.raises(oc.ProviderError, match="HTTP 401"):
        client._post("/chat/completions", {})
    assert len(calls) == 1


def test_missing_key_is_reported(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(oc.ProviderError, match="GROQ_API_KEY"):
        oc.OpenAIChatClient.from_env("groq")


def test_harness_presets_match_the_backend() -> None:
    assert set(harness.PROVIDER_NAMES) == set(oc.PROVIDERS)
    assert {n: p.key_env for n, p in oc.PROVIDERS.items()} == harness.KEY_ENV


def test_result_labels_are_safe_directory_names() -> None:
    assert harness.result_label("openrouter", "meta/llama-3.3:free") == (
        "openrouter-meta_llama-3.3_free"
    )
    assert harness.result_label("anthropic", "claude-sonnet-5-5") == "claude-sonnet-5-5"


class FakeProvider(oc.OpenAIChatClient):
    """Plays a model that writes the t05 reference fix, then stops; checks
    that every request it receives is a valid OpenAI conversation."""

    def __init__(self, commands):
        super().__init__(oc.PROVIDERS["groq"], "k", min_interval_s=0)
        self.commands = list(commands)
        self.requests = []

    def _post(self, path, body):
        self.requests.append(body)
        roles = [m["role"] for m in body["messages"]]
        assert roles[0] == "system" and roles[1] == "user"
        usage = {"prompt_tokens": 100, "completion_tokens": 10}
        if self.commands:
            cmd = self.commands.pop(0)
            call = {
                "id": f"c{len(self.commands)}",
                "function": {"name": "bash", "arguments": json.dumps({"command": cmd})},
            }
            msg = {"content": None, "tool_calls": [call]}
            return {"choices": [{"finish_reason": "tool_calls", "message": msg}], "usage": usage}
        return {
            "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
            "usage": usage,
        }


def test_agent_loop_through_the_openai_backend_earns_reward(tmp_path) -> None:
    work = tmp_path / "w"
    work.mkdir()
    work.chmod(0o755)
    app, instruction = harness.prepare_workspace("t05-pk-debug", work)
    client = FakeProvider(harness.oracle_commands("t05-pk-debug"))
    with (tmp_path / "t.jsonl").open("w") as log:
        stats = harness.agent_loop(client, "fake-model", instruction, app, log, _args())
    assert stats["stop"] == "end_turn" and stats["turns"] == 3
    assert stats["input_tokens"] == 300 and stats["output_tokens"] == 30
    last = client.requests[-1]["messages"]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    seeds = tmp_path / "seeds.txt"
    seeds.write_text(
        "\n".join((ROOT / "tasks/t05-pk-debug/tests/dev_seeds.txt").read_text().splitlines()[:2])
    )
    out = tmp_path / "out"
    out.mkdir()
    result = harness.grade("t05-pk-debug", app / "pkmodel", seeds, out)
    assert result == {"reward": 1, "failing_codes": []}, (out / "grade.log").read_text()


class FailingClient:
    def __init__(self):
        self.messages = self

    def create(self, **kw):
        raise oc.ProviderError("HTTP 429 from groq: daily limit")


def test_provider_errors_stop_the_attempt_and_are_excluded(tmp_path, monkeypatch) -> None:
    with (tmp_path / "t.jsonl").open("w") as log:
        stats = harness.agent_loop(FailingClient(), "m", "task", tmp_path, log, _args())
    assert stats["stop"].startswith("api_error: HTTP 429") and stats["turns"] == 1
    monkeypatch.setattr(harness, "RESULTS", tmp_path / "results")
    d = tmp_path / "results" / "groq-m" / "t05-pk-debug"
    for k, (stop, reward) in enumerate([("api_error: x", 0), ("end_turn", 1)], 1):
        (d / f"attempt-{k}").mkdir(parents=True)
        (d / f"attempt-{k}" / "summary.json").write_text(
            json.dumps(
                {
                    "model": "groq-m",
                    "task": "t05-pk-debug",
                    "stop": stop,
                    "reward": reward,
                    "failing_codes": [],
                    "turns": 3,
                    "total_tokens": 10,
                    "input_tokens": 5,
                    "output_tokens": 5,
                    "seconds": 6,
                }
            )
        )
    row = next(line for line in harness.summarise().splitlines() if "groq-m" in line)
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2] == "1" and cells[3].startswith("100%") and cells[-1] == "1"


def test_other_exceptions_are_not_swallowed(tmp_path) -> None:
    class Broken:
        def __init__(self):
            self.messages = self

        def create(self, **kw):
            raise KeyError("bug")

    with (tmp_path / "t.jsonl").open("w") as log, pytest.raises(KeyError):
        harness.agent_loop(Broken(), "m", "task", tmp_path, log, _args())


def test_real_http_round_trip_to_a_local_server(monkeypatch) -> None:
    """The transport over an actual socket: path, headers and JSON body."""
    for var in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["auth"] = self.headers["Authorization"]
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        local = oc.Provider("local", f"http://127.0.0.1:{server.server_port}/v1", "X", 0)
        client = oc.OpenAIChatClient(local, "secret", min_interval_s=0)
        r = client.messages.create(
            model="m", max_tokens=5, system="s", tools=[harness.BASH_TOOL], messages=HISTORY
        )
    finally:
        server.shutdown()
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "m" and seen["body"]["tools"][0]["function"]["name"] == "bash"
    assert r.content[0].text == "hi" and r.stop_reason == "end_turn"
