import time
from threading import Thread

import pytest
from fastapi.testclient import TestClient

import app.api.chat as chat_api
from app.core.config import ConfigurationError
from app.llm.base import ThinkingNotSupportedError, UpstreamServiceError
from app.main import app
from app.services.exceptions import ConversationNotFoundError


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_stream_chat_returns_chunk_only_when_thinking_disabled(client, monkeypatch):
    async def fake_stream_chat_events(request):
        assert request.conversation_id is None
        assert request.message == "你好"
        assert request.thinking_enabled is False
        assert request.search_enabled is False
        yield ("conversation", {"conversation_id": "conv-1", "title": "新对话", "run_id": "run-1"})
        yield ("chunk", {"content": "你好"})
        yield ("chunk", {"content": "！"})
        yield ("done", {"status": "completed", "run_id": "run-1", "sources": []})

    monkeypatch.setattr(chat_api, "stream_chat_events", fake_stream_chat_events)

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"conversation_id": None, "message": "你好"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: conversation" in body
    assert '"conversation_id":"conv-1"' in body
    assert '"run_id":"run-1"' in body
    assert "event: chunk" in body
    assert '"content":"你好"' in body
    assert '"content":"！"' in body
    assert "event: route" not in body
    assert "event: planner_done" not in body
    assert "event: thinking_block" not in body
    assert "event: trace_step" not in body
    assert "event: trace_done" not in body
    assert "event: done" in body


def test_stream_chat_returns_conversation_title_event(client, monkeypatch):
    async def fake_stream_chat_events(request):
        yield ("conversation", {"conversation_id": "conv-1", "title": "新对话", "run_id": "run-1"})
        yield ("chunk", {"content": "你好"})
        yield ("conversation_title", {"conversation_id": "conv-1", "title": "问候对话"})
        yield ("done", {"status": "completed", "run_id": "run-1"})

    monkeypatch.setattr(chat_api, "stream_chat_events", fake_stream_chat_events)

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "你好"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: conversation_title" in body
    assert '"title":"问候对话"' in body
    assert body.index("event: conversation_title") < body.index("event: done")


def test_stream_chat_returns_trace_steps_and_chunks_when_thinking_enabled(client, monkeypatch):
    async def fake_stream_chat_events(request):
        assert request.thinking_enabled is True
        yield ("conversation", {"conversation_id": "conv-1", "title": "新对话", "run_id": "run-1"})
        yield (
            "trace_step",
            {
                "step_id": "thinking-1",
                "type": "thinking",
                "thinking": "先判断用户是在问候还是要发起任务。",
                "signature": "sig-1",
                "index": 0,
                "status": "success",
                "timestamp": "2026-03-18T12:00:00+00:00",
                "order": 1,
            },
        )
        yield (
            "trace_step",
            {
                "step_id": "search-1",
                "type": "search",
                "kind": "result_list",
                "status": "success",
                "message": "已搜索到候选结果",
                "timestamp": "2026-03-18T12:00:01+00:00",
                "order": 2,
                "payload": {"items": [{"title": "结果1", "url": "https://example.com"}]},
            },
        )
        yield ("chunk", {"content": "你"})
        yield ("chunk", {"content": "好"})
        yield ("trace_done", {"status": "completed"})
        yield ("done", {"status": "completed", "run_id": "run-1"})

    monkeypatch.setattr(chat_api, "stream_chat_events", fake_stream_chat_events)

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "你好", "thinking_enabled": True},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: conversation" in body
    assert "event: chunk" in body
    assert '"content":"你"' in body
    assert '"content":"好"' in body
    assert "event: thinking_block" not in body
    assert body.count("event: trace_step") == 2
    assert '"type":"thinking"' in body
    assert '"type":"search"' in body
    assert "event: trace_done" in body
    assert "event: done" in body


def test_stream_chat_accepts_thinking_enabled(client, monkeypatch):
    async def fake_stream_chat_events(request):
        assert request.thinking_enabled is True
        yield ("conversation", {"conversation_id": "conv-1", "title": "新对话", "run_id": "run-1"})
        yield ("done", {"status": "completed", "run_id": "run-1"})

    monkeypatch.setattr(chat_api, "stream_chat_events", fake_stream_chat_events)

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "你好", "thinking_enabled": True},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: conversation" in body
    assert "event: done" in body


def test_stream_chat_accepts_search_enabled_and_done_sources(client, monkeypatch):
    async def fake_stream_chat_events(request):
        assert request.search_enabled is True
        yield ("conversation", {"conversation_id": "conv-1", "title": "新对话", "run_id": "run-1"})
        yield ("chunk", {"content": "这里是答案[1]"})
        yield (
            "done",
            {
                "status": "completed",
                "run_id": "run-1",
                "sources": [
                    {
                        "index": 1,
                        "title": "Astral AI",
                        "url": "https://example.com/astral",
                        "snippet": "Latest update",
                    }
                ],
            },
        )

    monkeypatch.setattr(chat_api, "stream_chat_events", fake_stream_chat_events)

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "Astral AI 最新消息", "search_enabled": True},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"search_enabled":true' not in body
    assert '"sources":[{"index":1,"title":"Astral AI","url":"https://example.com/astral","snippet":"Latest update"}]' in body


def test_stream_chat_returns_stopped_done_status(client, monkeypatch):
    async def fake_stream_chat_events(request):
        yield ("conversation", {"conversation_id": "conv-1", "title": "新对话", "run_id": "run-1"})
        yield ("chunk", {"content": "部分回答"})
        yield ("done", {"status": "stopped", "run_id": "run-1"})

    monkeypatch.setattr(chat_api, "stream_chat_events", fake_stream_chat_events)

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "你好"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"status":"stopped"' in body
    assert '"run_id":"run-1"' in body


def test_stream_chat_emits_first_event_before_full_stream_finishes(client, monkeypatch):
    async def fake_stream_chat_events(request):
        yield ("conversation", {"conversation_id": "conv-1", "title": "新对话", "run_id": "run-1"})
        await __import__("asyncio").sleep(0.2)
        yield ("chunk", {"content": "你好"})
        await __import__("asyncio").sleep(0.2)
        yield ("done", {"status": "completed", "run_id": "run-1"})

    monkeypatch.setattr(chat_api, "stream_chat_events", fake_stream_chat_events)

    observed: dict[str, object] = {}

    def consume_stream() -> None:
        with client.stream("POST", "/api/chat/stream", json={"message": "你好"}) as response:
            observed["status_code"] = response.status_code
            start = time.perf_counter()
            lines = response.iter_lines()
            observed["first_line"] = next(lines)
            observed["first_line_elapsed"] = time.perf_counter() - start
            observed["rest"] = list(lines)

    worker = Thread(target=consume_stream)
    start = time.perf_counter()
    worker.start()
    worker.join(timeout=0.15)

    assert worker.is_alive(), "流消费线程应仍在等待后续事件，不能在 0.15s 内一次性完成"

    worker.join(timeout=2)

    assert worker.is_alive() is False
    assert observed["status_code"] == 200
    assert observed["first_line_elapsed"] < 0.15
    assert observed["first_line"] == "event: conversation"
    assert any(line == "event: chunk" for line in observed["rest"])
    assert any(line == "event: done" for line in observed["rest"])
    assert time.perf_counter() - start >= 0.35


def test_stop_chat_run_returns_202(client, monkeypatch):
    async def fake_request_stop_chat_run(run_id):
        assert str(run_id) == "8bc85d87-ea36-46de-aeeb-d26c17e57ef3"
        return {"run_id": str(run_id), "status": "stop_requested"}

    monkeypatch.setattr(chat_api, "request_stop_chat_run", fake_request_stop_chat_run)

    response = client.post("/api/chat/runs/8bc85d87-ea36-46de-aeeb-d26c17e57ef3/stop")

    assert response.status_code == 202
    assert response.json() == {
        "run_id": "8bc85d87-ea36-46de-aeeb-d26c17e57ef3",
        "status": "stop_requested",
    }


def test_stop_chat_run_returns_404(client, monkeypatch):
    async def fake_request_stop_chat_run(run_id):
        raise chat_api.ChatRunNotFoundError("chat run not found")

    monkeypatch.setattr(chat_api, "request_stop_chat_run", fake_request_stop_chat_run)

    response = client.post("/api/chat/runs/8bc85d87-ea36-46de-aeeb-d26c17e57ef3/stop")

    assert response.status_code == 404
    assert response.json() == {"detail": "chat run not found"}


@pytest.mark.parametrize(
    ("payload", "field_name"),
    [
        ({"message": ""}, "message"),
        ({"conversation_id": "not-a-uuid", "message": "hi"}, "conversation_id"),
    ],
)
def test_stream_chat_validates_request_body(client, payload, field_name):
    response = client.post("/api/chat/stream", json=payload)

    assert response.status_code == 422
    assert field_name in response.text


def test_stream_chat_returns_500_when_config_missing(client, monkeypatch):
    async def fake_stream_chat_events(request):
        raise ConfigurationError("DATABASE_URL is not configured")
        yield

    monkeypatch.setattr(chat_api, "stream_chat_events", fake_stream_chat_events)

    response = client.post("/api/chat/stream", json={"message": "你好"})

    assert response.status_code == 500
    assert response.json() == {"detail": "DATABASE_URL is not configured"}


def test_stream_chat_returns_404_when_conversation_missing(client, monkeypatch):
    async def fake_stream_chat_events(request):
        raise ConversationNotFoundError("conversation not found")
        yield

    monkeypatch.setattr(chat_api, "stream_chat_events", fake_stream_chat_events)

    response = client.post(
        "/api/chat/stream",
        json={"conversation_id": "8bc85d87-ea36-46de-aeeb-d26c17e57ef3", "message": "你好"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "conversation not found"}


def test_stream_chat_returns_502_when_upstream_fails_before_stream(client, monkeypatch):
    async def fake_stream_chat_events(request):
        raise UpstreamServiceError("model upstream failed")
        yield

    monkeypatch.setattr(chat_api, "stream_chat_events", fake_stream_chat_events)

    response = client.post("/api/chat/stream", json={"message": "你好"})

    assert response.status_code == 502
    assert response.json() == {"detail": "model upstream failed"}


def test_stream_chat_returns_400_when_thinking_not_supported(client, monkeypatch):
    async def fake_stream_chat_events(request):
        raise ThinkingNotSupportedError("provider openai does not support thinking")
        yield

    monkeypatch.setattr(chat_api, "stream_chat_events", fake_stream_chat_events)

    response = client.post("/api/chat/stream", json={"message": "你好", "thinking_enabled": True})

    assert response.status_code == 400
    assert response.json() == {"detail": "provider openai does not support thinking"}


def test_cors_preflight_returns_allow_origin_header(client):
    response = client.options(
        "/api/chat/stream",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
