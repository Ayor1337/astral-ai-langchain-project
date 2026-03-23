from uuid import uuid4

from fastapi.testclient import TestClient

import app.api.conversations as conversations_api
from app.main import app
from app.core.config import ConfigurationError
from app.schemas.conversation import (
    ConversationDetail,
    ConversationListItem,
    ConversationMessageView,
)
from app.services.exceptions import ConversationNotFoundError


def _client():
    return TestClient(app)


def test_create_conversation_returns_201(monkeypatch):
    conversation_id = uuid4()

    async def fake_create_conversation():
        return ConversationListItem(
            id=conversation_id,
            title="新对话",
            summary=None,
            created_at="2026-03-17T00:00:00Z",
            updated_at="2026-03-17T00:00:00Z",
        )

    monkeypatch.setattr(conversations_api, "create_conversation", fake_create_conversation)

    with _client() as client:
        response = client.post("/api/conversations")

    assert response.status_code == 201
    assert response.json()["id"] == str(conversation_id)
    assert response.json()["title"] == "新对话"
    assert response.json()["summary"] is None


def test_list_conversations_returns_sorted_items(monkeypatch):
    conversation_id = uuid4()

    async def fake_list_conversations():
        return [
            ConversationListItem(
                id=conversation_id,
                title="测试会话",
                summary="摘要",
                created_at="2026-03-17T00:00:00Z",
                updated_at="2026-03-17T01:00:00Z",
            )
        ]

    monkeypatch.setattr(conversations_api, "list_conversations", fake_list_conversations)

    with _client() as client:
        response = client.get("/api/conversations")

    assert response.status_code == 200
    assert response.json()[0]["id"] == str(conversation_id)
    assert response.json()[0]["title"] == "测试会话"


def test_get_conversation_detail_returns_messages(monkeypatch):
    conversation_id = uuid4()

    async def fake_get_conversation_detail(target_id):
        assert target_id == conversation_id
        return ConversationDetail(
            id=conversation_id,
            title="测试会话",
            summary="摘要",
            created_at="2026-03-17T00:00:00Z",
            updated_at="2026-03-17T01:00:00Z",
            messages=[
                ConversationMessageView(
                    role="user",
                    content="你好",
                    sequence=1,
                    trace_steps=None,
                    created_at="2026-03-17T00:00:00Z",
                )
            ],
        )

    monkeypatch.setattr(conversations_api, "get_conversation_detail", fake_get_conversation_detail)

    with _client() as client:
        response = client.get(f"/api/conversations/{conversation_id}")

    assert response.status_code == 200
    assert response.json()["messages"][0]["content"] == "你好"
    assert response.json()["messages"][0]["trace_steps"] is None
    assert "content_blocks" not in response.json()["messages"][0]


def test_update_conversation_title_returns_updated_resource(monkeypatch):
    conversation_id = uuid4()

    async def fake_update_conversation_title(target_id, title):
        assert target_id == conversation_id
        assert title == "新标题"
        return ConversationListItem(
            id=conversation_id,
            title=title,
            summary=None,
            created_at="2026-03-17T00:00:00Z",
            updated_at="2026-03-17T01:00:00Z",
        )

    monkeypatch.setattr(
        conversations_api,
        "update_conversation_title",
        fake_update_conversation_title,
    )

    with _client() as client:
        response = client.patch(
            f"/api/conversations/{conversation_id}",
            json={"title": "新标题"},
        )

    assert response.status_code == 200
    assert response.json()["title"] == "新标题"


def test_delete_conversation_returns_204(monkeypatch):
    conversation_id = uuid4()

    async def fake_delete_conversation(target_id):
        assert target_id == conversation_id

    monkeypatch.setattr(conversations_api, "delete_conversation", fake_delete_conversation)

    with _client() as client:
        response = client.delete(f"/api/conversations/{conversation_id}")

    assert response.status_code == 204
    assert response.text == ""


def test_get_conversation_detail_returns_404(monkeypatch):
    conversation_id = uuid4()

    async def fake_get_conversation_detail(target_id):
        raise ConversationNotFoundError("conversation not found")

    monkeypatch.setattr(conversations_api, "get_conversation_detail", fake_get_conversation_detail)

    with _client() as client:
        response = client.get(f"/api/conversations/{conversation_id}")

    assert response.status_code == 404
    assert response.json() == {"detail": "conversation not found"}


def test_create_conversation_returns_500_when_config_missing(monkeypatch):
    async def fake_create_conversation():
        raise ConfigurationError("DATABASE_URL is not configured")

    monkeypatch.setattr(conversations_api, "create_conversation", fake_create_conversation)

    with _client() as client:
        response = client.post("/api/conversations")

    assert response.status_code == 500
    assert response.json() == {"detail": "DATABASE_URL is not configured"}
