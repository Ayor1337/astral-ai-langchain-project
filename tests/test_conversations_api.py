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
from tests.auth_utils import build_auth_header


def _client():
    return TestClient(app)


def test_create_conversation_returns_201(monkeypatch):
    conversation_id = uuid4()

    async def fake_create_conversation(user_id):
        assert user_id == "11111111-1111-1111-1111-111111111111"
        return ConversationListItem(
            id=conversation_id,
            title="新对话",
            summary=None,
            created_at="2026-03-17T00:00:00Z",
            updated_at="2026-03-17T00:00:00Z",
        )

    monkeypatch.setattr(conversations_api, "create_conversation", fake_create_conversation)

    with _client() as client:
        response = client.post("/api/conversations", headers=build_auth_header())

    assert response.status_code == 201
    assert response.json()["id"] == str(conversation_id)
    assert response.json()["title"] == "新对话"
    assert response.json()["summary"] is None


def test_list_conversations_returns_sorted_items(monkeypatch):
    conversation_id = uuid4()

    async def fake_list_conversations(user_id):
        assert user_id == "11111111-1111-1111-1111-111111111111"
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
        response = client.get("/api/conversations", headers=build_auth_header())

    assert response.status_code == 200
    assert response.json()[0]["id"] == str(conversation_id)
    assert response.json()[0]["title"] == "测试会话"


def test_get_conversation_detail_returns_messages(monkeypatch):
    conversation_id = uuid4()

    async def fake_get_conversation_detail(target_id, user_id):
        assert target_id == conversation_id
        assert user_id == "11111111-1111-1111-1111-111111111111"
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
        response = client.get(
            f"/api/conversations/{conversation_id}",
            headers=build_auth_header(),
        )

    assert response.status_code == 200
    assert response.json()["messages"][0]["content"] == "你好"
    assert response.json()["messages"][0]["trace_steps"] is None
    assert "content_blocks" not in response.json()["messages"][0]


def test_update_conversation_title_returns_updated_resource(monkeypatch):
    conversation_id = uuid4()

    async def fake_update_conversation_title(target_id, user_id, title):
        assert target_id == conversation_id
        assert user_id == "11111111-1111-1111-1111-111111111111"
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
            headers=build_auth_header(),
        )

    assert response.status_code == 200
    assert response.json()["title"] == "新标题"


def test_delete_conversation_returns_204(monkeypatch):
    conversation_id = uuid4()

    async def fake_delete_conversation(target_id, user_id):
        assert target_id == conversation_id
        assert user_id == "11111111-1111-1111-1111-111111111111"

    monkeypatch.setattr(conversations_api, "delete_conversation", fake_delete_conversation)

    with _client() as client:
        response = client.delete(
            f"/api/conversations/{conversation_id}",
            headers=build_auth_header(),
        )

    assert response.status_code == 204
    assert response.text == ""


def test_get_conversation_detail_returns_404(monkeypatch):
    conversation_id = uuid4()

    async def fake_get_conversation_detail(target_id, user_id):
        raise ConversationNotFoundError("conversation not found")

    monkeypatch.setattr(conversations_api, "get_conversation_detail", fake_get_conversation_detail)

    with _client() as client:
        response = client.get(
            f"/api/conversations/{conversation_id}",
            headers=build_auth_header(),
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "conversation not found"}


def test_create_conversation_returns_500_when_config_missing(monkeypatch):
    async def fake_create_conversation(user_id):
        raise ConfigurationError("DATABASE_URL is not configured")

    monkeypatch.setattr(conversations_api, "create_conversation", fake_create_conversation)

    with _client() as client:
        response = client.post("/api/conversations", headers=build_auth_header())

    assert response.status_code == 500
    assert response.json() == {"detail": "DATABASE_URL is not configured"}


def test_list_conversations_returns_401_without_token() -> None:
    with _client() as client:
        response = client.get("/api/conversations")

    assert response.status_code == 401


def test_other_user_cannot_access_conversation_detail() -> None:
    username_a = f"user_{uuid4().hex[:8]}"
    username_b = f"user_{uuid4().hex[:8]}"

    with _client() as client:
        register_a = client.post(
            "/api/auth/register",
            json={"username": username_a, "nickname": "User A", "password": "password123"},
        )
        register_b = client.post(
            "/api/auth/register",
            json={"username": username_b, "nickname": "User B", "password": "password123"},
        )
        token_a = register_a.json()["access_token"]
        token_b = register_b.json()["access_token"]
        create_response = client.post(
            "/api/conversations",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        conversation_id = create_response.json()["id"]
        detail_response = client.get(
            f"/api/conversations/{conversation_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )

    assert register_a.status_code == 200
    assert register_b.status_code == 200
    assert create_response.status_code == 201
    assert detail_response.status_code == 404


def test_other_user_cannot_see_conversation_in_list() -> None:
    username_a = f"user_{uuid4().hex[:8]}"
    username_b = f"user_{uuid4().hex[:8]}"

    with _client() as client:
        register_a = client.post(
            "/api/auth/register",
            json={"username": username_a, "nickname": "User A", "password": "password123"},
        )
        register_b = client.post(
            "/api/auth/register",
            json={"username": username_b, "nickname": "User B", "password": "password123"},
        )
        token_a = register_a.json()["access_token"]
        token_b = register_b.json()["access_token"]
        create_response = client.post(
            "/api/conversations",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        list_response = client.get(
            "/api/conversations",
            headers={"Authorization": f"Bearer {token_b}"},
        )

    assert register_a.status_code == 200
    assert register_b.status_code == 200
    assert create_response.status_code == 201
    assert list_response.status_code == 200
    assert list_response.json() == []
