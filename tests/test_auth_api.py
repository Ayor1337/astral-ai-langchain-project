from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def _client() -> TestClient:
    return TestClient(app)


def test_register_returns_token_and_user() -> None:
    username = f"user_{uuid4().hex[:8]}"
    with _client() as client:
        response = client.post(
            "/api/auth/register",
            json={
                "username": username,
                "nickname": "Alice",
                "password": "password123",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["expires_in"] == 604800
    assert payload["access_token"]
    assert payload["user"]["username"] == username
    assert payload["user"]["nickname"] == "Alice"


def test_login_returns_token_and_user() -> None:
    username = f"user_{uuid4().hex[:8]}"
    with _client() as client:
        client.post(
            "/api/auth/register",
            json={
                "username": username,
                "nickname": "Alice",
                "password": "password123",
            },
        )
        response = client.post(
            "/api/auth/login",
            json={
                "username": username,
                "password": "password123",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["expires_in"] == 604800
    assert payload["access_token"]
    assert payload["user"]["username"] == username


def test_me_returns_401_without_token() -> None:
    with _client() as client:
        response = client.get("/api/auth/me")

    assert response.status_code == 401


def test_register_returns_409_for_duplicate_username() -> None:
    username = f"user_{uuid4().hex[:8]}"
    with _client() as client:
        first_response = client.post(
            "/api/auth/register",
            json={
                "username": username,
                "nickname": "Alice",
                "password": "password123",
            },
        )
        second_response = client.post(
            "/api/auth/register",
            json={
                "username": username.upper(),
                "nickname": "Alice 2",
                "password": "password123",
            },
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 409
    assert second_response.json() == {"detail": "username already exists"}


def test_login_returns_401_for_invalid_password() -> None:
    username = f"user_{uuid4().hex[:8]}"
    with _client() as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "username": username,
                "nickname": "Alice",
                "password": "password123",
            },
        )
        response = client.post(
            "/api/auth/login",
            json={
                "username": username,
                "password": "password124",
            },
        )

    assert register_response.status_code == 200
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid username or password"}


def test_me_returns_current_user_with_token() -> None:
    username = f"user_{uuid4().hex[:8]}"
    with _client() as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "username": username,
                "nickname": "Alice",
                "password": "password123",
            },
        )
        token = register_response.json()["access_token"]
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert register_response.status_code == 200
    assert response.status_code == 200
    assert response.json()["username"] == username
    assert response.json()["nickname"] == "Alice"


def test_change_username_returns_token_and_updated_user() -> None:
    old_username = f"user_{uuid4().hex[:8]}"
    new_username = f"user_{uuid4().hex[:8]}"

    with _client() as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "username": old_username,
                "nickname": "Alice",
                "password": "password123",
            },
        )
        old_token = register_response.json()["access_token"]
        response = client.post(
            "/api/auth/change-username",
            json={"username": new_username},
            headers={"Authorization": f"Bearer {old_token}"},
        )

    assert register_response.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    assert payload["access_token"]
    assert payload["access_token"] != old_token
    assert payload["user"]["username"] == new_username
    assert payload["user"]["nickname"] == "Alice"


def test_change_username_returns_401_without_token() -> None:
    with _client() as client:
        response = client.post(
            "/api/auth/change-username",
            json={"username": "new_user_01"},
        )

    assert response.status_code == 401


def test_change_username_returns_409_when_username_exists() -> None:
    username_a = f"user_{uuid4().hex[:8]}"
    username_b = f"user_{uuid4().hex[:8]}"

    with _client() as client:
        register_a = client.post(
            "/api/auth/register",
            json={
                "username": username_a,
                "nickname": "Alice",
                "password": "password123",
            },
        )
        register_b = client.post(
            "/api/auth/register",
            json={
                "username": username_b,
                "nickname": "Bob",
                "password": "password123",
            },
        )
        token_a = register_a.json()["access_token"]
        response = client.post(
            "/api/auth/change-username",
            json={"username": username_b},
            headers={"Authorization": f"Bearer {token_a}"},
        )

    assert register_a.status_code == 200
    assert register_b.status_code == 200
    assert response.status_code == 409
    assert response.json() == {"detail": "username already exists"}


def test_change_username_returns_422_for_invalid_username() -> None:
    username = f"user_{uuid4().hex[:8]}"

    with _client() as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "username": username,
                "nickname": "Alice",
                "password": "password123",
            },
        )
        token = register_response.json()["access_token"]
        response = client.post(
            "/api/auth/change-username",
            json={"username": "Invalid-Name"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert register_response.status_code == 200
    assert response.status_code == 422


def test_change_username_returns_200_for_same_normalized_username() -> None:
    username = f"user_{uuid4().hex[:8]}"

    with _client() as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "username": username,
                "nickname": "Alice",
                "password": "password123",
            },
        )
        token = register_response.json()["access_token"]
        response = client.post(
            "/api/auth/change-username",
            json={"username": f"  {username.upper()}  "},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert register_response.status_code == 200
    assert response.status_code == 200
    assert response.json()["user"]["username"] == username


def test_change_username_allows_login_with_new_username_only() -> None:
    old_username = f"user_{uuid4().hex[:8]}"
    new_username = f"user_{uuid4().hex[:8]}"

    with _client() as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "username": old_username,
                "nickname": "Alice",
                "password": "password123",
            },
        )
        token = register_response.json()["access_token"]
        change_response = client.post(
            "/api/auth/change-username",
            json={"username": new_username},
            headers={"Authorization": f"Bearer {token}"},
        )
        old_login_response = client.post(
            "/api/auth/login",
            json={
                "username": old_username,
                "password": "password123",
            },
        )
        new_login_response = client.post(
            "/api/auth/login",
            json={
                "username": new_username,
                "password": "password123",
            },
        )

    assert register_response.status_code == 200
    assert change_response.status_code == 200
    assert old_login_response.status_code == 401
    assert new_login_response.status_code == 200
    assert new_login_response.json()["user"]["username"] == new_username


def test_change_username_keeps_old_and_new_tokens_usable() -> None:
    old_username = f"user_{uuid4().hex[:8]}"
    new_username = f"user_{uuid4().hex[:8]}"

    with _client() as client:
        register_response = client.post(
            "/api/auth/register",
            json={
                "username": old_username,
                "nickname": "Alice",
                "password": "password123",
            },
        )
        old_token = register_response.json()["access_token"]
        change_response = client.post(
            "/api/auth/change-username",
            json={"username": new_username},
            headers={"Authorization": f"Bearer {old_token}"},
        )
        new_token = change_response.json()["access_token"]
        old_token_me_response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {old_token}"},
        )
        new_token_me_response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {new_token}"},
        )

    assert register_response.status_code == 200
    assert change_response.status_code == 200
    assert old_token_me_response.status_code == 200
    assert new_token_me_response.status_code == 200
    assert old_token_me_response.json()["username"] == new_username
    assert new_token_me_response.json()["username"] == new_username
