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


def test_update_profile_returns_updated_user() -> None:
    username = f"user_{uuid4().hex[:8]}"
    new_nickname = "Alice Cooper"

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
            "/api/auth/profile",
            json={"nickname": new_nickname},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert register_response.status_code == 200
    assert response.status_code == 200
    payload = response.json()
    assert payload["username"] == username
    assert payload["nickname"] == new_nickname


def test_update_profile_returns_401_without_token() -> None:
    with _client() as client:
        response = client.post(
            "/api/auth/profile",
            json={"nickname": "Alice Cooper"},
        )

    assert response.status_code == 401


def test_update_profile_returns_422_for_invalid_nickname() -> None:
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
            "/api/auth/profile",
            json={"nickname": "   "},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert register_response.status_code == 200
    assert response.status_code == 422


def test_update_profile_returns_422_for_extra_fields() -> None:
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
            "/api/auth/profile",
            json={"nickname": "Alice Cooper", "username": "other_name"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert register_response.status_code == 200
    assert response.status_code == 422


def test_update_profile_returns_200_for_same_normalized_nickname() -> None:
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
            "/api/auth/profile",
            json={"nickname": "  Alice  "},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert register_response.status_code == 200
    assert response.status_code == 200
    assert response.json()["nickname"] == "Alice"


def test_update_profile_updates_me_response() -> None:
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
        update_response = client.post(
            "/api/auth/profile",
            json={"nickname": "Alice Cooper"},
            headers={"Authorization": f"Bearer {token}"},
        )
        me_response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert register_response.status_code == 200
    assert update_response.status_code == 200
    assert me_response.status_code == 200
    assert me_response.json()["nickname"] == "Alice Cooper"


def test_update_profile_keeps_existing_token_usable() -> None:
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
        update_response = client.post(
            "/api/auth/profile",
            json={"nickname": "Alice Cooper"},
            headers={"Authorization": f"Bearer {token}"},
        )
        me_response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert register_response.status_code == 200
    assert update_response.status_code == 200
    assert me_response.status_code == 200
    assert me_response.json()["nickname"] == "Alice Cooper"
