from uuid import UUID

from app.core.config import AuthSettings
from app.core.security import create_access_token


TEST_AUTH_SETTINGS = AuthSettings(
    jwt_secret_key="test-secret-key",
    jwt_expire_seconds=604800,
    jwt_algorithm="HS256",
)


def build_auth_header(
    *,
    user_id: str = "11111111-1111-1111-1111-111111111111",
    username: str = "tester",
) -> dict[str, str]:
    token = create_access_token(
        user_id=UUID(user_id),
        username=username,
        auth=TEST_AUTH_SETTINGS,
    )
    return {"Authorization": f"Bearer {token}"}
