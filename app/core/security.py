from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import json
import os
from typing import Any
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import AuthSettings, ConfigurationError, get_settings

_bearer_scheme = HTTPBearer(auto_error=False)


class TokenDecodeError(Exception):
    """表示令牌解码或校验失败。"""


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    """表示经 JWT 验证后的当前用户。"""

    id: UUID
    username: str


def get_auth_settings() -> AuthSettings:
    """读取认证配置。

    Returns:
        已校验的认证配置。

    Raises:
        ConfigurationError: 认证未配置时抛出。
    """
    settings = get_settings()
    if settings.auth is None:
        raise ConfigurationError("JWT_SECRET_KEY is not configured")
    return settings.auth


def normalize_username(username: str) -> str:
    """归一化用户名。

    Args:
        username: 原始用户名。

    Returns:
        去首尾空格并转小写后的用户名。
    """
    return username.strip().lower()


def _base64url_encode(raw: bytes) -> str:
    """编码 JWT 片段。

    Args:
        raw: 原始字节。

    Returns:
        不带填充的 base64url 字符串。
    """
    return urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _base64url_decode(raw: str) -> bytes:
    """解码 JWT 片段。

    Args:
        raw: 不带填充的 base64url 字符串。

    Returns:
        解码后的原始字节。
    """
    padding = "=" * (-len(raw) % 4)
    return urlsafe_b64decode(f"{raw}{padding}")


def hash_password(password: str) -> str:
    """使用 `scrypt` 生成密码哈希。

    Args:
        password: 原始明文密码。

    Returns:
        可持久化的密码哈希字符串。
    """
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
    )
    return f"scrypt${_base64url_encode(salt)}${_base64url_encode(digest)}"


def verify_password(password: str, password_hash: str) -> bool:
    """校验明文密码是否匹配。

    Args:
        password: 待验证明文密码。
        password_hash: 持久化密码哈希。

    Returns:
        匹配时返回 `True`，否则返回 `False`。
    """
    try:
        algorithm, raw_salt, raw_digest = password_hash.split("$", 2)
    except ValueError:
        return False
    if algorithm != "scrypt":
        return False
    salt = _base64url_decode(raw_salt)
    expected_digest = _base64url_decode(raw_digest)
    current_digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
    )
    return hmac.compare_digest(current_digest, expected_digest)


def create_access_token(*, user_id: UUID, username: str, auth: AuthSettings) -> str:
    """生成最小 access token。

    Args:
        user_id: 用户主键。
        username: 已归一化用户名。
        auth: 认证配置。

    Returns:
        已签名 JWT。
    """
    header = {"alg": auth.jwt_algorithm, "typ": "JWT"}
    exp = datetime.now(UTC) + timedelta(seconds=auth.jwt_expire_seconds)
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": int(exp.timestamp()),
    }
    header_segment = _base64url_encode(
        json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    payload_segment = _base64url_encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    signature = hmac.new(
        auth.jwt_secret_key.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    signature_segment = _base64url_encode(signature)
    return f"{header_segment}.{payload_segment}.{signature_segment}"


def decode_access_token(token: str, auth: AuthSettings) -> AuthenticatedUser:
    """校验并解析 access token。

    Args:
        token: 原始 bearer token。
        auth: 认证配置。

    Returns:
        解析后的当前用户信息。

    Raises:
        TokenDecodeError: 令牌格式、签名或过期校验失败时抛出。
    """
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError as exc:
        raise TokenDecodeError("invalid token format") from exc

    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    expected_signature = hmac.new(
        auth.jwt_secret_key.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()
    actual_signature = _base64url_decode(signature_segment)
    if not hmac.compare_digest(expected_signature, actual_signature):
        raise TokenDecodeError("invalid token signature")

    try:
        header: dict[str, Any] = json.loads(_base64url_decode(header_segment))
        payload: dict[str, Any] = json.loads(_base64url_decode(payload_segment))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TokenDecodeError("invalid token payload") from exc

    if header.get("alg") != auth.jwt_algorithm:
        raise TokenDecodeError("invalid token algorithm")

    exp = payload.get("exp")
    if not isinstance(exp, int):
        raise TokenDecodeError("invalid token exp")
    if exp <= int(datetime.now(UTC).timestamp()):
        raise TokenDecodeError("token expired")

    subject = payload.get("sub")
    username = payload.get("username")
    if not isinstance(subject, str) or not isinstance(username, str):
        raise TokenDecodeError("invalid token subject")

    try:
        user_id = UUID(subject)
    except ValueError as exc:
        raise TokenDecodeError("invalid token subject") from exc

    return AuthenticatedUser(id=user_id, username=username)


def _unauthorized(detail: str = "unauthorized") -> HTTPException:
    """统一构造鉴权失败响应。

    Args:
        detail: 错误详情。

    Returns:
        可直接抛出的 HTTP 异常。
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthenticatedUser:
    """解析并验证当前请求的 bearer token。

    Args:
        credentials: FastAPI 解析出的认证头。

    Returns:
        已认证用户。

    Raises:
        HTTPException: 鉴权失败时抛出 401。
    """
    if credentials is None or not credentials.credentials:
        raise _unauthorized()
    try:
        auth = get_auth_settings()
        return decode_access_token(credentials.credentials, auth)
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except TokenDecodeError as exc:
        raise _unauthorized(str(exc)) from exc
