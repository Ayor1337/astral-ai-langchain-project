from datetime import datetime
import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


USERNAME_PATTERN = re.compile(r"^[a-z0-9_]{3,32}$")


def _normalize_and_validate_username(value: str) -> str:
    """归一化并校验用户名。

    Args:
        value: 原始用户名。

    Returns:
        归一化用户名。

    Raises:
        ValueError: 用户名格式非法时抛出。
    """
    normalized = value.strip().lower()
    if not USERNAME_PATTERN.fullmatch(normalized):
        raise ValueError("username must contain only lowercase letters, digits, or underscores")
    return normalized


class RegisterRequest(BaseModel):
    """注册请求体。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    username: str = Field(min_length=3, max_length=32)
    nickname: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=8)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        """校验并归一化用户名。

        Args:
            value: 原始用户名。

        Returns:
            归一化用户名。

        Raises:
            ValueError: 用户名格式非法时抛出。
        """
        return _normalize_and_validate_username(value)

    @field_validator("nickname")
    @classmethod
    def validate_nickname(cls, value: str) -> str:
        """校验昵称非空。

        Args:
            value: 原始昵称。

        Returns:
            去首尾空格后的昵称。

        Raises:
            ValueError: 昵称为空时抛出。
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("nickname must not be empty")
        return normalized


class LoginRequest(BaseModel):
    """登录请求体。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8)

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        """归一化登录用户名。

        Args:
            value: 原始用户名。

        Returns:
            归一化后的用户名。
        """
        return value.strip().lower()


class AuthUserView(BaseModel):
    """认证上下文中的用户视图。"""

    id: UUID
    username: str
    nickname: str
    created_at: datetime


class ChangeUsernameRequest(BaseModel):
    """修改用户名请求体。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    username: str = Field(min_length=3, max_length=32)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        """校验并归一化用户名。

        Args:
            value: 原始用户名。

        Returns:
            归一化后的用户名。
        """
        return _normalize_and_validate_username(value)


class TokenResponse(BaseModel):
    """登录态响应体。"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: AuthUserView
