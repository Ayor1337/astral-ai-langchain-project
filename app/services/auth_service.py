from __future__ import annotations

from app.core.security import create_access_token, get_auth_settings, hash_password, verify_password
from app.db.models import User
from app.db.session import get_session_factory
from app.repositories.users import UserRepository
from app.schemas.auth import (
    AuthUserView,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
)
from app.services.exceptions import AuthenticationError, UserAlreadyExistsError, UserNotFoundError


def _to_user_view(user: User) -> AuthUserView:
    """将用户实体映射为对外视图。

    Args:
        user: 用户实体。

    Returns:
        可序列化用户视图。
    """
    return AuthUserView(
        id=user.id,
        username=user.username,
        nickname=user.nickname,
        created_at=user.created_at,
    )


async def register_user(request: RegisterRequest) -> TokenResponse:
    """注册用户并直接签发 access token。

    Args:
        request: 注册请求。

    Returns:
        登录态响应。

    Raises:
        UserAlreadyExistsError: 用户名已存在时抛出。
    """
    auth = get_auth_settings()
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = UserRepository(session)
        if await repository.get_by_username(request.username) is not None:
            raise UserAlreadyExistsError("username already exists")
        user = await repository.create_user(
            username=request.username,
            nickname=request.nickname,
            password_hash=hash_password(request.password),
        )
        await session.commit()
    return TokenResponse(
        access_token=create_access_token(user_id=user.id, username=user.username, auth=auth),
        expires_in=auth.jwt_expire_seconds,
        user=_to_user_view(user),
    )


async def login_user(request: LoginRequest) -> TokenResponse:
    """校验用户名与密码并签发 access token。

    Args:
        request: 登录请求。

    Returns:
        登录态响应。

    Raises:
        AuthenticationError: 用户名或密码错误时抛出。
    """
    auth = get_auth_settings()
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = UserRepository(session)
        user = await repository.get_by_username(request.username)
        if user is None or not verify_password(request.password, user.password_hash):
            raise AuthenticationError("invalid username or password")
    return TokenResponse(
        access_token=create_access_token(user_id=user.id, username=user.username, auth=auth),
        expires_in=auth.jwt_expire_seconds,
        user=_to_user_view(user),
    )


async def get_user_profile(user_id: str) -> AuthUserView:
    """读取当前登录用户资料。

    Args:
        user_id: 当前用户 ID。

    Returns:
        当前用户视图。

    Raises:
        UserNotFoundError: 用户不存在时抛出。
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = UserRepository(session)
        user = await repository.get_by_id(user_id)
        if user is None:
            raise UserNotFoundError("user not found")
        return _to_user_view(user)


async def update_profile(user_id: str, request: UpdateProfileRequest) -> AuthUserView:
    """更新当前用户资料并返回最新用户视图。

    Args:
        user_id: 当前用户 ID。
        request: 更新资料请求。

    Returns:
        最新用户视图。

    Raises:
        UserNotFoundError: 当前用户不存在时抛出。
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = UserRepository(session)
        user = await repository.get_by_id(user_id)
        if user is None:
            raise UserNotFoundError("user not found")

        if request.nickname != user.nickname:
            user = await repository.update_nickname(user, request.nickname)

        await session.commit()

    return _to_user_view(user)
