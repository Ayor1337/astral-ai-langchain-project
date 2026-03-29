from fastapi import APIRouter, Depends, HTTPException

from app.core.config import ConfigurationError
from app.core.security import AuthenticatedUser, get_current_user
from app.schemas.auth import AuthUserView, LoginRequest, RegisterRequest, TokenResponse
from app.services.auth_service import get_user_profile, login_user, register_user
from app.services.exceptions import AuthenticationError, UserAlreadyExistsError, UserNotFoundError

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=TokenResponse,
    summary="注册账号",
    description="创建本地账号，并在注册成功后直接返回 JWT access token。",
    responses={
        200: {"description": "注册成功"},
        409: {"description": "用户名已存在"},
        500: {"description": "认证配置错误"},
    },
)
async def register_route(request: RegisterRequest) -> TokenResponse:
    """注册用户并返回登录态。

    Args:
        request: 注册请求体。

    Returns:
        登录态响应。
    """
    try:
        return await register_user(request)
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="账号登录",
    description="校验用户名和密码，并返回 JWT access token。",
    responses={
        200: {"description": "登录成功"},
        401: {"description": "用户名或密码错误"},
        500: {"description": "认证配置错误"},
    },
)
async def login_route(request: LoginRequest) -> TokenResponse:
    """登录并返回访问令牌。

    Args:
        request: 登录请求体。

    Returns:
        登录态响应。
    """
    try:
        return await login_user(request)
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get(
    "/me",
    response_model=AuthUserView,
    summary="当前用户",
    description="返回当前 access token 对应的用户资料。",
    responses={
        200: {"description": "查询成功"},
        401: {"description": "认证失败"},
        404: {"description": "用户不存在"},
    },
)
async def me_route(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthUserView:
    """返回当前登录用户资料。

    Args:
        current_user: 当前登录用户。

    Returns:
        当前用户视图。
    """
    try:
        return await get_user_profile(str(current_user.id))
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except UserNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
