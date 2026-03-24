from uuid import UUID

from fastapi import APIRouter, HTTPException, Response

from app.core.config import ConfigurationError
from app.schemas.conversation import ConversationDetail, ConversationListItem, ConversationUpdateRequest
from app.services.conversation_service import (
    create_conversation,
    delete_conversation,
    get_conversation_detail,
    list_conversations,
    update_conversation_title,
)
from app.services.exceptions import ConversationNotFoundError

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.post(
    "",
    response_model=ConversationListItem,
    status_code=201,
    summary="新建空会话",
    description="显式创建一个空会话，返回会话元数据。推荐前端在点击“新建对话”时先调用此接口，再使用返回的 conversation_id 发起聊天。",
    responses={
        201: {"description": "会话创建成功"},
        500: {"description": "服务配置错误"},
    },
)
async def create_conversation_route() -> ConversationListItem:
    """显式创建空会话，便于前端先拿到稳定的 conversation_id。"""
    try:
        return await create_conversation()
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "",
    response_model=list[ConversationListItem],
    summary="获取会话列表",
    description="返回所有未软删除的会话，按 updated_at 倒序排列。",
)
async def list_conversations_route() -> list[ConversationListItem]:
    """返回所有未删除会话，具体排序由服务层和仓储层统一维护。"""
    try:
        return await list_conversations()
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
    summary="获取会话详情",
    description="返回单个会话的元数据、摘要和完整消息历史。新建空会话时 messages 为空数组。",
)
async def get_conversation_detail_route(conversation_id: UUID) -> ConversationDetail:
    """加载单个会话及其完整消息历史。"""
    try:
        return await get_conversation_detail(conversation_id)
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch(
    "/{conversation_id}",
    response_model=ConversationListItem,
    summary="更新会话标题",
    description="仅更新会话标题，不影响摘要和消息历史。",
)
async def update_conversation_route(
    conversation_id: UUID,
    request: ConversationUpdateRequest,
) -> ConversationListItem:
    """只更新会话标题，不触碰摘要和消息。"""
    try:
        return await update_conversation_title(conversation_id, request.title)
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete(
    "/{conversation_id}",
    status_code=204,
    summary="软删除会话",
    description="将会话标记为已删除。删除后的会话不会出现在列表中，详情与聊天访问将返回 404。",
)
async def delete_conversation_route(conversation_id: UUID) -> Response:
    """执行软删除，让列表和详情接口统一感知会话已不可见。"""
    try:
        await delete_conversation(conversation_id)
    except ConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)
