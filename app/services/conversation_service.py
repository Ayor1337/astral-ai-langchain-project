from uuid import UUID

from app.db.models import Conversation, ConversationMessage
from app.db.session import get_session_factory
from app.repositories.conversations import ConversationRepository
from app.schemas.conversation import ConversationDetail, ConversationListItem, ConversationMessageView
from app.services.exceptions import ConversationNotFoundError

DEFAULT_CONVERSATION_TITLE = "新对话"


def _to_list_item(conversation: Conversation) -> ConversationListItem:
    """将 ORM 对象收敛为列表视图模型。

    只保留列表页需要的最小字段。
    """
    return ConversationListItem(
        id=conversation.id,
        title=conversation.title,
        summary=conversation.summary,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _to_message_view(message: ConversationMessage) -> ConversationMessageView:
    """将消息实体转换为详情接口返回结构。

    把数据库字段映射为前端可直接消费的消息视图。
    """
    return ConversationMessageView(
        role=message.role,
        content=message.content,
        trace_steps=message.trace_steps,
        sequence=message.sequence,
        created_at=message.created_at,
    )


async def create_conversation(user_id: str) -> ConversationListItem:
    """创建空会话并返回列表项视图。

    用于前端在发出第一条消息前先拿到稳定的会话资源。

    Args:
        user_id: 当前用户 ID。

    Returns:
        新创建的会话列表项。
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = ConversationRepository(session)
        conversation = await repository.create_conversation(
            title=DEFAULT_CONVERSATION_TITLE,
            user_id=user_id,
        )
        await session.commit()
        return _to_list_item(conversation)


async def list_conversations(user_id: str) -> list[ConversationListItem]:
    """加载会话列表，不在服务层重复处理排序逻辑。

    排序和过滤规则都由仓储层统一负责。

    Args:
        user_id: 当前用户 ID。

    Returns:
        当前用户可见的会话列表项。
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = ConversationRepository(session)
        conversations = await repository.list_active_conversations(user_id=user_id)
        return [_to_list_item(conversation) for conversation in conversations]


async def get_conversation_detail(conversation_id: UUID, user_id: str) -> ConversationDetail:
    """返回会话元数据与按顺序排列的消息历史。

    如果会话不存在，直接抛出统一的领域错误。

    Args:
        conversation_id: 会话 ID。
        user_id: 当前用户 ID。

    Returns:
        当前用户可访问的会话详情。
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = ConversationRepository(session)
        conversation = await repository.get_conversation(conversation_id, user_id=user_id)
        if conversation is None:
            raise ConversationNotFoundError("conversation not found")
        messages = await repository.list_messages(conversation_id)
        return ConversationDetail(
            **_to_list_item(conversation).model_dump(),
            messages=[_to_message_view(message) for message in messages],
        )


async def update_conversation_title(conversation_id: UUID, user_id: str, title: str) -> ConversationListItem:
    """更新标题并返回最新列表视图。

    提交后立即返回与列表页一致的数据。

    Args:
        conversation_id: 会话 ID。
        user_id: 当前用户 ID。
        title: 新标题。

    Returns:
        更新后的会话列表项。
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = ConversationRepository(session)
        conversation = await repository.get_conversation(conversation_id, user_id=user_id)
        if conversation is None:
            raise ConversationNotFoundError("conversation not found")
        conversation = await repository.update_title(conversation, title)
        await session.commit()
        return _to_list_item(conversation)


async def delete_conversation(conversation_id: UUID, user_id: str) -> None:
    """执行软删除，让历史数据仍可保留在数据库中。

    删除后仅隐藏会话，不清理底层记录。

    Args:
        conversation_id: 会话 ID。
        user_id: 当前用户 ID。
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        repository = ConversationRepository(session)
        conversation = await repository.get_conversation(conversation_id, user_id=user_id)
        if conversation is None:
            raise ConversationNotFoundError("conversation not found")
        await repository.soft_delete(conversation)
        await session.commit()
