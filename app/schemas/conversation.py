from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ConversationMessageView(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    trace_steps: list[dict[str, object]] | None = None
    sequence: int
    created_at: datetime


class ConversationListItem(BaseModel):
    id: UUID
    title: str
    summary: str | None
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationListItem):
    messages: list[ConversationMessageView]


class ConversationUpdateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1)
