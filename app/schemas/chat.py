from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    conversation_id: UUID | None = None
    message: str = Field(min_length=1)
    thinking_enabled: bool = False
    search_enabled: bool = False
    user_id: str = ""


class ChatRunStopResponse(BaseModel):
    run_id: UUID
    status: Literal["stop_requested"]
