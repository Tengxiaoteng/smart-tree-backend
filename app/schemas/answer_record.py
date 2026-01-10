from datetime import datetime
from typing import Any, Optional, List

from pydantic import BaseModel, ConfigDict, field_validator


def _parse_jsonish_text(raw: Any) -> Any:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    trimmed = raw.strip()
    if not trimmed:
        return ""
    if (trimmed.startswith("[") and trimmed.endswith("]")) or (trimmed.startswith('"') and trimmed.endswith('"')):
        try:
            import json

            parsed = json.loads(trimmed)
            if isinstance(parsed, (list, str)):
                return parsed
        except Exception:
            return raw
    return raw


# 答题追问对话消息
class QuestionChatMessageSchema(BaseModel):
    id: str
    role: str  # "user" | "assistant"
    content: str
    timestamp: Optional[datetime] = None


class AnswerRecordCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    questionId: str
    nodeId: str
    userAnswer: Optional[Any] = None
    isCorrect: bool
    timeSpent: Optional[int] = None
    followUpMessages: Optional[List[QuestionChatMessageSchema]] = None
    createdAt: Optional[datetime] = None


class AnswerRecordUpdate(BaseModel):
    """更新答题记录（主要用于追加追问对话）"""
    model_config = ConfigDict(extra="ignore")

    followUpMessages: Optional[List[QuestionChatMessageSchema]] = None


class AnswerRecordResponse(BaseModel):
    id: str
    userId: str
    questionId: str
    nodeId: str
    userAnswer: Optional[Any]
    isCorrect: bool
    timeSpent: Optional[int]
    followUpMessages: Optional[List[QuestionChatMessageSchema]] = None
    createdAt: datetime

    @field_validator("userAnswer", mode="before")
    @classmethod
    def _normalize_user_answer(cls, value: Any):
        return _parse_jsonish_text(value)

    model_config = ConfigDict(from_attributes=True)
