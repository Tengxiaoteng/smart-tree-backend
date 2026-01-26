from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator


def _map_difficulty_to_client(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value in {"easy", "medium", "hard"}:
        return value
    return {
        "beginner": "easy",
        "intermediate": "medium",
        "advanced": "hard",
    }.get(value, value)


def _parse_jsonish_text(raw: Any) -> Any:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    trimmed = raw.strip()
    if not trimmed:
        return ""
    if trimmed.lower() in {"null", "none"}:
        return None
    if (trimmed.startswith("[") and trimmed.endswith("]")) or (trimmed.startswith('"') and trimmed.endswith('"')):
        try:
            import json

            parsed = json.loads(trimmed)
            if isinstance(parsed, (list, str)):
                return parsed
        except Exception:
            return raw
    return raw


class QuestionCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    nodeId: str
    type: str
    content: str
    options: Optional[List[str]] = None
    answer: Optional[Any] = None
    explanation: Optional[str] = None
    difficulty: Optional[str] = None
    targetGoalIds: Optional[List[str]] = None
    targetGoalNames: Optional[List[str]] = None
    isFavorite: Optional[bool] = None
    hints: Optional[Any] = None
    relatedConcepts: Optional[Any] = None
    userNotes: Optional[str] = None
    tags: Optional[List[str]] = None

    # 题目来源
    source: Optional[str] = None
    sourceMaterialIds: Optional[List[str]] = None
    sourceMaterialNames: Optional[List[str]] = None
    sourceContext: Optional[str] = None
    difficultyReason: Optional[str] = None

    # 衍生题关联
    derivedFromQuestionId: Optional[str] = None
    derivedFromRecordId: Optional[str] = None
    isDerivedQuestion: Optional[bool] = None
    parentQuestionId: Optional[str] = None
    derivationType: Optional[str] = None

    createdAt: Optional[datetime] = None


class QuestionUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Optional[str] = None
    content: Optional[str] = None
    options: Optional[List[str]] = None
    answer: Optional[Any] = None
    explanation: Optional[str] = None
    difficulty: Optional[str] = None
    isFavorite: Optional[bool] = None
    targetGoalIds: Optional[List[str]] = None
    parentQuestionId: Optional[str] = None
    derivationType: Optional[str] = None
    hints: Optional[Any] = None
    relatedConcepts: Optional[Any] = None

    # 兼容前端字段
    derivedFromQuestionId: Optional[str] = None
    isDerivedQuestion: Optional[bool] = None


class QuestionResponse(BaseModel):
    id: str
    userId: str
    nodeId: str
    type: str
    difficulty: Optional[str]
    content: str
    options: Optional[List[str]]
    answer: Optional[Any]
    explanation: Optional[str]
    hints: Optional[Any] = None
    relatedConcepts: Optional[Any] = None
    targetGoalIds: Optional[List[str]] = None
    targetGoalNames: Optional[List[str]] = None
    isFavorite: bool
    userNotes: Optional[str] = None
    tags: Optional[List[str]] = None

    # 题目来源
    source: Optional[str] = None
    sourceMaterialIds: Optional[List[str]] = None
    sourceMaterialNames: Optional[List[str]] = None
    sourceContext: Optional[str] = None
    difficultyReason: Optional[str] = None

    # 衍生题关联
    derivedFromQuestionId: Optional[str] = None
    derivedFromRecordId: Optional[str] = None
    isDerivedQuestion: Optional[bool] = None
    parentQuestionId: Optional[str] = None
    derivationType: Optional[str] = None

    createdAt: datetime
    updatedAt: datetime

    @field_validator("difficulty", mode="before")
    @classmethod
    def _normalize_difficulty(cls, value: Any):
        return _map_difficulty_to_client(value if value is None or isinstance(value, str) else str(value))

    @field_validator("answer", mode="before")
    @classmethod
    def _normalize_answer(cls, value: Any):
        return _parse_jsonish_text(value)

    @field_validator("hints", mode="before")
    @classmethod
    def _normalize_hints(cls, value: Any):
        return _parse_jsonish_text(value)

    @field_validator("relatedConcepts", mode="before")
    @classmethod
    def _normalize_related_concepts(cls, value: Any):
        return _parse_jsonish_text(value)

    model_config = ConfigDict(from_attributes=True)
