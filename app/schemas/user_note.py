from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class UserNoteCreate(BaseModel):
    id: Optional[str] = None
    nodeId: str
    questionId: Optional[str] = None
    content: str
    source: Optional[str] = "manual"
    tags: Optional[List[str]] = None
    createdAt: Optional[datetime] = None


class UserNoteUpdate(BaseModel):
    questionId: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[List[str]] = None


class UserNoteResponse(BaseModel):
    id: str
    userId: str
    nodeId: str
    questionId: Optional[str]
    content: str
    source: str
    tags: Optional[List[str]]
    createdAt: datetime
    updatedAt: datetime

    model_config = ConfigDict(from_attributes=True)
