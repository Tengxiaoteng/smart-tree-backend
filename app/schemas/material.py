from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any
from datetime import datetime


class MaterialCreate(BaseModel):
    id: Optional[str] = None
    topicId: Optional[str] = None
    type: str = "text"
    name: str
    content: Optional[str] = None
    url: Optional[str] = None
    fileSize: Optional[int] = None
    nodeIds: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    organizedContent: Optional[str] = None
    aiSummary: Optional[str] = None
    extractedConcepts: Optional[List[str]] = None
    isOrganized: Optional[bool] = None
    structuredContent: Optional[Any] = None
    isStructured: Optional[bool] = None
    # 快速匹配字段
    contentDigest: Optional[str] = None
    keyTopics: Optional[List[str]] = None
    contentHash: Optional[str] = None
    digestGeneratedAt: Optional[datetime] = None


class MaterialUpdate(BaseModel):
    topicId: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None
    fileSize: Optional[int] = None
    nodeIds: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    organizedContent: Optional[str] = None
    aiSummary: Optional[str] = None
    extractedConcepts: Optional[List[str]] = None
    isOrganized: Optional[bool] = None
    structuredContent: Optional[Any] = None
    isStructured: Optional[bool] = None
    # 快速匹配字段
    contentDigest: Optional[str] = None
    keyTopics: Optional[List[str]] = None
    contentHash: Optional[str] = None
    digestGeneratedAt: Optional[datetime] = None


class MaterialResponse(BaseModel):
    id: str
    userId: str
    topicId: Optional[str]
    type: str
    name: str
    content: Optional[str]
    url: Optional[str]
    fileSize: Optional[int]
    nodeIds: Optional[List[str]]
    tags: Optional[List[str]]
    organizedContent: Optional[str]
    aiSummary: Optional[str]
    extractedConcepts: Optional[List[str]]
    isOrganized: Optional[bool]
    structuredContent: Optional[Any]
    isStructured: Optional[bool]
    # 快速匹配字段
    contentDigest: Optional[str] = None
    keyTopics: Optional[List[str]] = None
    contentHash: Optional[str] = None
    digestGeneratedAt: Optional[datetime] = None
    createdAt: datetime
    updatedAt: datetime

    model_config = ConfigDict(from_attributes=True)
