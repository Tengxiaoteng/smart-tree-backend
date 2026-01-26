from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime


class TopicCreate(BaseModel):
    """创建主题的请求"""
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    scope: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    # 允许前端在创建时一并写入根节点（例如“资料 -> 创建新知识树”场景）。
    # 注意：根节点可能稍后才创建，因此这里只做字段透传，不强制校验节点存在性。
    rootNodeId: Optional[str] = None


class TopicUpdate(BaseModel):
    """更新主题的请求"""
    name: Optional[str] = None
    description: Optional[str] = None
    scope: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    rootNodeId: Optional[str] = None


class TopicCloneRequest(BaseModel):
    """复制主题（知识树）的请求"""
    name: Optional[str] = None


class TopicResponse(BaseModel):
    """主题响应"""
    id: str
    userId: str
    name: str
    description: Optional[str]
    scope: Optional[List[str]]
    keywords: Optional[List[str]]
    rootNodeId: Optional[str]
    isShared: Optional[bool] = False
    sourceShareId: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime

    model_config = ConfigDict(from_attributes=True)
