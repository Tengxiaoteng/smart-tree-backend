"""
分享功能相关的 Pydantic Schemas
"""
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============ TreeShare Schemas ============

class TreeShareCreate(BaseModel):
    """创建分享的请求"""
    topicId: str = Field(..., description="要分享的知识树ID")
    shareType: str = Field(default="public", description="分享类型: public/private")
    sharePassword: Optional[str] = Field(None, description="私密分享的密码")
    shareTitle: Optional[str] = Field(None, description="分享标题（默认使用知识树名称）")
    shareDescription: Optional[str] = Field(None, description="分享描述")
    allowCopy: bool = Field(default=True, description="是否允许复制")


class TreeShareUpdate(BaseModel):
    """更新分享的请求"""
    shareType: Optional[str] = None
    sharePassword: Optional[str] = None
    shareTitle: Optional[str] = None
    shareDescription: Optional[str] = None
    isActive: Optional[bool] = None


class TreeShareResponse(BaseModel):
    """分享记录响应"""
    id: str
    topicId: str
    ownerId: str
    shareCode: str
    shareType: str
    shareTitle: Optional[str]
    shareDescription: Optional[str]
    isActive: bool
    currentVersion: int
    subscriberCount: int
    copyCount: int
    viewCount: int
    allowCopy: bool
    createdAt: datetime
    updatedAt: datetime
    
    # 额外信息（可选）
    topicName: Optional[str] = None
    ownerName: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class TreeSharePublicInfo(BaseModel):
    """公开的分享信息（用于分享码查询）"""
    id: str
    shareCode: str
    shareType: str
    shareTitle: Optional[str]
    shareDescription: Optional[str]
    currentVersion: int
    subscriberCount: int
    copyCount: int
    allowCopy: bool
    createdAt: datetime
    
    # 知识树基本信息
    topicName: str
    topicDescription: Optional[str]
    nodeCount: int
    materialCount: int
    
    # 作者信息
    ownerName: str
    
    # 是否需要密码
    requiresPassword: bool


# ============ TreeSubscription Schemas ============

class TreeSubscriptionCreate(BaseModel):
    """创建订阅的请求"""
    shareCode: Optional[str] = Field(None, description="分享码（可选，URL中已包含）")
    sharePassword: Optional[str] = Field(None, description="私密分享的密码")
    newTopicName: Optional[str] = Field(None, description="新知识树的名称（可选）")


class TreeSubscriptionResponse(BaseModel):
    """订阅记录响应"""
    id: str
    shareId: str
    subscriberId: str
    localTopicId: str
    syncedVersion: int
    lastSyncedAt: datetime
    autoSync: bool
    notifyOnUpdate: bool
    createdAt: datetime
    updatedAt: datetime
    
    # 额外信息
    shareCode: Optional[str] = None
    originalTopicName: Optional[str] = None
    localTopicName: Optional[str] = None
    currentVersion: Optional[int] = None
    hasPendingUpdates: Optional[bool] = None

    model_config = ConfigDict(from_attributes=True)


# ============ TreeVersion Schemas ============

class TreeVersionPublish(BaseModel):
    """发布新版本的请求"""
    changeLog: Optional[str] = Field(None, description="变更说明")


class TreeVersionResponse(BaseModel):
    """版本记录响应"""
    id: str
    shareId: str
    version: int
    changeLog: Optional[str]
    changesSummary: Optional[Dict[str, Any]]
    publishedAt: datetime

    model_config = ConfigDict(from_attributes=True)


# ============ TreeUpdateNotification Schemas ============

class TreeUpdateNotificationResponse(BaseModel):
    """更新通知响应"""
    id: str
    subscriptionId: str
    versionId: str
    userId: str
    isRead: bool
    isApplied: bool
    createdAt: datetime
    readAt: Optional[datetime]
    appliedAt: Optional[datetime]

    # 额外信息
    version: Optional[int] = None
    changeLog: Optional[str] = None
    topicName: Optional[str] = None
    shareCode: Optional[str] = None
    changesSummary: Optional[Dict[str, Any]] = None  # 变更摘要

    model_config = ConfigDict(from_attributes=True)


# ============ Diff & Sync Schemas ============

class NodeDiff(BaseModel):
    """节点差异"""
    nodeId: str
    nodeName: str
    changeType: str  # added, modified, deleted
    changes: Optional[Dict[str, Any]] = None  # 具体变更内容


class MaterialDiff(BaseModel):
    """资料差异"""
    materialId: str
    materialName: str
    changeType: str  # added, modified, deleted
    changes: Optional[Dict[str, Any]] = None


class TreeDiffResponse(BaseModel):
    """差异报告响应"""
    fromVersion: int
    toVersion: int
    nodeChanges: List[NodeDiff]
    materialChanges: List[MaterialDiff]
    summary: Dict[str, int]  # nodesAdded, nodesModified, nodesDeleted, etc.


class SyncRequest(BaseModel):
    """同步请求"""
    acceptedNodeIds: Optional[List[str]] = Field(None, description="接受更新的节点ID列表")
    acceptedMaterialIds: Optional[List[str]] = Field(None, description="接受更新的资料ID列表")
    acceptAll: bool = Field(default=False, description="是否接受所有更新")


class SyncResponse(BaseModel):
    """同步响应"""
    success: bool
    syncedVersion: int
    nodesUpdated: int
    materialsUpdated: int
    conflicts: Optional[List[Dict[str, Any]]] = None

