"""
TreeSubscription 模型 - 知识树订阅记录
"""
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Boolean, Index
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy import Text
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
from datetime import datetime


class TreeSubscription(Base):
    """
    知识树订阅记录
    
    当用户复制一棵分享的知识树时，会创建一条订阅记录。
    订阅记录用于追踪用户的本地副本与原树的同步状态。
    """
    __tablename__ = "tree_subscription"
    __table_args__ = (
        Index("idx_tree_subscription_share", "shareId"),
        Index("idx_tree_subscription_subscriber", "subscriberId"),
        Index("idx_tree_subscription_local_topic", "localTopicId"),
        # 确保同一用户不能重复订阅同一个分享
        Index("idx_tree_subscription_unique", "shareId", "subscriberId", unique=True),
    )

    # 主键
    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # 关联的分享记录ID
    shareId = Column(String(191), ForeignKey("tree_share.id", ondelete="CASCADE"), nullable=False)
    
    # 订阅者ID
    subscriberId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    
    # 订阅者本地复制的知识树ID
    localTopicId = Column(String(191), ForeignKey("topic.id", ondelete="CASCADE"), nullable=False)
    
    # 已同步到的版本号
    syncedVersion = Column(Integer, default=1, nullable=False)
    
    # 最后同步时间
    lastSyncedAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # 是否启用自动同步（预留功能）
    autoSync = Column(Boolean, default=False, nullable=False)
    
    # 是否接收更新通知
    notifyOnUpdate = Column(Boolean, default=True, nullable=False)
    
    # 节点ID映射：原始节点ID -> 本地节点ID（JSON格式）
    # 格式: {"original_node_id": "local_node_id", ...}
    nodeMapping = Column(Text().with_variant(LONGTEXT, "mysql"), nullable=True)
    
    # 资料ID映射：原始资料ID -> 本地资料ID（JSON格式）
    materialMapping = Column(Text().with_variant(LONGTEXT, "mysql"), nullable=True)
    
    # 时间戳
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # 关系
    share = relationship("TreeShare", back_populates="subscriptions")
    subscriber = relationship("User", back_populates="tree_subscriptions")
    localTopic = relationship("Topic", back_populates="subscription", foreign_keys=[localTopicId])
    notifications = relationship("TreeUpdateNotification", back_populates="subscription", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TreeSubscription {self.id} - User {self.subscriberId} subscribed to Share {self.shareId}>"
    
    def update_synced_version(self, version: int):
        """更新已同步的版本号"""
        self.syncedVersion = version
        self.lastSyncedAt = datetime.utcnow()
    
    def has_pending_updates(self, current_version: int) -> bool:
        """检查是否有待同步的更新"""
        return self.syncedVersion < current_version

