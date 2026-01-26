"""
TreeUpdateNotification 模型 - 知识树更新通知
"""
from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, Index
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
from datetime import datetime


class TreeUpdateNotification(Base):
    """
    知识树更新通知
    
    当原作者发布新版本时，会为所有订阅者创建更新通知。
    用户可以查看通知，了解更新内容，并选择是否应用更新。
    """
    __tablename__ = "tree_update_notification"
    __table_args__ = (
        Index("idx_notification_user", "userId"),
        Index("idx_notification_subscription", "subscriptionId"),
        Index("idx_notification_version", "versionId"),
        Index("idx_notification_user_unread", "userId", "isRead"),
    )

    # 主键
    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # 关联的订阅记录ID
    subscriptionId = Column(String(191), ForeignKey("tree_subscription.id", ondelete="CASCADE"), nullable=False)
    
    # 关联的版本ID
    versionId = Column(String(191), ForeignKey("tree_version.id", ondelete="CASCADE"), nullable=False)
    
    # 接收通知的用户ID（冗余字段，方便查询）
    userId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    
    # 是否已读
    isRead = Column(Boolean, default=False, nullable=False)
    
    # 是否已应用更新
    isApplied = Column(Boolean, default=False, nullable=False)
    
    # 创建时间
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # 已读时间
    readAt = Column(DateTime, nullable=True)
    
    # 应用更新时间
    appliedAt = Column(DateTime, nullable=True)
    
    # 关系
    subscription = relationship("TreeSubscription", back_populates="notifications")
    version = relationship("TreeVersion", back_populates="notifications")
    user = relationship("User", back_populates="tree_update_notifications")

    def __repr__(self):
        return f"<TreeUpdateNotification {self.id} for User {self.userId}>"
    
    def mark_as_read(self):
        """标记为已读"""
        self.isRead = True
        self.readAt = datetime.utcnow()
    
    def mark_as_applied(self):
        """标记为已应用"""
        self.isApplied = True
        self.appliedAt = datetime.utcnow()

