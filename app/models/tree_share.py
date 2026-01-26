"""
TreeShare 模型 - 知识树分享记录
"""
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Boolean, Index
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
import secrets
import string
from datetime import datetime


def generate_share_code(length: int = 8) -> str:
    """生成唯一的分享码（字母数字组合）"""
    alphabet = string.ascii_uppercase + string.digits
    # 移除容易混淆的字符
    alphabet = alphabet.replace('O', '').replace('0', '').replace('I', '').replace('1', '').replace('L', '')
    return ''.join(secrets.choice(alphabet) for _ in range(length))


class TreeShare(Base):
    """
    知识树分享记录
    
    当用户分享一棵知识树时，会创建一条分享记录。
    其他用户可以通过分享码复制该知识树到自己的账户。
    """
    __tablename__ = "tree_share"
    __table_args__ = (
        Index("idx_tree_share_owner", "ownerId"),
        Index("idx_tree_share_topic", "topicId"),
        Index("idx_tree_share_code", "shareCode", unique=True),
    )

    # 主键
    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # 被分享的知识树ID
    topicId = Column(String(191), ForeignKey("topic.id", ondelete="CASCADE"), nullable=False)
    
    # 分享者（原作者）ID
    ownerId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    
    # 唯一分享码（用于分享链接）
    shareCode = Column(String(20), unique=True, nullable=False, default=generate_share_code)
    
    # 分享类型：public（公开）/ private（私密，需要密码）
    shareType = Column(String(20), default="public", nullable=False)
    
    # 分享密码（仅当 shareType 为 private 时使用）
    sharePassword = Column(String(255), nullable=True)
    
    # 分享标题（可选，默认使用知识树名称）
    shareTitle = Column(String(255), nullable=True)
    
    # 分享描述
    shareDescription = Column(Text, nullable=True)
    
    # 是否有效（可以禁用分享而不删除记录）
    isActive = Column(Boolean, default=True, nullable=False)
    
    # 当前发布的版本号（每次发布更新时递增）
    currentVersion = Column(Integer, default=1, nullable=False)
    
    # 订阅者数量（冗余字段，方便查询）
    subscriberCount = Column(Integer, default=0, nullable=False)
    
    # 复制次数（统计）
    copyCount = Column(Integer, default=0, nullable=False)
    
    # 是否允许复制
    allowCopy = Column(Boolean, default=True, nullable=False)
    
    # 浏览次数（统计）
    viewCount = Column(Integer, default=0, nullable=False)
    
    # 时间戳
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # 关系
    owner = relationship("User", back_populates="tree_shares")
    topic = relationship("Topic", back_populates="shares", foreign_keys=[topicId])
    subscriptions = relationship("TreeSubscription", back_populates="share", cascade="all, delete-orphan")
    versions = relationship("TreeVersion", back_populates="share", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TreeShare {self.shareCode} for Topic {self.topicId}>"
    
    def increment_version(self) -> int:
        """递增版本号并返回新版本号"""
        self.currentVersion += 1
        return self.currentVersion
    
    def increment_subscriber_count(self):
        """增加订阅者数量"""
        self.subscriberCount += 1
    
    def decrement_subscriber_count(self):
        """减少订阅者数量"""
        if self.subscriberCount > 0:
            self.subscriberCount -= 1
    
    def increment_copy_count(self):
        """增加复制次数"""
        self.copyCount += 1
    
    def increment_view_count(self):
        """增加浏览次数"""
        self.viewCount += 1

