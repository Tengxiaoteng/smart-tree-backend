from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, Boolean, func
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
from datetime import datetime


class Topic(Base):
    __tablename__ = "topic"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    scope = Column(JSON, nullable=True)  # 列表，存储主题范围
    keywords = Column(JSON, nullable=True)  # 列表，存储关键词
    rootNodeId = Column(String(191), nullable=True)  # 根节点ID

    # ===== 分享相关字段 =====
    # 如果是从分享复制来的，记录来源分享ID
    sourceShareId = Column(String(191), ForeignKey("tree_share.id", ondelete="SET NULL"), nullable=True, index=True)
    # 是否已分享（冗余字段，方便查询）
    isShared = Column(Boolean, default=False, nullable=False)
    # 原始节点ID到本地节点ID的映射（JSON格式）
    # 格式: {"original_node_id": "local_node_id", ...}
    originalNodeMapping = Column(JSON, nullable=True)
    # 原始资料ID到本地资料ID的映射（JSON格式）
    originalMaterialMapping = Column(JSON, nullable=True)

    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 关系
    user = relationship("User", back_populates="topics")
    nodes = relationship("KnowledgeNode", back_populates="topic", cascade="all, delete-orphan")
    materials = relationship("Material", back_populates="topic", cascade="all, delete-orphan")

    # 知识树设置（一对一）
    settings = relationship("TreeSettings", back_populates="topic", uselist=False, cascade="all, delete-orphan")

    # 分享相关关系
    shares = relationship("TreeShare", back_populates="topic", cascade="all, delete-orphan", foreign_keys="TreeShare.topicId")
    sourceShare = relationship("TreeShare", foreign_keys=[sourceShareId])
    subscription = relationship("TreeSubscription", back_populates="localTopic", uselist=False, foreign_keys="TreeSubscription.localTopicId", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Topic {self.name}>"
