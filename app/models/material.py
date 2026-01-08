from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, Integer, Boolean, Index
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
from datetime import datetime


class Material(Base):
    __tablename__ = "material"
    __table_args__ = (
        Index("idx_material_user_topic", "userId", "topicId"),
    )

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    topicId = Column(String(191), ForeignKey("topic.id", ondelete="CASCADE"), nullable=True, index=True)

    # 数据库字段名是 title（与 Prisma schema 对齐），对外使用 name
    name = Column("title", String(255), nullable=False)
    type = Column(String(50), nullable=False, default="text")  # pdf, image, text, ...
    content = Column(Text().with_variant(LONGTEXT, "mysql"), nullable=True)
    url = Column(String(1024), nullable=True)
    fileSize = Column(Integer, nullable=True)

    # 兼容前端/旧数据
    nodeIds = Column(JSON, nullable=True)  # 关联的节点 ID 列表
    tags = Column(JSON, nullable=True)

    # AI/结构化字段（用于“AI 整理 / 出题化”后可持久化）
    organizedContent = Column(Text().with_variant(LONGTEXT, "mysql"), nullable=True)
    aiSummary = Column(Text().with_variant(LONGTEXT, "mysql"), nullable=True)
    extractedConcepts = Column(JSON, nullable=True)
    isOrganized = Column(Boolean, nullable=False, default=False)
    structuredContent = Column(JSON, nullable=True)
    isStructured = Column(Boolean, nullable=False, default=False)

    # ===== 快速匹配字段（用于重复检测优化）=====
    # 内容摘要：50字以内的简短描述，用于快速比较
    contentDigest = Column(String(255), nullable=True)
    # 关键主题：5-10个关键词，用于快速匹配（JSON 数组）
    keyTopics = Column(JSON, nullable=True)
    # 内容哈希：用于快速排除完全相同的内容
    contentHash = Column(String(64), nullable=True, index=True)
    # 摘要生成时间：用于判断是否需要重新生成
    digestGeneratedAt = Column(DateTime, nullable=True)

    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 关系
    user = relationship("User", back_populates="materials")
    topic = relationship("Topic", back_populates="materials")

    def __repr__(self):
        return f"<Material {self.name}>"
