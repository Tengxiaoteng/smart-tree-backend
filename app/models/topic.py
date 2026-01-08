from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, func
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
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 关系
    user = relationship("User", back_populates="topics")
    nodes = relationship("KnowledgeNode", back_populates="topic", cascade="all, delete-orphan")
    materials = relationship("Material", back_populates="topic", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Topic {self.name}>"
