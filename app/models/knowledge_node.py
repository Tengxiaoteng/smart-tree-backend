from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON, Integer, Index
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
from datetime import datetime


class KnowledgeNode(Base):
    __tablename__ = "knowledgenode"
    __table_args__ = (
        Index("idx_user_topic", "userId", "topicId"),
        Index("idx_user_parent", "userId", "parentId"),
    )

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    topicId = Column(String(191), ForeignKey("topic.id", ondelete="CASCADE"), nullable=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    learningObjectives = Column(JSON, nullable=True)
    keyConcepts = Column(JSON, nullable=True)
    knowledgeType = Column(String(50), default="concept")  # concept, principle, procedure, application, other
    difficulty = Column(String(50), default="beginner")  # beginner, intermediate, advanced
    estimatedMinutes = Column(Integer, default=15)
    prerequisites = Column(JSON, nullable=True)
    questionPatterns = Column(JSON, nullable=True)
    commonMistakes = Column(JSON, nullable=True)
    children = Column(JSON, nullable=True)  # 子节点ID列表
    materialIds = Column(JSON, nullable=True)
    questionIds = Column(JSON, nullable=True)
    userNotes = Column(JSON, nullable=True)
    aiInferredGoals = Column(JSON, nullable=True)
    source = Column(String(50), default="manual")  # manual, ai_generated
    parentId = Column(String(191), ForeignKey("knowledgenode.id", ondelete="SET NULL"), nullable=True, index=True)
    mastery = Column(Integer, default=0)  # 0-100
    questionCount = Column(Integer, default=0)
    correctCount = Column(Integer, default=0)
    
    # 手动排序序号（仅在 sortMode='manual' 时使用）
    sortOrder = Column(Integer, default=0, nullable=False)

    # 学习状态跟踪（JSON 格式）
    # 格式: {"lastStudied": "2024-01-01T00:00:00", "nextReviewDate": "2024-01-08T00:00:00",
    #        "reviewCount": 0, "confidenceLevel": "low"|"medium"|"high"}
    learningStatus = Column(JSON, nullable=True)

    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 关系
    user = relationship("User", back_populates="nodes")
    topic = relationship("Topic", back_populates="nodes")
    parent = relationship("KnowledgeNode", remote_side=[id], backref="child_nodes")
    questions = relationship("Question", back_populates="node", cascade="all, delete-orphan")
    answerRecords = relationship("AnswerRecord", back_populates="node", cascade="all, delete-orphan")
    notes = relationship("UserNote", back_populates="node", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<KnowledgeNode {self.name}>"
