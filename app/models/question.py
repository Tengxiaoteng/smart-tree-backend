import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Text, JSON, text
from sqlalchemy.orm import relationship

from app.core.database import Base


class Question(Base):
    __tablename__ = "question"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    nodeId = Column(String(191), ForeignKey("knowledgenode.id", ondelete="CASCADE"), nullable=False, index=True)

    type = Column(String(50), nullable=False, default="single")
    difficulty = Column(String(50), nullable=True, default="beginner")
    content = Column(Text, nullable=False)
    options = Column(JSON, nullable=True)
    answer = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)
    hints = Column(Text, nullable=True)
    relatedConcepts = Column(Text, nullable=True)
    targetGoalIds = Column(JSON, nullable=True)
    targetGoalNames = Column(JSON, nullable=True)
    isFavorite = Column(Boolean, nullable=False, default=False, server_default=text("0"))
    userNotes = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)

    # 题目来源
    source = Column(String(50), nullable=True)
    sourceMaterialIds = Column(JSON, nullable=True)
    sourceMaterialNames = Column(JSON, nullable=True)
    sourceContext = Column(Text, nullable=True)
    difficultyReason = Column(Text, nullable=True)

    # 衍生题关联
    derivedFromQuestionId = Column(String(191), nullable=True)
    derivedFromRecordId = Column(String(191), nullable=True)
    isDerivedQuestion = Column(Boolean, nullable=False, default=False, server_default=text("0"))
    parentQuestionId = Column(String(191), nullable=True)
    derivationType = Column(String(255), nullable=True)

    # V3 RAG检索追踪
    ragTrace = Column(JSON, nullable=True)  # {retrievedMaterialIds, retrievedConcepts, avgSimilarity}

    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="questions")
    node = relationship("KnowledgeNode", back_populates="questions")
    answerRecords = relationship("AnswerRecord", back_populates="question", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Question id={self.id} nodeId={self.nodeId}>"
