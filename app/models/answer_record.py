import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship

from app.core.database import Base


class AnswerRecord(Base):
    __tablename__ = "answerrecord"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    questionId = Column(String(191), ForeignKey("question.id", ondelete="CASCADE"), nullable=False, index=True)
    nodeId = Column(String(191), ForeignKey("knowledgenode.id", ondelete="CASCADE"), nullable=False, index=True)

    # 与 Prisma schema 对齐：userAnswer 为 Text（可存 "A" 或 JSON 字符串）
    userAnswer = Column(Text, nullable=True)
    isCorrect = Column(Boolean, nullable=False, default=False)
    timeSpent = Column(Integer, nullable=True)  # seconds

    # 答题后的追问对话记录（JSON 数组）
    # 格式: [{"id": "xxx", "role": "user"|"assistant", "content": "...", "timestamp": "..."}]
    followUpMessages = Column(JSON, nullable=True)

    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="answerRecords")
    question = relationship("Question", back_populates="answerRecords")
    node = relationship("KnowledgeNode", back_populates="answerRecords")

    def __repr__(self) -> str:
        return f"<AnswerRecord id={self.id} questionId={self.questionId} nodeId={self.nodeId}>"
