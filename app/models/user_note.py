import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import relationship

from app.core.database import Base


class UserNote(Base):
    __tablename__ = "user_note"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    nodeId = Column(String(191), ForeignKey("knowledgenode.id", ondelete="CASCADE"), nullable=False, index=True)
    questionId = Column(String(191), ForeignKey("question.id", ondelete="SET NULL"), nullable=True, index=True)

    content = Column(Text, nullable=False)
    source = Column(String(50), nullable=False, default="manual")  # manual | from_question | from_chat
    tags = Column(JSON, nullable=True)

    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="userNotes")
    node = relationship("KnowledgeNode", back_populates="notes")
    question = relationship("Question")

    def __repr__(self) -> str:
        return f"<UserNote id={self.id} nodeId={self.nodeId}>"

