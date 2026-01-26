import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base


class UserSettings(Base):
    # 与 Prisma schema 对齐：@@map("user_setting")
    __tablename__ = "user_setting"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(
        String(191),
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    apiKey = Column(Text, nullable=True)
    modelId = Column(String(255), nullable=True)
    baseUrl = Column(String(1024), nullable=True)
    extras = Column(JSON, nullable=True)
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="settings")

    def __repr__(self):
        return f"<UserSettings userId={self.userId}>"
