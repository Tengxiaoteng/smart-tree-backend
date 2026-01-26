from sqlalchemy import Column, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base
from datetime import datetime


class UserProfile(Base):
    __tablename__ = "user_profile"

    userId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), primary_key=True)

    # ===== 基本信息 =====
    email = Column(String(255), nullable=True, index=True)
    avatarUrl = Column(String(1024), nullable=True)
    bio = Column(Text, nullable=True)
    timezone = Column(String(64), nullable=True)
    language = Column(String(32), nullable=True)

    # ===== 可扩展信息（JSON）=====
    education = Column(JSON, nullable=True)  # {school, major, grade, ...}
    preferences = Column(JSON, nullable=True)  # 学习偏好/目标等
    learningHabits = Column(JSON, nullable=True)  # 学习习惯/统计快照

    # ===== 树苗画像（AI 动态）=====
    seedlingPortrait = Column(JSON, nullable=True)
    portraitUpdatedAt = Column(DateTime, nullable=True)

    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="profile")

    def __repr__(self):
        return f"<UserProfile userId={self.userId}>"
