from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
from datetime import datetime


class User(Base):
    __tablename__ = "user"

    # 与 Prisma 默认 MySQL 类型兼容（String @id 通常为 VARCHAR(191)）
    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(255), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)
    nickname = Column(String(255), nullable=True)
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 关系
    topics = relationship("Topic", back_populates="user", cascade="all, delete-orphan")
    nodes = relationship("KnowledgeNode", back_populates="user", cascade="all, delete-orphan")
    settings = relationship("UserSettings", back_populates="user", cascade="all, delete-orphan", uselist=False)
    profile = relationship("UserProfile", back_populates="user", cascade="all, delete-orphan", uselist=False)
    materials = relationship("Material", back_populates="user", cascade="all, delete-orphan")
    questions = relationship("Question", back_populates="user", cascade="all, delete-orphan")
    answerRecords = relationship("AnswerRecord", back_populates="user", cascade="all, delete-orphan")
    userNotes = relationship("UserNote", back_populates="user", cascade="all, delete-orphan")

    creditAccount = relationship("UserCreditAccount", back_populates="user", cascade="all, delete-orphan", uselist=False)
    creditLedger = relationship("UserCreditLedger", back_populates="user", cascade="all, delete-orphan")
    batchJobs = relationship("UserBatchJob", back_populates="user", cascade="all, delete-orphan")

    # 分享与订阅相关关系
    tree_shares = relationship("TreeShare", back_populates="owner", cascade="all, delete-orphan")
    tree_subscriptions = relationship("TreeSubscription", back_populates="subscriber", cascade="all, delete-orphan")
    tree_update_notifications = relationship("TreeUpdateNotification", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.username}>"
