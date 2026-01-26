import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from app.core.database import Base


class UserBatchJob(Base):
    __tablename__ = "user_batch_job"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)

    provider = Column(String(64), nullable=False, default="dashscope")
    mode = Column(String(16), nullable=False)  # system | byok

    baseUrl = Column(String(1024), nullable=False)
    endpoint = Column(String(255), nullable=False)
    model = Column(String(255), nullable=False)
    completionWindow = Column(String(32), nullable=False, default="24h")

    batchId = Column(String(255), nullable=False, unique=True, index=True)
    inputFileId = Column(String(255), nullable=False)
    outputFileId = Column(String(255), nullable=True)
    errorFileId = Column(String(255), nullable=True)

    status = Column(String(32), nullable=False, default="validating")
    requestCounts = Column(JSON, nullable=True)
    job_metadata = Column("metadata", JSON, nullable=True)
    providerData = Column(JSON, nullable=True)

    reservedPoints = Column(Integer, nullable=True)
    chargedPoints = Column(Integer, nullable=True)
    promptTokens = Column(Integer, nullable=True)
    completionTokens = Column(Integer, nullable=True)
    totalTokens = Column(Integer, nullable=True)
    costRmbMilli = Column(Integer, nullable=True)
    billedAt = Column(DateTime, nullable=True)

    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="batchJobs")

    def __repr__(self):
        return f"<UserBatchJob id={self.id} userId={self.userId} status={self.status}>"

