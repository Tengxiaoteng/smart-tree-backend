from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from enum import Enum

from app.core.database import Base


class FileType(str, Enum):
    IMAGE = "image"
    PDF = "pdf"
    DOCUMENT = "document"


class UserFile(Base):
    """用户文件元数据表"""
    __tablename__ = "user_file"

    id = Column(String(36), primary_key=True)
    userId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    ossPath = Column(String(512), nullable=False)  # OSS 完整路径
    filename = Column(String(255), nullable=False)  # 原始文件名
    fileType = Column(String(32), nullable=False)  # 使用 String 避免 Enum 大小写问题
    fileSize = Column(Integer, nullable=False)
    mimeType = Column(String(64), nullable=True)
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
