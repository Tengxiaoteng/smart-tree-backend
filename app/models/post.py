from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
from datetime import datetime, timezone


def utc_now():
    """返回带时区的 UTC 时间"""
    return datetime.now(timezone.utc)


class Post(Base):
    """社区帖子"""
    __tablename__ = "post"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String(191), ForeignKey("user.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)  # 正文
    images = Column(JSON, nullable=True)  # 图片URL列表，如 ["/api/files/xxx", ...]
    nodeId = Column(String(191), nullable=True, index=True)  # 可选关联节点
    topicId = Column(String(191), nullable=True)  # 关联的Topic（用于分类继承，已废弃）
    shareCode = Column(String(20), nullable=True, index=True)  # 关联的分享码，其他用户可通过此码获取知识树
    categoryId = Column(String(36), nullable=True, index=True)  # 系统分类
    tags = Column(JSON, nullable=True)  # 用户自定义标签，如 ["数学", "线性代数"]
    viewCount = Column(Integer, default=0)
    commentCount = Column(Integer, default=0)
    shareCount = Column(Integer, default=0)
    createdAt = Column(DateTime(timezone=True), default=utc_now)
    updatedAt = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # 关系
    user = relationship("User", backref="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Post {self.title}>"

    def __repr__(self):
        return f"<Post {self.title}>"

