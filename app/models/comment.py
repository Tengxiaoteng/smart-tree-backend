from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
from datetime import datetime, timezone


def utc_now():
    """返回带时区的 UTC 时间"""
    return datetime.now(timezone.utc)


class Comment(Base):
    """帖子评论"""
    __tablename__ = "comment"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    postId = Column(String(36), ForeignKey("post.id", ondelete="CASCADE"), nullable=False, index=True)
    userId = Column(String(191), ForeignKey("user.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    parentId = Column(String(36), nullable=True, index=True)  # 回复的评论ID（楼中楼）
    replyToUserId = Column(String(191), nullable=True)  # 被回复的用户ID
    createdAt = Column(DateTime(timezone=True), default=utc_now)

    # 关系
    post = relationship("Post", back_populates="comments")
    user = relationship("User", backref="comments")

    def __repr__(self):
        return f"<Comment {self.id}>"

