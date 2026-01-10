from sqlalchemy import Column, String, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
from datetime import datetime, timezone


def utc_now():
    """返回带时区的 UTC 时间"""
    return datetime.now(timezone.utc)


class PostView(Base):
    """帖子浏览记录 - 用于统计去重浏览量"""
    __tablename__ = "post_view"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    postId = Column(String(36), ForeignKey("post.id", ondelete="CASCADE"), nullable=False)
    userId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    viewedAt = Column(DateTime(timezone=True), default=utc_now, index=True)

    # 唯一约束：同一用户对同一帖子只能有一条记录
    __table_args__ = (
        UniqueConstraint('postId', 'userId', name='uq_post_user_view'),
        Index('idx_post_view_post_time', 'postId', 'viewedAt'),
    )

    # 关系
    post = relationship("Post", backref="views")
    user = relationship("User", backref="post_views")

    def __repr__(self):
        return f"<PostView post={self.postId} user={self.userId}>"

