import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from app.core.database import Base


class TreeSettings(Base):
    """知识树个性化设置"""
    __tablename__ = "tree_settings"
    __table_args__ = (
        Index("idx_tree_settings_topic", "topicId"),
    )

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    topicId = Column(
        String(191),
        ForeignKey("topic.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    
    # 节点排序方式
    # createdAt_asc, createdAt_desc, name_asc, mastery_asc, mastery_desc, manual
    sortMode = Column(String(50), default="createdAt_desc", nullable=False)
    
    # 显示设置（JSON 格式）
    # { showMastery: boolean, showChildCount: boolean, showDescription: boolean }
    displayConfig = Column(JSON, nullable=True)
    
    # 可视化设置（JSON 格式）
    # { layoutDirection: 'horizontal'|'vertical', nodeSpacing: number, colorTheme: string }
    visualConfig = Column(JSON, nullable=True)
    
    # 默认展开设置（JSON 格式）
    # { defaultExpandLevel: number }  // -1 表示全部展开，0 表示全部折叠
    expandConfig = Column(JSON, nullable=True)
    
    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # 关系
    topic = relationship("Topic", back_populates="settings")

    def __repr__(self):
        return f"<TreeSettings topicId={self.topicId}>"
