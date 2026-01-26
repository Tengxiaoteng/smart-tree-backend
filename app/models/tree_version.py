"""
TreeVersion 模型 - 知识树版本快照
"""
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Index
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship
from app.core.database import Base
import uuid
from datetime import datetime


class TreeVersion(Base):
    """
    知识树版本快照
    
    每当原作者发布更新时，会创建一个新的版本快照。
    快照包含知识树的完整结构数据，用于差异对比和同步。
    """
    __tablename__ = "tree_version"
    __table_args__ = (
        Index("idx_tree_version_share", "shareId"),
        Index("idx_tree_version_share_version", "shareId", "version", unique=True),
    )

    # 主键
    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # 关联的分享记录ID
    shareId = Column(String(191), ForeignKey("tree_share.id", ondelete="CASCADE"), nullable=False)
    
    # 版本号（递增）
    version = Column(Integer, nullable=False)
    
    # 快照数据（JSON格式，包含完整的知识树结构）
    # 格式: {
    #   "topic": { "name": "...", "description": "...", ... },
    #   "nodes": { "node_id": { "name": "...", "parentId": "...", ... }, ... },
    #   "materials": { "material_id": { "name": "...", "type": "...", ... }, ... }
    # }
    snapshotData = Column(Text().with_variant(LONGTEXT, "mysql"), nullable=False)
    
    # 变更日志（作者填写的更新说明）
    changeLog = Column(Text, nullable=True)
    
    # 变更统计（JSON格式）
    # 格式: {
    #   "nodesAdded": 5,
    #   "nodesModified": 3,
    #   "nodesDeleted": 1,
    #   "materialsAdded": 2,
    #   "materialsModified": 1,
    #   "materialsDeleted": 0
    # }
    changesSummary = Column(Text, nullable=True)
    
    # 发布时间
    publishedAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # 关系
    share = relationship("TreeShare", back_populates="versions")
    notifications = relationship("TreeUpdateNotification", back_populates="version", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TreeVersion {self.version} for Share {self.shareId}>"

