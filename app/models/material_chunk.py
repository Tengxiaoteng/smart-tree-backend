"""
MaterialChunk 模型 - 资料分块存储

用于V3智能出题的RAG检索，将资料分块并存储向量
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, String, Text, JSON, Integer, Index
from sqlalchemy.orm import relationship

from app.core.database import Base


class MaterialChunk(Base):
    """资料分块模型，用于RAG检索"""

    __tablename__ = "material_chunk"
    __table_args__ = (
        Index("idx_material_chunk_material", "materialId"),
        Index("idx_material_chunk_created", "createdAt"),
    )

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    materialId = Column(String(191), ForeignKey("material.id", ondelete="CASCADE"), nullable=False, index=True)

    # 分块信息
    chunkIndex = Column(Integer, nullable=False, default=0)  # 分块索引
    chunkText = Column(Text, nullable=False)  # 分块文本内容
    tokenCount = Column(Integer, nullable=True)  # Token数量（预估）

    # 向量存储
    embedding = Column(JSON, nullable=True)  # 512维向量
    embeddingModel = Column(String(100), nullable=True, default="text-embedding-v4")  # 使用的模型

    # 元数据
    metadata = Column(JSON, nullable=True)  # 额外元数据

    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # 关系
    material = relationship("Material", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<MaterialChunk id={self.id} materialId={self.materialId} index={self.chunkIndex}>"

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "materialId": self.materialId,
            "chunkIndex": self.chunkIndex,
            "chunkText": self.chunkText,
            "tokenCount": self.tokenCount,
            "embedding": self.embedding,
            "embeddingModel": self.embeddingModel,
            "metadata": self.metadata,
            "createdAt": self.createdAt.isoformat() if self.createdAt else None,
            "updatedAt": self.updatedAt.isoformat() if self.updatedAt else None,
        }
