from sqlalchemy import Column, String, Integer
from app.core.database import Base
import uuid


class Category(Base):
    """系统预设分类"""
    __tablename__ = "category"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(64), nullable=False, unique=True)
    icon = Column(String(64), nullable=True)  # 可选图标
    sortOrder = Column(Integer, default=0)

    def __repr__(self):
        return f"<Category {self.name}>"

