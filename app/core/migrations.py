"""
数据库迁移管理器

提供统一的迁移管理，支持：
1. 自动检测并补齐缺失的表和字段
2. 记录已执行的迁移（避免重复执行）
3. 支持 MySQL 和 SQLite 两种数据库
"""

import logging
from datetime import datetime
from typing import Callable, Optional

from sqlalchemy import Column, DateTime, String, Text, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import Base

logger = logging.getLogger(__name__)


class MigrationHistory(Base):
    """迁移历史记录表"""
    __tablename__ = "migration_history"

    id = Column(String(191), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    executed_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class MigrationManager:
    """迁移管理器"""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.is_mysql = engine.dialect.name == "mysql"
        self._migrations: list[tuple[str, str, Callable[["MigrationManager"], None]]] = []

    def register(self, name: str, description: str = ""):
        """注册迁移的装饰器"""
        def decorator(func: Callable[["MigrationManager"], None]):
            self._migrations.append((name, description, func))
            return func
        return decorator

    def _ensure_history_table(self):
        """确保迁移历史表存在"""
        try:
            MigrationHistory.__table__.create(bind=self.engine, checkfirst=True)
        except SQLAlchemyError as e:
            logger.warning(f"创建迁移历史表失败: {e}")

    def _is_executed(self, name: str) -> bool:
        """检查迁移是否已执行"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT 1 FROM migration_history WHERE name = :name"),
                    {"name": name}
                )
                return result.fetchone() is not None
        except SQLAlchemyError:
            return False

    def _mark_executed(self, name: str, description: str):
        """标记迁移已执行"""
        import uuid
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO migration_history (id, name, description, executed_at)
                        VALUES (:id, :name, :description, :executed_at)
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "name": name,
                        "description": description,
                        "executed_at": datetime.utcnow()
                    }
                )
        except SQLAlchemyError as e:
            logger.warning(f"记录迁移历史失败: {e}")

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """检查列是否存在"""
        if not self.table_exists(table_name):
            return False
        inspector = inspect(self.engine)
        columns = {col["name"] for col in inspector.get_columns(table_name)}
        return column_name in columns

    def index_exists(self, table_name: str, index_name: str) -> bool:
        """检查索引是否存在"""
        if not self.table_exists(table_name):
            return False
        inspector = inspect(self.engine)
        indexes = {idx["name"] for idx in inspector.get_indexes(table_name)}
        return index_name in indexes

    def execute(self, sql: str):
        """执行 SQL"""
        with self.engine.begin() as conn:
            conn.execute(text(sql))

    def add_column(self, table: str, column: str, mysql_type: str, other_type: Optional[str] = None):
        """添加列（如果不存在）"""
        if self.column_exists(table, column):
            return
        col_type = mysql_type if self.is_mysql else (other_type or mysql_type)
        self.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        logger.info(f"已添加列: {table}.{column}")

    def add_index(self, table: str, index_name: str, columns: list[str]):
        """添加索引（如果不存在）"""
        if self.index_exists(table, index_name):
            return
        cols = ", ".join(columns)
        self.execute(f"CREATE INDEX {index_name} ON {table} ({cols})")
        logger.info(f"已添加索引: {index_name} ON {table}")

    def create_table(self, table_name: str, definition: str):
        """创建表（如果不存在）"""
        if self.table_exists(table_name):
            logger.info(f"表已存在，跳过创建: {table_name}")
            return
        sql = f"CREATE TABLE {table_name} ({definition})"
        self.execute(sql)
        logger.info(f"已创建表: {table_name}")

    def drop_table(self, table_name: str):
        """删除表（如果存在）"""
        if not self.table_exists(table_name):
            return
        self.execute(f"DROP TABLE {table_name}")
        logger.info(f"已删除表: {table_name}")

    def run_all(self):
        """运行所有未执行的迁移"""
        self._ensure_history_table()

        for name, description, func in self._migrations:
            if self._is_executed(name):
                continue
            try:
                logger.info(f"执行迁移: {name}")
                func(self)
                self._mark_executed(name, description)
                logger.info(f"迁移完成: {name}")
            except SQLAlchemyError as e:
                logger.error(f"迁移失败 [{name}]: {e}")
                raise

