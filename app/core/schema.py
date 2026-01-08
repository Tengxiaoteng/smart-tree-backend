from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import UserFile


def ensure_user_file_schema(db: Session) -> None:
    """确保 user_file 表存在（兼容旧库/首次部署）。"""
    engine = db.get_bind()
    inspector = inspect(engine)
    table_name = UserFile.__tablename__

    try:
        tables = set(inspector.get_table_names())
        if table_name not in tables:
            UserFile.__table__.create(bind=engine, checkfirst=True)
    except SQLAlchemyError:
        # 让上层决定如何返回错误；这里不吞异常以免掩盖数据库权限/连接问题
        raise

