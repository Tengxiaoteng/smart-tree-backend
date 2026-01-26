from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.engine.url import make_url
from typing import Generator
import os
from dotenv import load_dotenv

def _default_env_file() -> str:
    # Prefer local SQLite config for dev/demo to avoid external DB dependency.
    for candidate in (".env.sqlite", ".env.sqlite.example", ".env"):
        if os.path.exists(candidate):
            return candidate
    return ".env"


ENV_FILE = os.getenv("ENV_FILE") or _default_env_file()
load_dotenv(ENV_FILE)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 未配置，请在环境变量或 .env 中设置")

_url = make_url(DATABASE_URL)
_engine_kwargs: dict = {"echo": False, "pool_pre_ping": True}

# 兼容 SQLite 本地开发（避免依赖外部 MySQL，解决无法连接导致后端无法启动的问题）
if _url.drivername.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    _engine_kwargs.update(
        {
            "pool_size": 10,
            "max_overflow": 20,
            "connect_args": {"charset": "utf8mb4", "connect_timeout": 5, "init_command": "SET time_zone = '+08:00'"},
        }
    )

engine = create_engine(DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """数据库会话依赖注入"""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
