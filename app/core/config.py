from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Optional
import os
import json


class Settings(BaseSettings):
    """应用配置"""

    # API 配置
    API_TITLE: str = "Smart Tree API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "AI-powered knowledge tree API"

    # JWT 配置
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    # Token 有效期：默认 7 天（10080 分钟），适合学习类应用的长会话场景
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080

    # 数据库配置
    DATABASE_URL: str

    # 系统内置 LLM（可选）：用于"使用官方额度（扣积分）"模式
    # 支持 DeepSeek / DashScope 等 OpenAI 兼容的 API
    SYSTEM_LLM_API_KEY: Optional[str] = None
    SYSTEM_LLM_BASE_URL: str = "https://api.deepseek.com"
    SYSTEM_LLM_MODEL: str = "deepseek-chat"  # DeepSeek-V3.2 非思考模式

    # 兼容旧配置（DASHSCOPE_* 会自动映射到 SYSTEM_LLM_*）
    DASHSCOPE_API_KEY: Optional[str] = None
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 积分（Credits）配置
    # - NEW_USER_BONUS_POINTS: 新用户赠送积分
    # - POINTS_PER_RMB: 积分与人民币的换算（默认 1000 积分 = 1 元）
    # - CREDITS_HMAC_SECRET: 账本签名密钥（缺省则复用 JWT_SECRET）
    NEW_USER_BONUS_POINTS: int = 2000
    POINTS_PER_RMB: int = 1000
    CREDITS_HMAC_SECRET: Optional[str] = None

    # 安全：是否允许将 baseUrl 指向 localhost/内网（默认不允许，防 SSRF；本地自用可设为 true）
    ALLOW_PRIVATE_LLM_BASEURL: bool = False

    # CORS 配置
    # 支持通过环境变量 CORS_ORIGINS 覆盖：
    # - JSON 数组：["https://a.com","https://b.com"]
    # - 逗号分隔：https://a.com,https://b.com
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:3002",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3002",
    ]

    # 默认放开本地开发端口（3000/3001/3002/...），避免端口变化导致 CORS 失败
    # 线上如需更严格控制，可通过环境变量 CORS_ALLOW_ORIGIN_REGEX 设置/覆盖
    CORS_ALLOW_ORIGIN_REGEX: Optional[str] = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            raw = v.strip()
            if not raw:
                return []
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            return [item.strip() for item in raw.split(",") if item.strip()]
        return v

    @field_validator("CORS_ALLOW_ORIGIN_REGEX", mode="before")
    @classmethod
    def _normalize_cors_regex(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            raw = v.strip()
            return raw or None
        return v

    @staticmethod
    def _default_env_file() -> str:
        env_file = os.getenv("ENV_FILE")
        if env_file:
            return env_file
        for candidate in (".env.sqlite", ".env.sqlite.example", ".env"):
            if os.path.exists(candidate):
                return candidate
        return ".env"

    model_config = SettingsConfigDict(env_file=_default_env_file.__func__(), extra="ignore")


settings = Settings()
