from dataclasses import dataclass
from typing import Any, Literal, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import UserSettings


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _safe_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


def system_llm_available() -> bool:
    api_key = getattr(settings, "DASHSCOPE_API_KEY", None)
    return bool(isinstance(api_key, str) and api_key.strip())


@dataclass(frozen=True)
class ResolvedLLMConfig:
    mode: Literal["system", "byok"]
    api_key: str
    base_url: str
    model_id: str | None
    routing: Literal["auto", "manual"]
    user_settings: UserSettings | None = None


def resolve_llm_config(
    db: Session,
    *,
    user_id: str,
    requested_use_system: bool | None,
    override_api_key: str | None,
    override_base_url: str | None,
    override_model_id: str | None,
    override_routing: str | None,
) -> ResolvedLLMConfig:
    system_available = system_llm_available()

    user_settings: UserSettings | None = None
    try:
        user_settings = db.query(UserSettings).filter(UserSettings.userId == user_id).first()
    except Exception:
        user_settings = None

    extras = user_settings.extras if user_settings and isinstance(user_settings.extras, dict) else {}
    stored_use_system = _safe_bool(extras.get("useSystemKey"))
    stored_routing = _safe_str(extras.get("routing"))

    user_has_config = bool(
        (override_api_key and override_base_url)
        or (user_settings and user_settings.apiKey and user_settings.baseUrl)
    )

    if requested_use_system is not None:
        use_system = requested_use_system
    elif stored_use_system is not None:
        use_system = stored_use_system
    else:
        # 默认：优先使用用户自己的配置；未配置时再尝试系统内置
        use_system = False if user_has_config else bool(system_available)

    if use_system:
        if not system_available:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="系统模型未配置（缺少 DASHSCOPE_API_KEY），请改用自己的 API 设置或联系管理员配置",
            )
        api_key = str(settings.DASHSCOPE_API_KEY).strip()
        base_url = str(getattr(settings, "DASHSCOPE_BASE_URL", "") or "").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return ResolvedLLMConfig(
            mode="system",
            api_key=api_key,
            base_url=base_url,
            model_id=None,
            routing="auto",
            user_settings=user_settings,
        )

    api_key = override_api_key or (user_settings.apiKey.strip() if user_settings and user_settings.apiKey else None)
    base_url = override_base_url or (user_settings.baseUrl.strip() if user_settings and user_settings.baseUrl else None)
    model_id = override_model_id or (user_settings.modelId.strip() if user_settings and user_settings.modelId else None)

    routing_value = (override_routing or stored_routing or "manual").strip().lower()
    routing: Literal["auto", "manual"] = "auto" if routing_value == "auto" else "manual"

    if not api_key or not base_url:
        if system_available and requested_use_system is None:
            api_key = str(settings.DASHSCOPE_API_KEY).strip()
            base_url = str(getattr(settings, "DASHSCOPE_BASE_URL", "") or "").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            return ResolvedLLMConfig(
                mode="system",
                api_key=api_key,
                base_url=base_url,
                model_id=None,
                routing="auto",
                user_settings=user_settings,
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请先在「API 设置」中配置 apiKey/baseUrl，或切换到“使用官方额度”模式",
        )

    return ResolvedLLMConfig(
        mode="byok",
        api_key=api_key,
        base_url=base_url,
        model_id=model_id,
        routing=routing,
        user_settings=user_settings,
    )

