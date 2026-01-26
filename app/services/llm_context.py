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


def _get_system_llm_api_key() -> str | None:
    """获取系统 LLM API Key（优先 SYSTEM_LLM_API_KEY，兼容 DASHSCOPE_API_KEY）"""
    api_key = getattr(settings, "SYSTEM_LLM_API_KEY", None)
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    # 兼容旧配置
    api_key = getattr(settings, "DASHSCOPE_API_KEY", None)
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    return None


def _get_system_llm_base_url() -> str:
    """获取系统 LLM Base URL（优先 SYSTEM_LLM_BASE_URL，兼容 DASHSCOPE_BASE_URL）"""
    base_url = getattr(settings, "SYSTEM_LLM_BASE_URL", None)
    if isinstance(base_url, str) and base_url.strip():
        return base_url.strip()
    # 兼容旧配置
    base_url = getattr(settings, "DASHSCOPE_BASE_URL", None)
    if isinstance(base_url, str) and base_url.strip():
        return base_url.strip()
    return "https://api.deepseek.com"


def _get_system_llm_model() -> str | None:
    """获取系统 LLM 默认模型"""
    model = getattr(settings, "SYSTEM_LLM_MODEL", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    return "deepseek-chat"


def system_llm_available() -> bool:
    return bool(_get_system_llm_api_key())


def _get_vision_llm_api_key() -> str | None:
    """获取多模态/视觉 LLM API Key（使用 DashScope）"""
    api_key = getattr(settings, "DASHSCOPE_API_KEY", None)
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    return None


def _get_vision_llm_base_url() -> str:
    """获取多模态/视觉 LLM Base URL（使用 DashScope）"""
    base_url = getattr(settings, "DASHSCOPE_BASE_URL", None)
    if isinstance(base_url, str) and base_url.strip():
        return base_url.strip()
    return "https://dashscope.aliyuncs.com/compatible-mode/v1"


def vision_llm_available() -> bool:
    """检查视觉/多模态 LLM 是否可用"""
    return bool(_get_vision_llm_api_key())


@dataclass(frozen=True)
class ResolvedVisionLLMConfig:
    """多模态/视觉 LLM 配置（固定使用 DashScope qwen-vl-plus）"""
    api_key: str
    base_url: str
    model_id: str = "qwen-vl-plus"


def get_vision_llm_config() -> ResolvedVisionLLMConfig | None:
    """获取多模态/视觉 LLM 配置（用于图片、PDF 分析）"""
    api_key = _get_vision_llm_api_key()
    if not api_key:
        return None
    return ResolvedVisionLLMConfig(
        api_key=api_key,
        base_url=_get_vision_llm_base_url(),
        model_id="qwen-vl-plus",
    )


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
        # 默认：优先使用官方提供的 API（系统内置）
        use_system = bool(system_available)

    if use_system:
        if not system_available:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="系统模型未配置（缺少 SYSTEM_LLM_API_KEY），请改用自己的 API 设置或联系管理员配置",
            )
        api_key = _get_system_llm_api_key()
        base_url = _get_system_llm_base_url()
        model_id = _get_system_llm_model()
        return ResolvedLLMConfig(
            mode="system",
            api_key=api_key,
            base_url=base_url,
            model_id=model_id,
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
            api_key = _get_system_llm_api_key()
            base_url = _get_system_llm_base_url()
            model_id = _get_system_llm_model()
            return ResolvedLLMConfig(
                mode="system",
                api_key=api_key,
                base_url=base_url,
                model_id=model_id,
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

