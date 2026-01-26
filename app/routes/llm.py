import json
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User
from app.schemas.llm import LLMChatRequest
from app.services import credits as credits_service
from app.services.llm_context import resolve_llm_config, get_vision_llm_config
from app.services.llm import (
    QWEN_MODELS,
    build_router_messages,
    calc_qwen_cost_points,
    call_openai_compatible_chat,
    estimate_prompt_tokens,
    new_request_id,
)
from app.services.credits_pricing import (
    FeatureType,
    get_chat_feature_type,
    calculate_fixed_points,
    estimate_reserve_points,
)

router = APIRouter()


def _safe_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return None


def _contains_image_content(messages: list) -> bool:
    """检查消息中是否包含图片内容"""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # 检查 image_url 类型
                    if item.get("type") == "image_url":
                        return True
                    # 检查 image 类型
                    if item.get("type") == "image":
                        return True
    return False


def _extract_router_choice(content: str) -> tuple[str, str | None]:
    if not content:
        return "qwen-plus", None
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return "qwen-plus", None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "qwen-plus", None
    model = parsed.get("model") if isinstance(parsed, dict) else None
    reason = parsed.get("reason") if isinstance(parsed, dict) else None
    model = model.strip() if isinstance(model, str) else ""
    if model not in QWEN_MODELS:
        model = "qwen-plus"
    return model, reason.strip() if isinstance(reason, str) and reason.strip() else None


@router.post("/chat")
async def chat(
    payload: LLMChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    messages = payload.messages or []
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="messages 不能为空")

    req_id_base = _safe_str(payload.requestId) or new_request_id()

    override_api_key = _safe_str(payload.apiKey)
    override_base_url = _safe_str(payload.baseUrl)
    override_model_id = _safe_str(payload.modelId)
    override_routing = _safe_str(payload.routing)
    requested_use_system = payload.useSystemKey if isinstance(payload.useSystemKey, bool) else None

    resolved = resolve_llm_config(
        db,
        user_id=current_user.id,
        requested_use_system=requested_use_system,
        override_api_key=override_api_key,
        override_base_url=override_base_url,
        override_model_id=override_model_id,
        override_routing=override_routing,
    )

    use_system = resolved.mode == "system"
    api_key = resolved.api_key
    base_url = resolved.base_url
    model_id = resolved.model_id
    routing = resolved.routing

    # 检测消息中是否包含图片，如果有则使用视觉模型
    has_image = _contains_image_content(messages)
    use_vision_model = False
    if has_image and use_system:
        vision_config = get_vision_llm_config()
        if vision_config:
            print(f"[LLM Chat] 检测到图片内容，切换到视觉模型: {vision_config.model_id}")
            api_key = vision_config.api_key
            base_url = vision_config.base_url
            model_id = vision_config.model_id
            use_vision_model = True
            # 视觉模型不需要路由
            routing = "manual"

    max_tokens = int(payload.max_tokens or 4096)
    if max_tokens < 1:
        max_tokens = 1
    if max_tokens > 8192:
        max_tokens = 8192
    temperature = float(payload.temperature if payload.temperature is not None else 0.7)
    if temperature < 0:
        temperature = 0.0
    if temperature > 2:
        temperature = 2.0

    # 清理历史遗留的“卡住预扣”（例如进程中断导致未 finalize）
    if use_system:
        credits_service.refund_stale_reservations(db, current_user.id)

    routed_model_reason: str | None = None

    # 自动路由：仅对特定 OpenAI 兼容服务生效（视觉模型不需要路由）
    should_route = routing == "auto" and "dashscope" in base_url and not use_vision_model
    if should_route:
        router_request_id = f"{req_id_base}:router"
        router_messages = build_router_messages(messages)
        router_est_prompt = estimate_prompt_tokens(router_messages)
        router_reserved_points, _ = calc_qwen_cost_points(
            model="qwen-flash",
            prompt_tokens=router_est_prompt,
            completion_tokens=256,
        )
        router_reserved_points = max(int(router_reserved_points or 1), 1)

        if use_system:
            credits_service.reserve_points(
                db,
                current_user.id,
                request_id=router_request_id,
                points=router_reserved_points,
                meta={"stage": "router", "model": "qwen-flash"},
            )

        try:
            router_data = await call_openai_compatible_chat(
                api_key=api_key,
                model_id="qwen-flash",
                base_url=base_url,
                messages=router_messages,
                max_tokens=256,
                temperature=0.0,
                timeout_seconds=60.0,
            )
        except Exception as exc:
            if use_system:
                credits_service.finalize_reservation(
                    db,
                    current_user.id,
                    request_id=router_request_id,
                    reserved_points=router_reserved_points,
                    actual_points=0,
                    model="qwen-flash",
                    meta={"stage": "router", "error": str(exc)},
                )
            raise

        router_usage = router_data.get("usage") if isinstance(router_data, dict) else None
        router_prompt_tokens = int(router_usage.get("prompt_tokens") or 0) if isinstance(router_usage, dict) else 0
        router_completion_tokens = int(router_usage.get("completion_tokens") or 0) if isinstance(router_usage, dict) else 0
        router_total_tokens = int(router_usage.get("total_tokens") or 0) if isinstance(router_usage, dict) else 0

        router_content = ""
        try:
            router_content = router_data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        except Exception:
            router_content = ""
        model_id, routed_model_reason = _extract_router_choice(router_content)

        if use_system:
            router_actual_points, router_cost_milli = calc_qwen_cost_points(
                model="qwen-flash",
                prompt_tokens=router_prompt_tokens or router_est_prompt,
                completion_tokens=router_completion_tokens or 64,
            )
            credits_service.finalize_reservation(
                db,
                current_user.id,
                request_id=router_request_id,
                reserved_points=router_reserved_points,
                actual_points=int(router_actual_points or router_reserved_points),
                model="qwen-flash",
                prompt_tokens=router_prompt_tokens or None,
                completion_tokens=router_completion_tokens or None,
                total_tokens=router_total_tokens or None,
                cost_rmb_milli=router_cost_milli,
                meta={
                    "stage": "router",
                    "routedModel": model_id,
                    "reason": routed_model_reason,
                },
            )

    # 手动模式需要 modelId
    if not should_route:
        if use_system:
            # 系统模式默认使用均衡款
            model_id = model_id or "qwen-plus"
        if not model_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="modelId 未配置")

    # 扣费/调用主模型
    if use_system:
        # 计算输入长度，确定功能类型（短/中/长对话）
        input_length = sum(
            len(msg.get("content", "")) if isinstance(msg.get("content"), str)
            else sum(len(p.get("text", "")) for p in msg.get("content", []) if isinstance(p, dict) and p.get("type") == "text")
            for msg in messages
        )
        chat_feature = get_chat_feature_type(input_length)
        fixed_points = calculate_fixed_points(chat_feature)

        # 优先尝试路由结果，其次 plus；余额不足时自动降级到更便宜的模型
        # 视觉模型不需要 fallback
        if use_vision_model:
            candidates = [model_id]
        else:
            candidates = [model_id]
            for fallback in ("qwen-plus", "qwen-flash"):
                if fallback not in candidates:
                    candidates.append(fallback)

        last_error: Exception | None = None
        for candidate in candidates:
            if "dashscope" in base_url and candidate not in QWEN_MODELS:
                continue

            chat_request_id = f"{req_id_base}:chat:{candidate}"

            try:
                credits_service.reserve_points(
                    db,
                    current_user.id,
                    request_id=chat_request_id,
                    points=fixed_points,
                    meta={
                        "stage": "chat",
                        "model": candidate,
                        "feature": chat_feature.value,
                        "inputLength": input_length,
                        "routerReason": routed_model_reason,
                    },
                )
            except HTTPException as exc:
                # 余额不足：尝试下一个更便宜的候选
                last_error = exc
                continue

            try:
                data = await call_openai_compatible_chat(
                    api_key=api_key,
                    model_id=candidate,
                    base_url=base_url,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_seconds=180.0,
                )
            except Exception as exc:
                credits_service.finalize_reservation(
                    db,
                    current_user.id,
                    request_id=chat_request_id,
                    reserved_points=fixed_points,
                    actual_points=0,
                    model=candidate,
                    meta={"stage": "chat", "error": str(exc)},
                )
                raise

            usage = data.get("usage") if isinstance(data, dict) else None
            prompt_tokens = int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0
            completion_tokens = int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0
            total_tokens = int(usage.get("total_tokens") or 0) if isinstance(usage, dict) else 0

            # 固定积分制：实际扣费等于预扣积分
            credits_service.finalize_reservation(
                db,
                current_user.id,
                request_id=chat_request_id,
                reserved_points=fixed_points,
                actual_points=fixed_points,
                model=candidate,
                prompt_tokens=prompt_tokens or None,
                completion_tokens=completion_tokens or None,
                total_tokens=total_tokens or None,
                meta={
                    "stage": "chat",
                    "feature": chat_feature.value,
                    "routerReason": routed_model_reason,
                    "routing": routing,
                },
            )

            # 附加一些可选的账单信息（前端可忽略）
            data.pop("model", None)
            data["billing"] = {
                "mode": "system",
                "chargedPoints": fixed_points,
                "feature": chat_feature.value,
                "balance": credits_service.get_balance(db, current_user.id, verify=False),
            }
            return data

        raise last_error if last_error else HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="积分不足")

    # BYOK（不扣积分）
    data = await call_openai_compatible_chat(
        api_key=api_key,
        model_id=model_id,
        base_url=base_url,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=180.0,
    )
    data["billing"] = {"mode": "byok", "model": model_id}
    return data
