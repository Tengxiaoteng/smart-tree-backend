import json
import re
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_UP
from typing import Any, Literal, Optional
from urllib.parse import urlparse
import ipaddress

import httpx
from fastapi import HTTPException, status

from app.core.config import settings


QWEN_MODELS: set[str] = {"qwen-max", "qwen-plus", "qwen-flash", "qwen-coder", "qwen-vl-plus", "qwen-vl-max"}

# 用户提供的“最低价格”档位（元/千 Token）
QWEN_PRICING_RMB_PER_1K: dict[str, dict[str, Decimal]] = {
    "qwen-max": {"input": Decimal("0.0032"), "output": Decimal("0.0128")},
    "qwen-plus": {"input": Decimal("0.0008"), "output": Decimal("0.002")},
    "qwen-flash": {"input": Decimal("0.00015"), "output": Decimal("0.0015")},
    "qwen-coder": {"input": Decimal("0.001"), "output": Decimal("0.004")},
}


def _normalize_chat_completions_url(base_url: str) -> str:
    resolved = base_url.strip() if isinstance(base_url, str) and base_url.strip() else ""
    if not resolved:
        raise ValueError("baseUrl 未配置")
    normalized = re.sub(r"/+$", "", resolved)
    return normalized if "chat/completions" in normalized else f"{normalized}/chat/completions"


def _is_private_host(hostname: str) -> bool:
    if not hostname:
        return True
    lowered = hostname.strip().lower()
    if lowered in {"localhost", "localhost.localdomain"}:
        return True
    if lowered.endswith(".local") or lowered.endswith(".internal"):
        return True
    try:
        ip = ipaddress.ip_address(lowered)
        return bool(
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )
    except ValueError:
        # 非 IP：仅做最小的 hostname 黑名单处理（更严格可加 DNS 解析 + 私网判断）
        return False


def _validate_outbound_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("baseUrl 协议必须是 http/https")
    if not bool(getattr(settings, "ALLOW_PRIVATE_LLM_BASEURL", False)) and _is_private_host(parsed.hostname or ""):
        raise ValueError("baseUrl 不允许指向本地/内网地址")


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    total_chars = 0
    image_parts = 0

    for msg in messages or []:
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
            continue
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    total_chars += len(part.get("text") or "")
                elif part.get("type") == "image_url":
                    image_parts += 1

    # 保守上界：字符数 * 2（覆盖中英文/混合），图片每张按 2000 tokens 预留，再加固定开销
    return int(total_chars * 2 + image_parts * 2000 + 100)


def calc_qwen_cost_points(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    price_multiplier: Decimal | float | str = Decimal("1"),
) -> tuple[int, int] | tuple[None, None]:
    pricing = QWEN_PRICING_RMB_PER_1K.get(model)
    if not pricing:
        return None, None

    prompt_tokens = max(int(prompt_tokens), 0)
    completion_tokens = max(int(completion_tokens), 0)

    multiplier = Decimal(str(price_multiplier))
    if multiplier <= 0:
        multiplier = Decimal("1")

    cost_rmb = (
        (Decimal(prompt_tokens) * pricing["input"] / Decimal(1000))
        + (Decimal(completion_tokens) * pricing["output"] / Decimal(1000))
    ) * multiplier

    points_per_rmb = Decimal(int(getattr(settings, "POINTS_PER_RMB", 1000)))
    points = int((cost_rmb * points_per_rmb).to_integral_value(rounding=ROUND_UP))
    cost_rmb_milli = int((cost_rmb * Decimal(1000)).to_integral_value(rounding=ROUND_UP))
    return points, cost_rmb_milli


async def call_openai_compatible_chat(
    *,
    api_key: str,
    model_id: str,
    base_url: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 1200,
    temperature: float = 0.7,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    chat_url = _normalize_chat_completions_url(base_url)
    _validate_outbound_url(chat_url)

    timeout = httpx.Timeout(timeout_seconds, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            chat_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API 请求失败 ({resp.status_code})",
        )

    try:
        return resp.json()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM API 返回格式异常",
        ) from exc


def _extract_first_json(content: str) -> dict[str, Any] | None:
    if not content:
        return None
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class RoutedModel:
    model: str
    reason: str | None = None
    router_request_id: str | None = None


def _build_router_messages(user_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excerpt: list[dict[str, Any]] = []
    for msg in (user_messages or [])[-8:]:
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if isinstance(content, str):
            excerpt.append({"role": role, "content": content[:8000]})
        else:
            # 非纯文本（多模态）只保留文本部分用于路由
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text") or "")
                excerpt.append({"role": role, "content": "\n".join(parts)[:8000]})

    system_prompt = (
        "你是模型路由器。你的任务是根据用户请求选择最合适的通义千问模型。\n"
        "可选模型：qwen-flash（简单/低成本/快），qwen-plus（均衡默认），qwen-max（复杂推理/高质量），qwen-coder（代码与工具调用）。\n"
        "输出要求：只输出 JSON，不要 Markdown 或解释，格式：\n"
        "{\"model\":\"qwen-plus|qwen-max|qwen-flash|qwen-coder\",\"reason\":\"简短中文理由\"}\n"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({"messages": excerpt}, ensure_ascii=False)},
    ]


def build_router_messages(user_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _build_router_messages(user_messages)


async def route_qwen_model(
    *,
    api_key: str,
    base_url: str,
    user_messages: list[dict[str, Any]],
    request_id: str | None = None,
) -> RoutedModel:
    router_messages = _build_router_messages(user_messages)
    data = await call_openai_compatible_chat(
        api_key=api_key,
        model_id="qwen-flash",
        base_url=base_url,
        messages=router_messages,
        max_tokens=256,
        temperature=0.0,
        timeout_seconds=60.0,
    )

    content = ""
    try:
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    except Exception:
        content = ""

    parsed = _extract_first_json(content)
    model = (parsed or {}).get("model") if isinstance(parsed, dict) else None
    reason = (parsed or {}).get("reason") if isinstance(parsed, dict) else None
    model = model.strip() if isinstance(model, str) else ""
    if model not in QWEN_MODELS:
        model = "qwen-plus"
    return RoutedModel(model=model, reason=reason if isinstance(reason, str) else None, router_request_id=request_id)


def new_request_id() -> str:
    return str(uuid.uuid4())
