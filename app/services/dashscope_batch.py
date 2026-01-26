import ipaddress
import json
import re
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, status

from app.core.config import settings


def _normalize_base_url(base_url: str) -> str:
    resolved = base_url.strip() if isinstance(base_url, str) and base_url.strip() else ""
    if not resolved:
        raise ValueError("baseUrl 未配置")
    return re.sub(r"/+$", "", resolved)


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
        return False


def _validate_outbound_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("baseUrl 协议必须是 http/https")
    if not bool(getattr(settings, "ALLOW_PRIVATE_LLM_BASEURL", False)) and _is_private_host(parsed.hostname or ""):
        raise ValueError("baseUrl 不允许指向本地/内网地址")


def _files_url(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    url = f"{normalized}/files"
    _validate_outbound_url(url)
    return url


def _batches_url(base_url: str, batch_id: str | None = None) -> str:
    normalized = _normalize_base_url(base_url)
    url = f"{normalized}/batches" if not batch_id else f"{normalized}/batches/{batch_id}"
    _validate_outbound_url(url)
    return url


def _file_content_url(base_url: str, file_id: str) -> str:
    normalized = _normalize_base_url(base_url)
    url = f"{normalized}/files/{file_id}/content"
    _validate_outbound_url(url)
    return url


async def upload_batch_jsonl(
    *,
    api_key: str,
    base_url: str,
    filename: str,
    content: bytes | None = None,
    file_path: str | None = None,
) -> str:
    url = _files_url(base_url)
    timeout = httpx.Timeout(120.0, connect=10.0)
    if (content is None and not file_path) or (content is not None and file_path):
        raise ValueError("upload_batch_jsonl 需要且仅需要提供 content 或 file_path")

    if file_path:
        with open(file_path, "rb") as fin:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data={"purpose": "batch"},
                    files={"file": (filename, fin, "application/jsonl")},
                )
    else:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                data={"purpose": "batch"},
                files={"file": (filename, content, "application/jsonl")},
            )
    if resp.status_code >= 400:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Batch 文件上传失败 ({resp.status_code})")
    data = resp.json()
    file_id = data.get("id") if isinstance(data, dict) else None
    if not isinstance(file_id, str) or not file_id:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Batch 文件上传返回格式异常")
    return file_id


async def create_batch(
    *,
    api_key: str,
    base_url: str,
    input_file_id: str,
    endpoint: str,
    completion_window: str = "24h",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = _batches_url(base_url)
    timeout = httpx.Timeout(60.0, connect=10.0)
    payload: dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    if metadata:
        payload["metadata"] = metadata
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            url,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json=payload,
        )
    if resp.status_code >= 400:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Batch 任务创建失败 ({resp.status_code})")
    data = resp.json()
    if not isinstance(data, dict) or not data.get("id"):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Batch 任务创建返回格式异常")
    return data


async def retrieve_batch(
    *,
    api_key: str,
    base_url: str,
    batch_id: str,
) -> dict[str, Any]:
    url = _batches_url(base_url, batch_id)
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    if resp.status_code >= 400:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Batch 任务查询失败 ({resp.status_code})")
    data = resp.json()
    if not isinstance(data, dict) or data.get("id") != batch_id:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Batch 任务查询返回格式异常")
    return data


async def cancel_batch(
    *,
    api_key: str,
    base_url: str,
    batch_id: str,
) -> dict[str, Any]:
    url = f"{_batches_url(base_url, batch_id)}/cancel"
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers={"Authorization": f"Bearer {api_key}"})
    if resp.status_code >= 400:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Batch 任务取消失败 ({resp.status_code})")
    data = resp.json()
    if not isinstance(data, dict) or data.get("id") != batch_id:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Batch 任务取消返回格式异常")
    return data


async def iter_file_lines(
    *,
    api_key: str,
    base_url: str,
    file_id: str,
) -> AsyncIterator[str]:
    url = _file_content_url(base_url, file_id)
    timeout = httpx.Timeout(300.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url, headers={"Authorization": f"Bearer {api_key}"}) as resp:
            if resp.status_code >= 400:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Batch 文件下载失败 ({resp.status_code})",
                )
            async for line in resp.aiter_lines():
                yield line


async def sum_usage_from_output_jsonl(
    *,
    api_key: str,
    base_url: str,
    output_file_id: str,
) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    line_count = 0
    ok_count = 0

    async for line in iter_file_lines(api_key=api_key, base_url=base_url, file_id=output_file_id):
        raw = line.strip()
        if not raw:
            continue
        line_count += 1
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        response = obj.get("response")
        if not isinstance(response, dict):
            continue
        body = response.get("body")
        if not isinstance(body, dict):
            continue
        usage = body.get("usage")
        if not isinstance(usage, dict):
            continue
        ok_count += 1
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "line_count": line_count,
        "ok_count": ok_count,
    }
