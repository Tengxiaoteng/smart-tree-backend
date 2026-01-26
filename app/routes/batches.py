import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User, UserBatchJob
from app.schemas.batch import BatchChatCreateRequest, BatchJobListResponse, BatchJobResponse
from app.services import credits as credits_service
from app.services.dashscope_batch import (
    cancel_batch,
    create_batch,
    iter_file_lines,
    retrieve_batch,
    sum_usage_from_output_jsonl,
    upload_batch_jsonl,
)
from app.services.llm import QWEN_PRICING_RMB_PER_1K, calc_qwen_cost_points, estimate_prompt_tokens
from app.services.llm_context import resolve_llm_config

router = APIRouter()

ALLOWED_BATCH_ENDPOINTS = {"/v1/chat/completions", "/v1/chat/ds-test"}

SYSTEM_BATCH_PRESETS: dict[str, str] = {
    "test": "batch-test-model",
    "fast": "qwen-flash",
    "balanced": "qwen-plus",
    "flagship": "qwen-max",
    "code": "qwen-coder",
}

SYSTEM_BATCH_MODEL_TO_PRESET = {model: preset for preset, model in SYSTEM_BATCH_PRESETS.items()}


def _is_dashscope_base_url(base_url: str) -> bool:
    try:
        host = urlparse(base_url).hostname or ""
    except Exception:
        return False
    return "dashscope" in host


def _parse_completion_window(value: str) -> str:
    raw = value.strip() if isinstance(value, str) else ""
    match = re.fullmatch(r"(\d+)([hd])", raw)
    if not match:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="completionWindow 仅支持如 24h / 14d")
    amount = int(match.group(1))
    unit = match.group(2)
    hours = amount * 24 if unit == "d" else amount
    if hours < 24 or hours > 336:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="completionWindow 范围需在 24h - 336h")
    return f"{amount}{unit}"


def _resolve_batch_model(model: str, *, mode: str) -> tuple[str, str | None]:
    raw = model.strip() if isinstance(model, str) else ""
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="model 不能为空")

    if mode != "system":
        return raw, None

    if raw in SYSTEM_BATCH_PRESETS:
        return SYSTEM_BATCH_PRESETS[raw], raw

    if raw in SYSTEM_BATCH_MODEL_TO_PRESET:
        return raw, SYSTEM_BATCH_MODEL_TO_PRESET[raw]

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="系统 Batch 暂仅支持预置档位：test / fast / balanced / flagship / code",
    )


def _strip_model_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_model_keys(v) for k, v in value.items() if k != "model"}
    if isinstance(value, list):
        return [_strip_model_keys(v) for v in value]
    return value


def _sanitize_jsonl_line(line: str) -> str:
    try:
        obj = json.loads(line)
    except Exception:
        return line
    obj = _strip_model_keys(obj)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _job_to_response(job: UserBatchJob) -> BatchJobResponse:
    model = job.model
    if job.mode == "system":
        model = SYSTEM_BATCH_MODEL_TO_PRESET.get(job.model) or "system"
    return BatchJobResponse(
        id=job.id,
        status=job.status,
        batchId=job.batchId,
        inputFileId=job.inputFileId,
        outputFileId=job.outputFileId,
        errorFileId=job.errorFileId,
        model=model,
        endpoint=job.endpoint,
        completionWindow=job.completionWindow,
        mode=job.mode,
        createdAt=job.createdAt,
        updatedAt=job.updatedAt,
        reservedPoints=job.reservedPoints,
        chargedPoints=job.chargedPoints,
        promptTokens=job.promptTokens,
        completionTokens=job.completionTokens,
        totalTokens=job.totalTokens,
        costRmbMilli=job.costRmbMilli,
        billedAt=job.billedAt,
        requestCounts=job.requestCounts,
        metadata=job.metadata,
    )


def _resolve_job_key(db: Session, job: UserBatchJob, *, override_api_key: str | None = None) -> tuple[str, str]:
    if job.mode == "system":
        api_key = getattr(settings, "DASHSCOPE_API_KEY", None)
        if not isinstance(api_key, str) or not api_key.strip():
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="系统 DASHSCOPE_API_KEY 未配置")
        return api_key.strip(), job.baseUrl

    if override_api_key and override_api_key.strip():
        return override_api_key.strip(), job.baseUrl

    resolved = resolve_llm_config(
        db,
        user_id=job.userId,
        requested_use_system=False,
        override_api_key=None,
        override_base_url=job.baseUrl,
        override_model_id=None,
        override_routing=None,
    )
    if resolved.mode != "byok":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="该 Batch 任务为 BYOK，请提供 apiKey 或先在设置中保存")
    return resolved.api_key, job.baseUrl


@router.post("", response_model=BatchJobResponse)
async def create_chat_batch(
    payload: BatchChatCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    completion_window = _parse_completion_window(payload.completionWindow)

    temperature = float(payload.temperature if payload.temperature is not None else 0.7)
    if temperature < 0:
        temperature = 0.0
    if temperature > 2:
        temperature = 2.0

    max_tokens = int(payload.max_tokens or 1024)
    if max_tokens < 1:
        max_tokens = 1
    if max_tokens > 8192:
        max_tokens = 8192

    items = list(payload.items or [])
    if not items and payload.prompts:
        prompts = [p for p in (payload.prompts or []) if isinstance(p, str) and p.strip()]
        system_prompt = payload.systemPrompt.strip() if isinstance(payload.systemPrompt, str) else "You are a helpful assistant."
        for idx, prompt in enumerate(prompts, start=1):
            items.append(
                {
                    "customId": str(idx),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt.strip()},
                    ],
                }
            )

    if not items:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="items/prompts 不能为空")
    if len(items) > 50_000:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="单任务最多 50,000 条请求")

    # Resolve auth (system vs byok)
    resolved = resolve_llm_config(
        db,
        user_id=current_user.id,
        requested_use_system=payload.useSystemKey if isinstance(payload.useSystemKey, bool) else None,
        override_api_key=payload.apiKey.strip() if isinstance(payload.apiKey, str) and payload.apiKey.strip() else None,
        override_base_url=payload.baseUrl.strip() if isinstance(payload.baseUrl, str) and payload.baseUrl.strip() else None,
        override_model_id=None,
        override_routing=None,
    )

    actual_model, _ = _resolve_batch_model(payload.model, mode=resolved.mode)

    endpoint = payload.endpoint.strip() if isinstance(payload.endpoint, str) and payload.endpoint.strip() else None
    if not endpoint:
        endpoint = "/v1/chat/ds-test" if actual_model == "batch-test-model" else "/v1/chat/completions"
    if endpoint not in ALLOWED_BATCH_ENDPOINTS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="endpoint 不支持（仅 /v1/chat/completions 或 /v1/chat/ds-test）")

    if not _is_dashscope_base_url(resolved.base_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch 当前仅支持系统已集成的 OpenAI 兼容 Batch 服务",
        )

    job_id = str(uuid.uuid4())
    job = UserBatchJob(
        id=job_id,
        userId=current_user.id,
        provider="dashscope",
        mode=resolved.mode,
        baseUrl=resolved.base_url,
        endpoint=endpoint,
        model=actual_model,
        completionWindow=completion_window,
        batchId="",
        inputFileId="",
        status="validating",
        metadata=payload.metadata,
    )

    reserved_points: int | None = None
    reservation_request_id = f"batch:{job.id}"

    tmp_path: str | None = None
    try:
        # Reserve credits for system mode (batch is 50% price)
        if resolved.mode == "system" and job.model in QWEN_PRICING_RMB_PER_1K:
            credits_service.refund_stale_reservations(db, current_user.id)
            est_prompt = 0
            custom_ids: set[str] = set()
            for idx, item in enumerate(items, start=1):
                custom_id = item.customId if hasattr(item, "customId") else item.get("customId")  # type: ignore[attr-defined]
                messages = item.messages if hasattr(item, "messages") else item.get("messages")  # type: ignore[attr-defined]
                if not isinstance(messages, list) or not messages:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="每条请求必须包含 messages")
                cid = str(custom_id).strip() if custom_id else str(idx)
                if cid in custom_ids:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"customId 重复：{cid}")
                custom_ids.add(cid)
                est_prompt += estimate_prompt_tokens(messages)

            est_completion = max_tokens * len(items)
            points, _ = calc_qwen_cost_points(
                model=job.model,
                prompt_tokens=est_prompt,
                completion_tokens=est_completion,
                price_multiplier="0.5",
            )
            reserved_points = max(int(points or 1), 1)
            credits_service.reserve_points(
                db,
                current_user.id,
                request_id=reservation_request_id,
                points=reserved_points,
                meta={
                    "stage": "batch_create",
                    "model": job.model,
                    "endpoint": endpoint,
                    "completionWindow": completion_window,
                    "discount": 0.5,
                    "itemCount": len(items),
                },
            )

        # Build JSONL into a temp file (avoid holding big payload in memory)
        tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".jsonl")
        tmp_path = tmp.name
        with tmp:
            used_custom_ids: set[str] = set()
            for idx, item in enumerate(items, start=1):
                custom_id = item.customId if hasattr(item, "customId") else item.get("customId")  # type: ignore[attr-defined]
                messages = item.messages if hasattr(item, "messages") else item.get("messages")  # type: ignore[attr-defined]
                cid = str(custom_id).strip() if custom_id else str(idx)
                if cid in used_custom_ids:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"customId 重复：{cid}")
                used_custom_ids.add(cid)

                body: dict[str, Any] = {
                    "model": job.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if payload.enable_thinking is not None:
                    body["enable_thinking"] = bool(payload.enable_thinking)
                if payload.thinking_budget is not None:
                    body["thinking_budget"] = int(payload.thinking_budget)

                line_obj = {"custom_id": cid, "method": "POST", "url": endpoint, "body": body}
                tmp.write(json.dumps(line_obj, ensure_ascii=False, separators=(",", ":")) + "\n")

        # Basic guardrails
        size = os.path.getsize(tmp_path)
        if size > 500 * 1024 * 1024:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="输入文件超过 500MB 上限")

        filename = f"batch_{job.id}.jsonl"
        file_id = await upload_batch_jsonl(
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            filename=filename,
            file_path=tmp_path,
        )

        batch_data = await create_batch(
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            input_file_id=file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=payload.metadata or None,
        )

        job.batchId = str(batch_data.get("id"))
        job.inputFileId = file_id
        job.outputFileId = batch_data.get("output_file_id")
        job.errorFileId = batch_data.get("error_file_id")
        job.status = str(batch_data.get("status") or "validating")
        job.requestCounts = batch_data.get("request_counts")
        job.providerData = batch_data
        job.reservedPoints = reserved_points

        db.add(job)
        db.commit()
        db.refresh(job)
        return _job_to_response(job)

    except Exception as exc:
        if reserved_points:
            try:
                credits_service.finalize_reservation(
                    db,
                    current_user.id,
                    request_id=reservation_request_id,
                    reserved_points=reserved_points,
                    actual_points=0,
                    model=job.model,
                    meta={"stage": "batch_create", "error": str(exc)},
                )
            except Exception:
                pass
        raise
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@router.get("", response_model=BatchJobListResponse)
async def list_batches(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows: list[UserBatchJob] = (
        db.query(UserBatchJob)
        .filter(UserBatchJob.userId == current_user.id)
        .order_by(UserBatchJob.updatedAt.desc(), UserBatchJob.createdAt.desc())
        .limit(limit)
        .all()
    )
    return BatchJobListResponse(items=[_job_to_response(row) for row in rows])


@router.get("/{job_id}", response_model=BatchJobResponse)
async def get_batch_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(UserBatchJob).filter(UserBatchJob.id == job_id, UserBatchJob.userId == current_user.id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch 任务不存在")
    return _job_to_response(job)


@router.post("/{job_id}/refresh", response_model=BatchJobResponse)
async def refresh_batch_job(
    job_id: str,
    payload: dict[str, Any] | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(UserBatchJob).filter(UserBatchJob.id == job_id, UserBatchJob.userId == current_user.id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch 任务不存在")

    override_api_key = None
    if isinstance(payload, dict):
        override_api_key = payload.get("apiKey")

    api_key, base_url = _resolve_job_key(db, job, override_api_key=override_api_key)

    batch_data = await retrieve_batch(api_key=api_key, base_url=base_url, batch_id=job.batchId)
    job.status = str(batch_data.get("status") or job.status)
    job.outputFileId = batch_data.get("output_file_id")
    job.errorFileId = batch_data.get("error_file_id")
    job.requestCounts = batch_data.get("request_counts")
    job.providerData = batch_data
    db.add(job)
    db.commit()
    db.refresh(job)

    terminal = job.status in {"completed", "failed", "expired", "cancelled"}
    can_bill = job.mode == "system" and job.billedAt is None and job.reservedPoints and job.model in QWEN_PRICING_RMB_PER_1K
    if terminal and can_bill:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if job.outputFileId:
            usage = await sum_usage_from_output_jsonl(api_key=api_key, base_url=base_url, output_file_id=job.outputFileId)

        points, cost_milli = calc_qwen_cost_points(
            model=job.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            price_multiplier="0.5",
        )
        actual_points = int(points or 0)

        credits_service.finalize_reservation(
            db,
            current_user.id,
            request_id=f"batch:{job.id}",
            reserved_points=int(job.reservedPoints or 0),
            actual_points=actual_points,
            model=job.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            cost_rmb_milli=cost_milli,
            meta={
                "stage": "batch_settle",
                "batchId": job.batchId,
                "status": job.status,
                "discount": 0.5,
            },
        )

        job.promptTokens = int(usage.get("prompt_tokens") or 0)
        job.completionTokens = int(usage.get("completion_tokens") or 0)
        job.totalTokens = int(usage.get("total_tokens") or 0)
        job.costRmbMilli = cost_milli
        job.chargedPoints = actual_points
        job.billedAt = datetime.utcnow()
        db.add(job)
        db.commit()
        db.refresh(job)

    return _job_to_response(job)


@router.post("/{job_id}/cancel", response_model=BatchJobResponse)
async def cancel_batch_job(
    job_id: str,
    payload: dict[str, Any] | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(UserBatchJob).filter(UserBatchJob.id == job_id, UserBatchJob.userId == current_user.id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch 任务不存在")

    override_api_key = None
    if isinstance(payload, dict):
        override_api_key = payload.get("apiKey")
    api_key, base_url = _resolve_job_key(db, job, override_api_key=override_api_key)

    batch_data = await cancel_batch(api_key=api_key, base_url=base_url, batch_id=job.batchId)
    job.status = str(batch_data.get("status") or job.status)
    job.providerData = batch_data
    db.add(job)
    db.commit()
    db.refresh(job)
    return _job_to_response(job)


@router.post("/{job_id}/output")
async def download_batch_output(
    job_id: str,
    payload: dict[str, Any] | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(UserBatchJob).filter(UserBatchJob.id == job_id, UserBatchJob.userId == current_user.id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch 任务不存在")
    if not job.outputFileId:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="该任务暂无 output_file_id")

    override_api_key = None
    if isinstance(payload, dict):
        override_api_key = payload.get("apiKey")
    api_key, base_url = _resolve_job_key(db, job, override_api_key=override_api_key)

    async def _iter():
        async for line in iter_file_lines(api_key=api_key, base_url=base_url, file_id=job.outputFileId):
            safe_line = _sanitize_jsonl_line(line) if job.mode == "system" else line
            yield (safe_line + "\n").encode("utf-8")

    return StreamingResponse(_iter(), media_type="application/jsonl")


@router.post("/{job_id}/error")
async def download_batch_error(
    job_id: str,
    payload: dict[str, Any] | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = db.query(UserBatchJob).filter(UserBatchJob.id == job_id, UserBatchJob.userId == current_user.id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch 任务不存在")
    if not job.errorFileId:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="该任务暂无 error_file_id")

    override_api_key = None
    if isinstance(payload, dict):
        override_api_key = payload.get("apiKey")
    api_key, base_url = _resolve_job_key(db, job, override_api_key=override_api_key)

    async def _iter():
        async for line in iter_file_lines(api_key=api_key, base_url=base_url, file_id=job.errorFileId):
            safe_line = _sanitize_jsonl_line(line) if job.mode == "system" else line
            yield (safe_line + "\n").encode("utf-8")

    return StreamingResponse(_iter(), media_type="application/jsonl")
