"""
知识树生成 API 路由
"""
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User
from app.services.tree_generation import (
    generate_knowledge_tree,
    generate_knowledge_tree_stream,
    TreeGenerationResult,
    ProgressEvent,
    KnowledgeTreeSchema,
)


router = APIRouter()


class TreeGenerationRequest(BaseModel):
    """知识树生成请求"""
    content: str
    useSystemKey: bool | None = None


@router.post("/generate", response_model=TreeGenerationResult)
async def generate_tree(
    payload: TreeGenerationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    生成知识树

    - 官方 API (useSystemKey=true): 使用 V3 两阶段并行架构（更快、更丰富）
    - 用户自配 API (useSystemKey=false): 使用单次生成方式
    """
    if not payload.content or not payload.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="内容不能为空",
        )

    result = await generate_knowledge_tree(
        db=db,
        user_id=current_user.id,
        content=payload.content.strip(),
        use_system=payload.useSystemKey,
    )

    return result


@router.post("/generate/stream")
async def generate_tree_stream(
    payload: TreeGenerationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    流式生成知识树（带进度反馈）

    返回 SSE 流，每个事件格式：
    - progress 事件: {"type": "progress", "data": {...}}
    - result 事件: {"type": "result", "data": {...}}
    """
    if not payload.content or not payload.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="内容不能为空",
        )

    async def event_generator():
        async for event in generate_knowledge_tree_stream(
            db=db,
            user_id=current_user.id,
            content=payload.content.strip(),
            use_system=payload.useSystemKey,
        ):
            if isinstance(event, ProgressEvent):
                data = {
                    "type": "progress",
                    "data": event.model_dump(),
                }
            elif isinstance(event, TreeGenerationResult):
                data = {
                    "type": "result",
                    "data": event.model_dump(by_alias=True),
                }
            else:
                continue

            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
