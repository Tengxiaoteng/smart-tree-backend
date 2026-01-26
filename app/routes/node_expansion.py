"""
节点扩展 API 路由
"""
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User, KnowledgeNode
from app.services.llm_context import resolve_llm_config
from app.services.node_expansion import expand_node_v2, ExpansionResult, GeneratedNodeContent
from app.services import credits as credits_service
from app.services.credits_pricing import FeatureType, calculate_fixed_points


router = APIRouter()


class ExpandNodeRequest(BaseModel):
    """扩展节点请求"""
    node_id: str = Field(..., description="要扩展的父节点 ID")
    topic_id: str = Field(..., description="知识树 ID")
    # 可选：覆盖 API 设置
    use_system_key: Optional[bool] = Field(None, description="是否使用系统 API")
    api_key: Optional[str] = Field(None, description="自定义 API Key")
    base_url: Optional[str] = Field(None, description="自定义 Base URL")
    model_id: Optional[str] = Field(None, description="自定义模型 ID")


class ExpandNodeResponse(BaseModel):
    """扩展节点响应"""
    success: bool
    parent_node_name: str
    existing_children: list[str] = Field(default_factory=list)
    planned_count: int = 0
    deduplicated_count: int = 0
    generated_nodes: list[GeneratedNodeContent] = Field(default_factory=list)
    error: Optional[str] = None
    
    class Config:
        from_attributes = True


@router.post("/expand", response_model=ExpandNodeResponse)
async def expand_node(
    request: ExpandNodeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    AI 智能扩展节点子节点
    
    流程：
    1. 获取父节点信息和已有子节点
    2. 规划需要的子节点（Planner）
    3. 排除与已有子节点重复的（Deduplicator）
    4. 并发生成节点内容（Workers）
    """
    # 获取父节点
    parent_node = db.query(KnowledgeNode).filter(
        KnowledgeNode.id == request.node_id,
        KnowledgeNode.topicId == request.topic_id,
        KnowledgeNode.userId == current_user.id,
    ).first()
    
    if not parent_node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="父节点不存在"
        )
    
    # 获取已有子节点
    existing_children = db.query(KnowledgeNode).filter(
        KnowledgeNode.parentId == request.node_id,
        KnowledgeNode.topicId == request.topic_id,
        KnowledgeNode.userId == current_user.id,
    ).all()
    existing_child_names = [child.name for child in existing_children]
    
    # 解析 LLM 配置
    try:
        resolved = resolve_llm_config(
            db,
            user_id=current_user.id,
            requested_use_system=request.use_system_key,
            override_api_key=request.api_key,
            override_base_url=request.base_url,
            override_model_id=request.model_id,
            override_routing=None,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"LLM 配置错误: {str(e)}"
        )
    
    # 积分扣除（仅系统模式）
    request_id = None
    fixed_points = 0
    if resolved.mode == "system":
        request_id = f"node_expand:{uuid.uuid4()}"
        fixed_points = calculate_fixed_points(FeatureType.NODE_EXPAND)

        try:
            credits_service.reserve_points(
                db, current_user.id,
                request_id=request_id,
                points=fixed_points,
                meta={"feature": "node_expand", "parentNode": parent_node.name},
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"积分不足: {str(e)}"
            )

    # 调用扩展服务
    try:
        result = await expand_node_v2(
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            model=resolved.model_id or "deepseek-chat",
            parent_name=parent_node.name,
            parent_description=parent_node.description or "",
            parent_difficulty=parent_node.difficulty or "beginner",
            parent_knowledge_type=parent_node.knowledgeType or "concept",
            parent_key_concepts=parent_node.keyConcepts or [],
            parent_learning_objectives=parent_node.learningObjectives or [],
            existing_children=existing_child_names,
            max_concurrency=5,
        )

        # 结算积分
        if resolved.mode == "system" and request_id:
            if result.success:
                credits_service.finalize_reservation(
                    db, current_user.id,
                    request_id=request_id,
                    reserved_points=fixed_points,
                    actual_points=fixed_points,
                    meta={"feature": "node_expand", "success": True, "generatedCount": len(result.generated_nodes)},
                )
            else:
                # 失败：退还积分
                credits_service.finalize_reservation(
                    db, current_user.id,
                    request_id=request_id,
                    reserved_points=fixed_points,
                    actual_points=0,
                    meta={"feature": "node_expand", "error": result.error},
                )

        return ExpandNodeResponse(
            success=result.success,
            parent_node_name=result.parent_node_name,
            existing_children=result.existing_children,
            planned_count=result.planned_count,
            deduplicated_count=result.deduplicated_count,
            generated_nodes=result.generated_nodes,
            error=result.error,
        )
    except Exception as e:
        # 异常：退还积分
        if resolved.mode == "system" and request_id:
            credits_service.finalize_reservation(
                db, current_user.id,
                request_id=request_id,
                reserved_points=fixed_points,
                actual_points=0,
                meta={"feature": "node_expand", "error": str(e)},
            )
        raise
