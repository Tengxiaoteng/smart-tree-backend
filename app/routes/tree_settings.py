from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User, Topic, TreeSettings
from app.schemas.tree_settings import TreeSettingsUpdate, TreeSettingsResponse
import uuid

router = APIRouter()

# 默认设置
DEFAULT_SETTINGS = {
    "sortMode": "createdAt_desc",
    "displayConfig": {
        "showMastery": True,
        "showChildCount": False,
        "showDescription": False,
    },
    "visualConfig": {
        "nodeSpacing": 60,
        "nodeSize": "medium",
        "lineStyle": "curve",
    },
    "expandConfig": {
        "defaultExpandLevel": 3,
    },
}


def _merge_with_defaults(settings: TreeSettings) -> dict:
    """合并设置与默认值"""
    return {
        "id": settings.id,
        "topicId": settings.topicId,
        "sortMode": settings.sortMode or DEFAULT_SETTINGS["sortMode"],
        "displayConfig": {**DEFAULT_SETTINGS["displayConfig"], **(settings.displayConfig or {})},
        "visualConfig": {**DEFAULT_SETTINGS["visualConfig"], **(settings.visualConfig or {})},
        "expandConfig": {**DEFAULT_SETTINGS["expandConfig"], **(settings.expandConfig or {})},
        "createdAt": settings.createdAt,
        "updatedAt": settings.updatedAt,
    }


@router.get("/{topic_id}", response_model=TreeSettingsResponse)
async def get_tree_settings(
    topic_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取知识树设置"""
    # 验证 topic 归属
    topic = db.query(Topic).filter(
        Topic.id == topic_id, 
        Topic.userId == current_user.id
    ).first()
    if not topic:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识树不存在")
    
    settings = db.query(TreeSettings).filter(TreeSettings.topicId == topic_id).first()
    
    # 如果没有设置记录，返回默认值
    if not settings:
        return TreeSettingsResponse(
            id="",
            topicId=topic_id,
            sortMode=DEFAULT_SETTINGS["sortMode"],
            displayConfig=DEFAULT_SETTINGS["displayConfig"],
            visualConfig=DEFAULT_SETTINGS["visualConfig"],
            expandConfig=DEFAULT_SETTINGS["expandConfig"],
            createdAt=topic.createdAt,
            updatedAt=topic.updatedAt,
        )
    
    merged = _merge_with_defaults(settings)
    return TreeSettingsResponse(**merged)


@router.put("/{topic_id}", response_model=TreeSettingsResponse)
async def update_tree_settings(
    topic_id: str,
    data: TreeSettingsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新知识树设置（upsert）"""
    # 验证 topic 归属
    topic = db.query(Topic).filter(
        Topic.id == topic_id, 
        Topic.userId == current_user.id
    ).first()
    if not topic:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识树不存在")
    
    settings = db.query(TreeSettings).filter(TreeSettings.topicId == topic_id).first()
    
    if not settings:
        # 创建新设置
        settings = TreeSettings(
            id=str(uuid.uuid4()),
            topicId=topic_id
        )
        db.add(settings)
    
    # 更新字段
    if data.sortMode is not None:
        settings.sortMode = data.sortMode
    if data.displayConfig is not None:
        settings.displayConfig = data.displayConfig.model_dump()
    if data.visualConfig is not None:
        settings.visualConfig = data.visualConfig.model_dump()
    if data.expandConfig is not None:
        settings.expandConfig = data.expandConfig.model_dump()
    
    db.commit()
    db.refresh(settings)
    
    merged = _merge_with_defaults(settings)
    return TreeSettingsResponse(**merged)
