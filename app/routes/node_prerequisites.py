from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User

router = APIRouter()


@router.get("")
async def get_node_prerequisites(
    nodeId: str = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取节点先决条件 (暂未实现)"""
    return []


@router.post("")
async def create_node_prerequisite(
    data: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建节点先决条件 (暂未实现)"""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="功能暂未实现"
    )


@router.delete("")
async def delete_node_prerequisite(
    nodeId: str = Query(...),
    prerequisiteId: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除节点先决条件 (暂未实现)"""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="功能暂未实现"
    )
