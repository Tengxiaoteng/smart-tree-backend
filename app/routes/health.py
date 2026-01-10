# 后端健康检查端点
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.database import get_db
import time

router = APIRouter()


@router.get("")
async def health_check(db: Session = Depends(get_db)):
    """
    健康检查端点 - 用于负载均衡器/监控系统
    返回服务状态和数据库连接状态
    """
    start = time.time()
    
    # 检查数据库连接
    db_status = "healthy"
    db_latency_ms = 0
    try:
        db_start = time.time()
        db.execute(text("SELECT 1"))
        db_latency_ms = round((time.time() - db_start) * 1000, 2)
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    total_latency_ms = round((time.time() - start) * 1000, 2)
    
    status = "healthy" if db_status == "healthy" else "degraded"
    
    return {
        "status": status,
        "timestamp": time.time(),
        "checks": {
            "database": {
                "status": db_status,
                "latency_ms": db_latency_ms
            }
        },
        "latency_ms": total_latency_ms
    }


@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """
    就绪检查 - Kubernetes readiness probe
    只有当服务完全准备好接收流量时才返回 200
    """
    try:
        db.execute(text("SELECT 1"))
        return {"ready": True}
    except Exception:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """
    存活检查 - Kubernetes liveness probe
    只要进程还在运行就返回 200
    """
    return {"alive": True}

