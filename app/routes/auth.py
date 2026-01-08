from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
import logging
from app.core.database import get_db
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user,
)
from app.models import User
from app.services.credits import grant_signup_bonus
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    nickname: str | None = None


class AuthResponse(BaseModel):
    id: str
    username: str
    nickname: str | None = None
    token: str


class UpdateMeRequest(BaseModel):
    nickname: str | None = None
    username: str | None = None


@router.post("/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    """用户登录"""
    username = request.username.strip() if request.username else ""
    password = request.password.strip() if request.password else ""
    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名或密码不能为空")

    logger.debug("用户登录请求: %s", username)
    user = db.query(User).filter(User.username == username).first()

    if not user or not verify_password(password, user.password):
        logger.info("登录失败: 用户名或密码错误 (%s)", username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    # 创建 JWT token
    token = create_access_token({"sub": user.id})
    logger.debug("登录成功: userId=%s username=%s", user.id, user.username)

    return AuthResponse(
        id=user.id,
        username=user.username,
        nickname=user.nickname,
        token=token,
    )


@router.post("/register", response_model=AuthResponse)
async def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    """用户注册"""
    username = request.username.strip() if request.username else ""
    password = request.password.strip() if request.password else ""
    nickname = request.nickname.strip() if request.nickname else None

    if not username or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名或密码不能为空")
    if len(username) < 3 or len(username) > 20:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名长度需为 3-20 个字符")
    if len(password) < 6:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="密码至少 6 个字符")

    # 检查用户名是否已存在
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在",
        )

    # 创建新用户
    user = User(
        id=str(uuid.uuid4()),
        username=username,
        password=get_password_hash(password),
        nickname=nickname or username,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # 新用户赠送积分（默认 2000，可通过环境变量 NEW_USER_BONUS_POINTS 调整）
    try:
        grant_signup_bonus(db, user.id)
    except Exception as exc:
        # 不阻断注册：即使积分初始化失败，也允许用户登录；后续可由管理员补发
        logger.exception("新用户积分初始化失败: userId=%s err=%s", user.id, exc)

    # 创建 JWT token
    token = create_access_token({"sub": user.id})

    return AuthResponse(
        id=user.id,
        username=user.username,
        nickname=user.nickname,
        token=token,
    )


@router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """获取当前登录用户信息"""
    return {
        "data": {
            "id": current_user.id,
            "username": current_user.username,
            "nickname": current_user.nickname,
        }
    }


@router.patch("/me")
async def update_current_user_info(
    data: UpdateMeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """更新当前登录用户基本信息"""
    if data.username is not None:
        new_username = data.username.strip() if data.username else ""
        if new_username != current_user.username:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="用户名不可修改")

    if data.nickname is not None:
        new_nickname = data.nickname.strip() if data.nickname else None
        if new_nickname != current_user.nickname:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="昵称不可修改")

    return {
        "data": {
            "id": current_user.id,
            "username": current_user.username,
            "nickname": current_user.nickname,
        }
    }
