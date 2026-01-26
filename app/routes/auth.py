from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, field_validator
import logging
import re
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

# 验证规则常量
USERNAME_MIN_LENGTH = 3
USERNAME_MAX_LENGTH = 20
PASSWORD_MIN_LENGTH = 6
PASSWORD_MAX_LENGTH = 18
NICKNAME_MAX_LENGTH = 30

# 用户名只允许字母、数字、下划线
USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]+$')


class LoginRequest(BaseModel):
    username: str
    password: str

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('用户名不能为空')
        if len(v) > USERNAME_MAX_LENGTH:
            raise ValueError(f'用户名长度不能超过 {USERNAME_MAX_LENGTH} 个字符')
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not v:
            raise ValueError('密码不能为空')
        if len(v) > PASSWORD_MAX_LENGTH:
            raise ValueError(f'密码长度不能超过 {PASSWORD_MAX_LENGTH} 个字符')
        return v


class RegisterRequest(BaseModel):
    username: str
    password: str
    confirmPassword: str | None = None
    nickname: str | None = None

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('用户名不能为空')
        if len(v) < USERNAME_MIN_LENGTH:
            raise ValueError(f'用户名长度不能少于 {USERNAME_MIN_LENGTH} 个字符')
        if len(v) > USERNAME_MAX_LENGTH:
            raise ValueError(f'用户名长度不能超过 {USERNAME_MAX_LENGTH} 个字符')
        if not USERNAME_PATTERN.match(v):
            raise ValueError('用户名只能包含字母、数字和下划线')
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not v:
            raise ValueError('密码不能为空')
        if len(v) < PASSWORD_MIN_LENGTH:
            raise ValueError(f'密码长度不能少于 {PASSWORD_MIN_LENGTH} 个字符')
        if len(v) > PASSWORD_MAX_LENGTH:
            raise ValueError(f'密码长度不能超过 {PASSWORD_MAX_LENGTH} 个字符')
        # 密码强度检查：至少包含字母和数字
        if not re.search(r'[a-zA-Z]', v):
            raise ValueError('密码必须包含至少一个字母')
        if not re.search(r'[0-9]', v):
            raise ValueError('密码必须包含至少一个数字')
        return v

    @field_validator('nickname')
    @classmethod
    def validate_nickname(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if len(v) > NICKNAME_MAX_LENGTH:
            raise ValueError(f'昵称长度不能超过 {NICKNAME_MAX_LENGTH} 个字符')
        return v


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
    # Pydantic 已验证并 strip，直接使用
    username = request.username
    password = request.password

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
    # Pydantic 已验证并 strip，直接使用
    username = request.username
    password = request.password
    confirm_password = request.confirmPassword
    nickname = request.nickname

    # 验证确认密码
    if confirm_password is not None and password != confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="两次输入的密码不一致",
        )

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
        db.commit()
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
    # 获取头像URL（从 UserProfile 中获取）
    avatar_url = None
    if current_user.profile:
        avatar_url = current_user.profile.avatarUrl

    return {
        "data": {
            "id": current_user.id,
            "username": current_user.username,
            "nickname": current_user.nickname,
            "avatarUrl": avatar_url,
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
