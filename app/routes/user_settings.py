from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError
from app.core.database import get_db
from app.core.security import get_current_user
from app.core.encryption import encrypt_api_key, decrypt_api_key
from app.models import User, UserSettings
from app.schemas.user_settings import UserSettingsUpdate, UserSettingsResponse

router = APIRouter()


def _settings_to_response(settings: UserSettings) -> UserSettingsResponse:
    extras = settings.extras or {}
    remember_override = extras.get("remember") if isinstance(extras, dict) else None
    use_system_override = extras.get("useSystemKey") if isinstance(extras, dict) else None
    routing_override = extras.get("routing") if isinstance(extras, dict) else None
    inferred = bool(settings.apiKey or settings.modelId or settings.baseUrl)

    # 解密 API Key（兼容旧的明文数据）
    decrypted_api_key = decrypt_api_key(settings.apiKey) if settings.apiKey else ""

    return UserSettingsResponse(
        apiKey=decrypted_api_key or "",
        modelId=settings.modelId or "",
        baseUrl=settings.baseUrl or "",
        remember=bool(remember_override) if remember_override is not None else inferred,
        useSystemKey=bool(use_system_override) if isinstance(use_system_override, bool) else True,
        routing=(routing_override or "manual") if isinstance(routing_override, str) and routing_override else "manual",
    )


@router.get("", response_model=UserSettingsResponse)
async def get_user_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取用户设置"""
    try:
        settings = db.query(UserSettings).filter(UserSettings.userId == current_user.id).first()
    except ProgrammingError:
        # 数据库中可能还未创建 user_setting 表；保持接口可用，返回默认值
        return UserSettingsResponse(apiKey="", modelId="", baseUrl="", remember=False, useSystemKey=True)
    if not settings:
        return UserSettingsResponse(apiKey="", modelId="", baseUrl="", remember=False, useSystemKey=True)
    return _settings_to_response(settings)


@router.post("", response_model=UserSettingsResponse)
async def save_user_settings(
    data: UserSettingsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """保存用户设置（upsert）"""
    try:
        settings = db.query(UserSettings).filter(UserSettings.userId == current_user.id).first()
    except ProgrammingError:
        # 尝试在运行时创建表（兼容旧库/首次部署）
        try:
            UserSettings.__table__.create(bind=db.get_bind(), checkfirst=True)
        except SQLAlchemyError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"数据库缺少 user_setting 表且无法自动创建: {exc}",
            ) from exc
        settings = db.query(UserSettings).filter(UserSettings.userId == current_user.id).first()
    if data.remember is False:
        # 不保存 BYOK 凭据，但仍允许保留/更新 extras（例如 useSystemKey/routing）
        if not settings:
            settings = UserSettings(userId=current_user.id)
            db.add(settings)
        settings.apiKey = None
        settings.modelId = None
        settings.baseUrl = None

        extras = settings.extras if isinstance(settings.extras, dict) else {}
        extras = {**(extras or {}), "remember": False}
        if data.useSystemKey is not None:
            extras["useSystemKey"] = bool(data.useSystemKey)
        if data.routing is not None:
            routing = data.routing.strip().lower() if isinstance(data.routing, str) else ""
            if routing and routing not in ("auto", "manual"):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="routing 仅支持 auto/manual")
            extras["routing"] = routing or "manual"
        settings.extras = extras

        db.commit()
        db.refresh(settings)
        return _settings_to_response(settings)

    if not settings:
        settings = UserSettings(userId=current_user.id)
        db.add(settings)

    if data.apiKey is not None:
        # 加密存储 API Key
        settings.apiKey = encrypt_api_key(data.apiKey) if data.apiKey else None
    if data.modelId is not None:
        settings.modelId = data.modelId
    if data.baseUrl is not None:
        settings.baseUrl = data.baseUrl
    if data.remember is not None:
        # 兼容前端的 remember 开关，写入 extras 便于后续扩展
        extras = settings.extras if isinstance(settings.extras, dict) else {}
        extras = {**(extras or {}), "remember": bool(data.remember)}
        settings.extras = extras
    if data.useSystemKey is not None:
        extras = settings.extras if isinstance(settings.extras, dict) else {}
        extras = {**(extras or {}), "useSystemKey": bool(data.useSystemKey)}
        settings.extras = extras
    if data.routing is not None:
        routing = data.routing.strip().lower() if isinstance(data.routing, str) else ""
        if routing and routing not in ("auto", "manual"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="routing 仅支持 auto/manual")
        extras = settings.extras if isinstance(settings.extras, dict) else {}
        extras = {**(extras or {}), "routing": routing or "manual"}
        settings.extras = extras

    db.commit()
    db.refresh(settings)
    return _settings_to_response(settings)


@router.patch("", response_model=UserSettingsResponse)
async def update_user_settings(
    data: UserSettingsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新用户设置（upsert）"""
    try:
        settings = db.query(UserSettings).filter(UserSettings.userId == current_user.id).first()
    except ProgrammingError:
        # 尝试在运行时创建表（兼容旧库/首次部署）
        try:
            UserSettings.__table__.create(bind=db.get_bind(), checkfirst=True)
        except SQLAlchemyError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"数据库缺少 user_setting 表且无法自动创建: {exc}",
            ) from exc
        settings = db.query(UserSettings).filter(UserSettings.userId == current_user.id).first()
    if data.remember is False:
        if not settings:
            settings = UserSettings(userId=current_user.id)
            db.add(settings)
        settings.apiKey = None
        settings.modelId = None
        settings.baseUrl = None

        extras = settings.extras if isinstance(settings.extras, dict) else {}
        extras = {**(extras or {}), "remember": False}
        if data.useSystemKey is not None:
            extras["useSystemKey"] = bool(data.useSystemKey)
        if data.routing is not None:
            routing = data.routing.strip().lower() if isinstance(data.routing, str) else ""
            if routing and routing not in ("auto", "manual"):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="routing 仅支持 auto/manual")
            extras["routing"] = routing or "manual"
        settings.extras = extras

        db.commit()
        db.refresh(settings)
        return _settings_to_response(settings)

    if not settings:
        settings = UserSettings(userId=current_user.id)
        db.add(settings)

    if data.apiKey is not None:
        # 加密存储 API Key
        settings.apiKey = encrypt_api_key(data.apiKey) if data.apiKey else None
    if data.modelId is not None:
        settings.modelId = data.modelId
    if data.baseUrl is not None:
        settings.baseUrl = data.baseUrl
    if data.remember is not None:
        extras = settings.extras if isinstance(settings.extras, dict) else {}
        extras = {**(extras or {}), "remember": bool(data.remember)}
        settings.extras = extras
    if data.useSystemKey is not None:
        extras = settings.extras if isinstance(settings.extras, dict) else {}
        extras = {**(extras or {}), "useSystemKey": bool(data.useSystemKey)}
        settings.extras = extras
    if data.routing is not None:
        routing = data.routing.strip().lower() if isinstance(data.routing, str) else ""
        if routing and routing not in ("auto", "manual"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="routing 仅支持 auto/manual")
        extras = settings.extras if isinstance(settings.extras, dict) else {}
        extras = {**(extras or {}), "routing": routing or "manual"}
        settings.extras = extras

    db.commit()
    db.refresh(settings)
    return _settings_to_response(settings)
