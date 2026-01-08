import json
import re
from datetime import datetime
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from sqlalchemy.exc import ProgrammingError, OperationalError, SQLAlchemyError

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User, UserProfile, Topic, KnowledgeNode, Material
from app.schemas.user_profile import UserProfileUpdate, UserProfileResponse, SeedlingRefreshRequest
from app.services.llm_context import resolve_llm_config
from app.services.llm import call_openai_compatible_chat, calc_qwen_cost_points, estimate_prompt_tokens, new_request_id
from app.services import credits as credits_service

router = APIRouter()


def _profile_to_response(profile: UserProfile | None, user_id: str) -> UserProfileResponse:
    if not profile:
        return UserProfileResponse(userId=user_id)
    return UserProfileResponse.model_validate(profile, from_attributes=True)

def _normalize_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _normalize_chat_completions_url(base_url: str) -> str:
    resolved = base_url.strip() if isinstance(base_url, str) and base_url.strip() else ""
    if not resolved:
        raise ValueError("baseUrl 未配置")

    normalized = re.sub(r"/+$", "", resolved)
    return normalized if "chat/completions" in normalized else f"{normalized}/chat/completions"


def _extract_json(content: str) -> dict | None:
    if not content:
        return None
    # 允许模型输出包含多余文本，尽量提取第一段 JSON 对象
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _build_default_seedling_portrait(stats: dict) -> dict:
    # 无 AI 时的保底画像：基于简单统计生成可展示的数据结构
    node_count = int(stats.get("nodeCount") or 0)
    avg_mastery = float(stats.get("avgMastery") or 0)

    stage = "seed"
    if node_count >= 30 or avg_mastery >= 70:
        stage = "sapling"
    elif node_count >= 10 or avg_mastery >= 40:
        stage = "sprout"

    level = min(max(int(round(avg_mastery / 10)), 1), 10)
    health = int(min(max(round(avg_mastery), 0), 100))

    return {
        "version": 1,
        "stage": stage,  # seed | sprout | sapling | tree
        "level": level,
        "health": health,
        "traits": ["稳扎稳打", "持续生长"] if node_count else ["从一粒种子开始"],
        "summary": "你的知识树正在生长：保持规律练习与复盘，成长会更快。",
        "nextActions": ["每天 15 分钟复习", "把错题转成薄弱点目标", "给节点补充资料与例题"],
        "stats": stats,
        "visual": {"theme": "green", "accessory": "leaf"},
    }


async def _call_openai_compatible_chat(
    *,
    api_key: str,
    model_id: str,
    base_url: str,
    messages: list[dict],
    max_tokens: int = 1200,
    temperature: float = 0.7,
) -> str:
    chat_url = _normalize_chat_completions_url(base_url)
    parsed = urlparse(chat_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("baseUrl 协议必须是 http/https")

    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            chat_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API 请求失败 ({resp.status_code})",
        )

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"] or ""
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM API 返回格式异常",
        )


def _collect_learning_stats(db: Session, user_id: str) -> dict:
    topic_count = db.query(Topic).filter(Topic.userId == user_id).count()
    node_query = db.query(KnowledgeNode).filter(KnowledgeNode.userId == user_id)
    node_count = node_query.count()
    material_count = db.query(Material).filter(Material.userId == user_id).count()

    masteries = [n.mastery or 0 for n in node_query.all()]
    avg_mastery = round(sum(masteries) / len(masteries), 2) if masteries else 0.0

    last_node_update = (
        db.query(KnowledgeNode.updatedAt)
        .filter(KnowledgeNode.userId == user_id)
        .order_by(KnowledgeNode.updatedAt.desc())
        .first()
    )
    last_material_update = (
        db.query(Material.updatedAt)
        .filter(Material.userId == user_id)
        .order_by(Material.updatedAt.desc())
        .first()
    )

    def _dt(val):
        return val[0].isoformat() if val and val[0] else None

    return {
        "topicCount": topic_count,
        "nodeCount": node_count,
        "materialCount": material_count,
        "avgMastery": avg_mastery,
        "lastNodeUpdatedAt": _dt(last_node_update),
        "lastMaterialUpdatedAt": _dt(last_material_update),
    }

def _ensure_user_profile_schema(db: Session) -> None:
    """确保 user_profile 表及字段存在（兼容旧库/首次部署）。"""
    engine = db.get_bind()
    inspector = inspect(engine)
    is_mysql = engine.dialect.name == "mysql"

    table_name = "user_profile"
    tables = set(inspector.get_table_names())

    if table_name not in tables:
        UserProfile.__table__.create(bind=engine, checkfirst=True)
        return

    existing_cols = {col["name"] for col in inspector.get_columns(table_name)}
    alter_statements: list[str] = []

    def _add(stmt_mysql: str, stmt_other: str) -> None:
        alter_statements.append(stmt_mysql if is_mysql else stmt_other)

    if "email" not in existing_cols:
        _add(
            "ALTER TABLE user_profile ADD COLUMN email VARCHAR(255) NULL",
            "ALTER TABLE user_profile ADD COLUMN email VARCHAR(255)",
        )
    if "avatarUrl" not in existing_cols:
        _add(
            "ALTER TABLE user_profile ADD COLUMN avatarUrl VARCHAR(1024) NULL",
            "ALTER TABLE user_profile ADD COLUMN avatarUrl VARCHAR(1024)",
        )
    if "bio" not in existing_cols:
        _add(
            "ALTER TABLE user_profile ADD COLUMN bio LONGTEXT NULL",
            "ALTER TABLE user_profile ADD COLUMN bio TEXT",
        )
    if "timezone" not in existing_cols:
        _add(
            "ALTER TABLE user_profile ADD COLUMN timezone VARCHAR(64) NULL",
            "ALTER TABLE user_profile ADD COLUMN timezone VARCHAR(64)",
        )
    if "language" not in existing_cols:
        _add(
            "ALTER TABLE user_profile ADD COLUMN language VARCHAR(32) NULL",
            "ALTER TABLE user_profile ADD COLUMN language VARCHAR(32)",
        )
    if "education" not in existing_cols:
        _add(
            "ALTER TABLE user_profile ADD COLUMN education JSON NULL",
            "ALTER TABLE user_profile ADD COLUMN education TEXT",
        )
    if "preferences" not in existing_cols:
        _add(
            "ALTER TABLE user_profile ADD COLUMN preferences JSON NULL",
            "ALTER TABLE user_profile ADD COLUMN preferences TEXT",
        )
    if "learningHabits" not in existing_cols:
        _add(
            "ALTER TABLE user_profile ADD COLUMN learningHabits JSON NULL",
            "ALTER TABLE user_profile ADD COLUMN learningHabits TEXT",
        )
    if "seedlingPortrait" not in existing_cols:
        _add(
            "ALTER TABLE user_profile ADD COLUMN seedlingPortrait JSON NULL",
            "ALTER TABLE user_profile ADD COLUMN seedlingPortrait TEXT",
        )
    if "portraitUpdatedAt" not in existing_cols:
        _add(
            "ALTER TABLE user_profile ADD COLUMN portraitUpdatedAt DATETIME NULL",
            "ALTER TABLE user_profile ADD COLUMN portraitUpdatedAt DATETIME",
        )
    if "createdAt" not in existing_cols:
        _add(
            "ALTER TABLE user_profile ADD COLUMN createdAt DATETIME NULL",
            "ALTER TABLE user_profile ADD COLUMN createdAt DATETIME",
        )
    if "updatedAt" not in existing_cols:
        _add(
            "ALTER TABLE user_profile ADD COLUMN updatedAt DATETIME NULL",
            "ALTER TABLE user_profile ADD COLUMN updatedAt DATETIME",
        )

    if not alter_statements:
        return

    with engine.begin() as conn:
        for stmt in alter_statements:
            conn.execute(text(stmt))


@router.get("", response_model=UserProfileResponse)
async def get_user_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取用户资料（包含树苗画像等可扩展信息）"""
    try:
        _ensure_user_profile_schema(db)
        profile = db.query(UserProfile).filter(UserProfile.userId == current_user.id).first()
    except (ProgrammingError, OperationalError, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="数据库缺少 user_profile 表或字段，请执行迁移/检查数据库权限后重试",
        ) from exc
    return _profile_to_response(profile, current_user.id)


@router.post("", response_model=UserProfileResponse)
async def upsert_user_profile(
    data: UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """保存用户资料（upsert）"""
    try:
        _ensure_user_profile_schema(db)
        profile = db.query(UserProfile).filter(UserProfile.userId == current_user.id).first()
    except (ProgrammingError, OperationalError, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="数据库缺少 user_profile 表或字段，请执行迁移/检查数据库权限后重试",
        ) from exc
    if not profile:
        profile = UserProfile(userId=current_user.id)
        db.add(profile)

    if data.email is not None:
        profile.email = _normalize_str(data.email)
    if data.avatarUrl is not None:
        profile.avatarUrl = _normalize_str(data.avatarUrl)
    if data.bio is not None:
        profile.bio = _normalize_str(data.bio)
    if data.timezone is not None:
        profile.timezone = _normalize_str(data.timezone)
    if data.language is not None:
        profile.language = _normalize_str(data.language)
    if data.education is not None:
        profile.education = data.education
    if data.preferences is not None:
        profile.preferences = data.preferences
    if data.learningHabits is not None:
        profile.learningHabits = data.learningHabits
    if data.seedlingPortrait is not None:
        profile.seedlingPortrait = data.seedlingPortrait
        profile.portraitUpdatedAt = datetime.utcnow()

    db.commit()
    db.refresh(profile)
    return _profile_to_response(profile, current_user.id)


@router.patch("", response_model=UserProfileResponse)
async def update_user_profile(
    data: UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """更新用户资料（upsert）"""
    return await upsert_user_profile(data, db, current_user)


@router.post("/seedling/refresh", response_model=UserProfileResponse)
async def refresh_seedling_portrait(
    payload: SeedlingRefreshRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """根据学习习惯快照刷新树苗画像（使用用户的 LLM 设置）"""
    resolved = resolve_llm_config(
        db,
        user_id=current_user.id,
        requested_use_system=None,
        override_api_key=None,
        override_base_url=None,
        override_model_id=None,
        override_routing=None,
    )

    api_key = resolved.api_key
    base_url = resolved.base_url
    model_id = resolved.model_id or ("qwen-plus" if resolved.mode == "system" else None)
    use_system = resolved.mode == "system"
    if not model_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请先在「API 设置」中配置 modelId")

    try:
        _ensure_user_profile_schema(db)
        profile = db.query(UserProfile).filter(UserProfile.userId == current_user.id).first()
    except (ProgrammingError, OperationalError, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="数据库缺少 user_profile 表或字段，请执行迁移/检查数据库权限后重试",
        ) from exc
    if not profile:
        profile = UserProfile(userId=current_user.id)
        db.add(profile)

    stats = _collect_learning_stats(db, current_user.id)
    merged_snapshot = {
        "stats": stats,
        "clientSnapshot": payload.learningSnapshot or {},
        "profile": {
            "timezone": profile.timezone,
            "language": profile.language,
            "education": profile.education,
            "preferences": profile.preferences,
        },
    }

    system_prompt = (
        '你是学习习惯分析助手。请根据用户学习快照生成"树苗画像"，用于前端展示。\n'
        '要求：只输出 JSON（不要 Markdown/解释），所有文本字段必须使用中文，字段必须包含：\n'
        '{\n'
        '  "version": 1,\n'
        '  "stage": "seed|sprout|sapling|tree",\n'
        '  "level": 1-10,\n'
        '  "health": 0-100,\n'
        '  "traits": ["中文特质描述..."],\n'
        '  "summary": "中文总结...",\n'
        '  "nextActions": ["中文行动建议..."],\n'
        '  "visual": {"theme":"green|blue|gold|purple","accessory":"leaf|book|star|flower"},\n'
        '  "stats": {...}\n'
        '}\n'
        '请确保 JSON 可被严格解析。'
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(merged_snapshot, ensure_ascii=False)},
    ]

    try:
        if use_system:
            credits_service.refund_stale_reservations(db, current_user.id)
            request_id = f"seedling:{new_request_id()}"
            est_prompt_tokens = estimate_prompt_tokens(messages)
            reserved_points, _ = calc_qwen_cost_points(model=model_id, prompt_tokens=est_prompt_tokens, completion_tokens=1200)
            reserved_points = max(int(reserved_points or 1), 1)
            credits_service.reserve_points(
                db,
                current_user.id,
                request_id=request_id,
                points=reserved_points,
                meta={"stage": "seedling", "model": model_id},
            )

            try:
                resp = await call_openai_compatible_chat(
                    api_key=api_key,
                    model_id=model_id,
                    base_url=base_url,
                    messages=messages,
                    max_tokens=1200,
                    temperature=0.6,
                    timeout_seconds=120.0,
                )
            except Exception as exc:
                credits_service.finalize_reservation(
                    db,
                    current_user.id,
                    request_id=request_id,
                    reserved_points=reserved_points,
                    actual_points=0,
                    model=model_id,
                    meta={"stage": "seedling", "error": str(exc)},
                )
                raise

            usage = resp.get("usage") if isinstance(resp, dict) else None
            prompt_tokens = int(usage.get("prompt_tokens") or 0) if isinstance(usage, dict) else 0
            completion_tokens = int(usage.get("completion_tokens") or 0) if isinstance(usage, dict) else 0
            total_tokens = int(usage.get("total_tokens") or 0) if isinstance(usage, dict) else 0
            actual_points, cost_milli = calc_qwen_cost_points(
                model=model_id,
                prompt_tokens=prompt_tokens or est_prompt_tokens,
                completion_tokens=completion_tokens or 0,
            )
            credits_service.finalize_reservation(
                db,
                current_user.id,
                request_id=request_id,
                reserved_points=reserved_points,
                actual_points=int(actual_points or reserved_points),
                model=model_id,
                prompt_tokens=prompt_tokens or None,
                completion_tokens=completion_tokens or None,
                total_tokens=total_tokens or None,
                cost_rmb_milli=cost_milli,
                meta={"stage": "seedling"},
            )

            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        else:
            content = await _call_openai_compatible_chat(
                api_key=api_key,
                model_id=model_id,
                base_url=base_url,
                messages=messages,
                max_tokens=1200,
                temperature=0.6,
            )
        portrait = _extract_json(content) or _build_default_seedling_portrait(stats)
    except Exception:
        portrait = _build_default_seedling_portrait(stats)

    profile.seedlingPortrait = portrait
    profile.portraitUpdatedAt = datetime.utcnow()
    db.commit()
    db.refresh(profile)
    return _profile_to_response(profile, current_user.id)
