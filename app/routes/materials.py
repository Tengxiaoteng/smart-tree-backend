from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import inspect, or_, text
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from app.core.database import get_db
from app.core.file_refs import extract_file_id_from_proxy_url
from app.core.oss import OSSPath, delete_file_by_key, get_oss_path, normalize_oss_key
from app.core.security import get_current_user
from app.core.schema import ensure_user_file_schema
from app.core.user_files import delete_user_file_record_and_maybe_blob
from app.models import User, Material, Topic, UserFile
from app.schemas.material import MaterialCreate, MaterialUpdate, MaterialResponse

router = APIRouter()

def _ensure_material_ai_columns(db: Session) -> None:
    """确保 material 表存在 AI/结构化字段（兼容旧库）。"""
    engine = db.get_bind()
    inspector = inspect(engine)
    if "material" not in inspector.get_table_names():
        return

    existing_cols = {col["name"] for col in inspector.get_columns("material")}
    is_mysql = engine.dialect.name == "mysql"

    alter_statements: list[str] = []

    def _add(stmt_mysql: str, stmt_other: str) -> None:
        alter_statements.append(stmt_mysql if is_mysql else stmt_other)

    if "organizedContent" not in existing_cols:
        _add(
            "ALTER TABLE material ADD COLUMN organizedContent LONGTEXT NULL",
            "ALTER TABLE material ADD COLUMN organizedContent TEXT",
        )
    if "aiSummary" not in existing_cols:
        _add(
            "ALTER TABLE material ADD COLUMN aiSummary LONGTEXT NULL",
            "ALTER TABLE material ADD COLUMN aiSummary TEXT",
        )
    if "extractedConcepts" not in existing_cols:
        _add(
            "ALTER TABLE material ADD COLUMN extractedConcepts JSON NULL",
            "ALTER TABLE material ADD COLUMN extractedConcepts TEXT",
        )
    if "isOrganized" not in existing_cols:
        _add(
            "ALTER TABLE material ADD COLUMN isOrganized TINYINT(1) NOT NULL DEFAULT 0",
            "ALTER TABLE material ADD COLUMN isOrganized BOOLEAN NOT NULL DEFAULT 0",
        )
    if "structuredContent" not in existing_cols:
        _add(
            "ALTER TABLE material ADD COLUMN structuredContent JSON NULL",
            "ALTER TABLE material ADD COLUMN structuredContent TEXT",
        )
    if "isStructured" not in existing_cols:
        _add(
            "ALTER TABLE material ADD COLUMN isStructured TINYINT(1) NOT NULL DEFAULT 0",
            "ALTER TABLE material ADD COLUMN isStructured BOOLEAN NOT NULL DEFAULT 0",
        )

    if not alter_statements:
        return

    with engine.begin() as conn:
        for stmt in alter_statements:
            conn.execute(text(stmt))


@router.get("", response_model=list[MaterialResponse])
async def get_materials(
    topicId: str = Query(None),
    includeNoTopic: bool = Query(False),
    nodeId: str = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取学习资料列表"""
    query = db.query(Material).filter(Material.userId == current_user.id)
    if topicId:
        if includeNoTopic:
            query = query.filter(or_(Material.topicId == topicId, Material.topicId.is_(None)))
        else:
            query = query.filter(Material.topicId == topicId)

    try:
        materials = query.order_by(Material.createdAt.desc()).all()
    except OperationalError as exc:
        # 旧库缺少 AI 字段列时，自动补齐后重试一次
        msg = str(exc.orig) if getattr(exc, "orig", None) is not None else str(exc)
        if "Unknown column" in msg or "1054" in msg:
            try:
                _ensure_material_ai_columns(db)
            except SQLAlchemyError as migrate_exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"material 表缺少 AI 字段且无法自动补齐，请执行 migrations/add_material_ai_fields.sql: {migrate_exc}",
                ) from migrate_exc
            materials = query.order_by(Material.createdAt.desc()).all()
        else:
            raise

    # nodeId 过滤（JSON 字段兼容实现，避免依赖不同数据库的 JSON 查询方言）
    if nodeId:
        materials = [m for m in materials if (m.nodeIds or []) and nodeId in (m.nodeIds or [])]

    print(f"[获取资料列表] topicId: {topicId}, includeNoTopic: {includeNoTopic}, nodeId: {nodeId}, 返回数量: {len(materials)}")
    return materials


@router.get("/{material_id}", response_model=MaterialResponse)
async def get_material(
    material_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取单个资料"""
    try:
        material = db.query(Material).filter(
            Material.id == material_id,
            Material.userId == current_user.id,
        ).first()
    except OperationalError as exc:
        msg = str(exc.orig) if getattr(exc, "orig", None) is not None else str(exc)
        if "Unknown column" in msg or "1054" in msg:
            try:
                _ensure_material_ai_columns(db)
            except SQLAlchemyError as migrate_exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"material 表缺少 AI 字段且无法自动补齐，请执行 migrations/add_material_ai_fields.sql: {migrate_exc}",
                ) from migrate_exc
            material = db.query(Material).filter(
                Material.id == material_id,
                Material.userId == current_user.id,
            ).first()
        else:
            raise
    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="资料不存在"
        )
    return material


@router.post("", response_model=MaterialResponse)
async def create_material(
    data: MaterialCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建资料"""
    # 兼容旧库：确保 AI 字段列存在，避免插入/读取时报错
    try:
        _ensure_material_ai_columns(db)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"material 表缺少 AI 字段且无法自动补齐，请执行 migrations/add_material_ai_fields.sql: {exc}",
        ) from exc

    # 前端可能自带 id（用于与本地状态对齐）
    if data.id:
        if len(data.id) > 191:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="资料 ID 过长")
        existing = db.query(Material).filter(Material.id == data.id).first()
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="资料 ID 已存在")

    if data.topicId:
        topic = db.query(Topic).filter(
            Topic.id == data.topicId,
            Topic.userId == current_user.id,
        ).first()
        if not topic:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="主题不存在")

    material_kwargs = {
        "userId": current_user.id,
        "topicId": data.topicId,
        "type": data.type,
        "name": data.name,
        "content": data.content,
        "url": data.url,
        "fileSize": data.fileSize,
        "nodeIds": data.nodeIds or [],
        "tags": data.tags or [],
        "organizedContent": data.organizedContent,
        "aiSummary": data.aiSummary,
        "extractedConcepts": data.extractedConcepts,
        "isOrganized": bool(data.isOrganized) if data.isOrganized is not None else False,
        "structuredContent": data.structuredContent,
        "isStructured": bool(data.isStructured) if data.isStructured is not None else False,
        # 快速匹配字段
        "contentDigest": data.contentDigest,
        "keyTopics": data.keyTopics,
        "contentHash": data.contentHash,
        "digestGeneratedAt": data.digestGeneratedAt,
    }
    if data.id:
        material_kwargs["id"] = data.id

    material = Material(**material_kwargs)
    db.add(material)

    try:
        db.commit()
        print(f"[资料保存成功] ID: {material.id}, 名称: {material.name}, topicId: {material.topicId}, nodeIds: {material.nodeIds}")
    except Exception as e:
        db.rollback()
        print(f"[资料保存失败] 错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"资料保存失败: {str(e)}"
        )

    db.refresh(material)
    return material


@router.patch("/{material_id}", response_model=MaterialResponse)
async def update_material(
    material_id: str,
    data: MaterialUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新资料"""
    try:
        _ensure_material_ai_columns(db)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"material 表缺少 AI 字段且无法自动补齐，请执行 migrations/add_material_ai_fields.sql: {exc}",
        ) from exc

    material = db.query(Material).filter(
        Material.id == material_id,
        Material.userId == current_user.id,
    ).first()
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="资料不存在")

    if data.topicId is not None:
        if data.topicId:
            topic = db.query(Topic).filter(
                Topic.id == data.topicId,
                Topic.userId == current_user.id,
            ).first()
            if not topic:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="主题不存在")
        material.topicId = data.topicId

    if data.type is not None:
        material.type = data.type
    if data.name is not None:
        material.name = data.name
    if data.content is not None:
        material.content = data.content
    if data.url is not None:
        material.url = data.url
    if data.fileSize is not None:
        material.fileSize = data.fileSize
    if data.nodeIds is not None:
        material.nodeIds = data.nodeIds
    if data.tags is not None:
        material.tags = data.tags
    if data.organizedContent is not None:
        material.organizedContent = data.organizedContent
    if data.aiSummary is not None:
        material.aiSummary = data.aiSummary
    if data.extractedConcepts is not None:
        material.extractedConcepts = data.extractedConcepts
    if data.isOrganized is not None:
        material.isOrganized = data.isOrganized
    if data.structuredContent is not None:
        material.structuredContent = data.structuredContent
    if data.isStructured is not None:
        material.isStructured = data.isStructured

    db.commit()
    db.refresh(material)
    return material


@router.delete("/{material_id}")
async def delete_material(
    material_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除资料（若为上传文件资料，同时清理 OSS 对象与 user_file 记录）。"""
    material = db.query(Material).filter(
        Material.id == material_id,
        Material.userId == current_user.id,
    ).first()
    if not material:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="资料不存在")

    file_id = extract_file_id_from_proxy_url(material.url)
    if file_id:
        ensure_user_file_schema(db)
        # 如果同一个 file_id 被多条资料复用，则不删底层文件（只删当前资料）。
        shared_count = (
            db.query(Material)
            .filter(
                Material.userId == current_user.id,
                Material.url == material.url,
                Material.id != material_id,
            )
            .count()
        )
        if shared_count == 0:
            file_record = db.query(UserFile).filter(
                UserFile.id == file_id,
                UserFile.userId == current_user.id,
            ).first()
            if file_record:
                ok = delete_user_file_record_and_maybe_blob(db, file_record)
                if not ok:
                    raise HTTPException(status_code=502, detail="OSS 文件删除失败")
    else:
        # 兼容旧数据：material.url 直接存 OSS 公网 URL 时，尽量删除 materials/<user_id>/... 下的对象
        key = normalize_oss_key(material.url)
        if key:
            allowed_prefixes = {
                get_oss_path(OSSPath.MATERIAL_IMAGES, current_user.id).rstrip("/"),
                get_oss_path(OSSPath.MATERIAL_DOCUMENTS, current_user.id).rstrip("/"),
                get_oss_path(OSSPath.MATERIAL_AUDIO, current_user.id).rstrip("/"),
            }
            if any(key.startswith(f"{prefix}/") for prefix in allowed_prefixes):
                delete_file_by_key(key)

    db.delete(material)
    db.commit()
    return {"success": True}
