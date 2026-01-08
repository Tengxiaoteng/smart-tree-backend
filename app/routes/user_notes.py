from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.file_refs import extract_http_urls_from_text, extract_proxy_file_ids_from_text
from app.core.oss import OSSPath, delete_file_by_key, get_oss_path, normalize_oss_key
from app.core.schema import ensure_user_file_schema
from app.core.security import get_current_user
from app.core.user_files import delete_user_file_record_and_maybe_blob
from app.models import User, UserNote, KnowledgeNode, Question, Material, UserFile
from app.schemas.user_note import UserNoteCreate, UserNoteUpdate, UserNoteResponse

router = APIRouter()


@router.get("", response_model=list[UserNoteResponse])
async def get_user_notes(
    nodeId: str = Query(None),
    topicId: str = Query(None),
    includeNoTopic: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取用户笔记列表"""
    query = db.query(UserNote).filter(UserNote.userId == current_user.id)
    if nodeId:
        query = query.filter(UserNote.nodeId == nodeId)
    if topicId:
        query = query.join(KnowledgeNode, KnowledgeNode.id == UserNote.nodeId).filter(
            KnowledgeNode.userId == current_user.id
        )
        if includeNoTopic:
            query = query.filter(or_(KnowledgeNode.topicId == topicId, KnowledgeNode.topicId.is_(None)))
        else:
            query = query.filter(KnowledgeNode.topicId == topicId)
    return query.order_by(UserNote.updatedAt.desc()).all()


@router.get("/{note_id}", response_model=UserNoteResponse)
async def get_user_note(
    note_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取单个笔记"""
    note = db.query(UserNote).filter(
        UserNote.id == note_id,
        UserNote.userId == current_user.id,
    ).first()
    if not note:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="笔记不存在")
    return note


@router.post("", response_model=UserNoteResponse)
async def create_user_note(
    data: UserNoteCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建笔记"""
    if data.id:
        if len(data.id) > 191:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="笔记 ID 过长")
        existing = db.query(UserNote).filter(UserNote.id == data.id).first()
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="笔记 ID 已存在")

    node = db.query(KnowledgeNode).filter(
        KnowledgeNode.id == data.nodeId,
        KnowledgeNode.userId == current_user.id,
    ).first()
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识点不存在")

    if data.questionId:
        question = db.query(Question).filter(
            Question.id == data.questionId,
            Question.userId == current_user.id,
        ).first()
        if not question:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="题目不存在")

    note_kwargs = {
        "userId": current_user.id,
        "nodeId": data.nodeId,
        "questionId": data.questionId,
        "content": data.content,
        "source": data.source or "manual",
        "tags": data.tags,
    }
    if data.createdAt is not None:
        note_kwargs["createdAt"] = data.createdAt
        note_kwargs["updatedAt"] = data.createdAt
    if data.id:
        note_kwargs["id"] = data.id

    note = UserNote(**note_kwargs)
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


@router.patch("/{note_id}", response_model=UserNoteResponse)
async def update_user_note(
    note_id: str,
    data: UserNoteUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新笔记"""
    note = db.query(UserNote).filter(
        UserNote.id == note_id,
        UserNote.userId == current_user.id,
    ).first()
    if not note:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="笔记不存在")

    payload = data.model_dump(exclude_unset=True)

    if "questionId" in payload and payload["questionId"]:
        question = db.query(Question).filter(
            Question.id == payload["questionId"],
            Question.userId == current_user.id,
        ).first()
        if not question:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="题目不存在")

    for field, value in payload.items():
        setattr(note, field, value)

    db.commit()
    db.refresh(note)
    return note


@router.delete("/{note_id}")
async def delete_user_note(
    note_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除笔记"""
    note = db.query(UserNote).filter(
        UserNote.id == note_id,
        UserNote.userId == current_user.id,
    ).first()
    if not note:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="笔记不存在")

    content = note.content or ""

    # 清理：删除笔记中引用的文件（仅当不再被其他笔记/资料引用）
    proxy_file_ids = extract_proxy_file_ids_from_text(content)
    if proxy_file_ids:
        ensure_user_file_schema(db)
        for file_id in sorted(proxy_file_ids):
            proxy_url = f"/api/files/{file_id}"
            material_refs = (
                db.query(Material)
                .filter(Material.userId == current_user.id, Material.url == proxy_url)
                .count()
            )
            other_note_refs = (
                db.query(UserNote)
                .filter(
                    UserNote.userId == current_user.id,
                    UserNote.id != note_id,
                    UserNote.content.contains(proxy_url),
                )
                .count()
            )
            if material_refs or other_note_refs:
                continue

            file_record = db.query(UserFile).filter(
                UserFile.id == file_id,
                UserFile.userId == current_user.id,
            ).first()
            if not file_record:
                continue

            ok = delete_user_file_record_and_maybe_blob(db, file_record)
            if not ok:
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="OSS 文件删除失败")

    # 兼容旧数据：node/image 直链写进笔记时，尝试删除 nodes/<user_id>/images 下的 OSS 对象（best-effort）
    node_images_prefix = get_oss_path(OSSPath.NODE_IMAGES, current_user.id).rstrip("/")
    for url in extract_http_urls_from_text(content):
        key = normalize_oss_key(url)
        if not key:
            continue
        if not key.startswith(f"{node_images_prefix}/"):
            continue
        delete_file_by_key(key)

    db.delete(note)
    db.commit()
    return {"success": True}
