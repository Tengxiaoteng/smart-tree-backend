from fastapi import APIRouter, Depends, HTTPException, status, Query
import logging
from sqlalchemy import or_, text
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.file_refs import extract_http_urls_from_text, extract_proxy_file_ids_from_text
from app.core.oss import OSSPath, delete_file_by_key, get_oss_path, normalize_oss_key
from app.core.schema import ensure_user_file_schema
from app.core.security import get_current_user
from app.core.user_files import delete_user_file_record_and_maybe_blob
from app.models import KnowledgeNode, User, Topic, Question, AnswerRecord, UserNote, Material, UserFile
from app.schemas.knowledge_node import KnowledgeNodeCreate, KnowledgeNodeUpdate, KnowledgeNodeResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _would_create_cycle(db: Session, user_id: str, node_id: str, new_parent_id: str) -> bool:
    """检查把 node_id 挂到 new_parent_id 下是否会形成环。"""
    current_id = new_parent_id
    seen: set[str] = set()

    for _ in range(1000):
        if current_id == node_id:
            return True
        if current_id in seen:
            return True
        seen.add(current_id)

        parent_id = (
            db.query(KnowledgeNode.parentId)
            .filter(KnowledgeNode.userId == user_id, KnowledgeNode.id == current_id)
            .scalar()
        )
        if not parent_id:
            return False
        current_id = parent_id

    return True


@router.get("", response_model=list[KnowledgeNodeResponse])
async def get_nodes(
    topicId: str = Query(None),
    includeNoTopic: bool = Query(False),
    sortMode: str = Query(None),  # 排序模式参数
    limit: int | None = Query(None, ge=1, le=5000),
    offset: int | None = Query(None, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取节点列表"""
    query = db.query(KnowledgeNode).filter(KnowledgeNode.userId == current_user.id)

    if topicId:
        if includeNoTopic:
            query = query.filter(or_(KnowledgeNode.topicId == topicId, KnowledgeNode.topicId.is_(None)))
        else:
            query = query.filter(KnowledgeNode.topicId == topicId)

    # 获取排序模式（如果没有指定，尝试从 TreeSettings 获取）
    effective_sort_mode = sortMode
    if not effective_sort_mode and topicId:
        settings = None  # TreeSettings temporarily disabled
        effective_sort_mode = settings.sortMode if settings else "createdAt_desc"
    if not effective_sort_mode:
        effective_sort_mode = "createdAt_desc"
    
    # 应用排序
    if effective_sort_mode == "createdAt_asc":
        query = query.order_by(KnowledgeNode.createdAt.asc())
    elif effective_sort_mode == "createdAt_desc":
        query = query.order_by(KnowledgeNode.createdAt.desc())
    elif effective_sort_mode == "name_asc":
        query = query.order_by(KnowledgeNode.name.asc())
    elif effective_sort_mode == "mastery_asc":
        query = query.order_by(KnowledgeNode.mastery.asc(), KnowledgeNode.createdAt.desc())
    elif effective_sort_mode == "mastery_desc":
        query = query.order_by(KnowledgeNode.mastery.desc(), KnowledgeNode.createdAt.desc())
    elif effective_sort_mode == "manual":
        query = query.order_by(KnowledgeNode.sortOrder.asc(), KnowledgeNode.createdAt.desc())
    else:
        query = query.order_by(KnowledgeNode.createdAt.desc())

    if offset is not None:
        query = query.offset(offset)
    if limit is not None:
        query = query.limit(limit)

    nodes = query.all()
    return nodes


@router.post("/batch", response_model=list[KnowledgeNodeResponse])
async def create_nodes_batch(
    nodes_data: list[KnowledgeNodeCreate],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """批量创建节点（减少 AI 一次生成多节点时的请求往返）。"""
    if not nodes_data:
        return []

    # 批量创建要求每个节点携带 id，方便前端本地状态对齐与 parentId 引用
    ids: list[str] = []
    seen_ids: set[str] = set()
    for n in nodes_data:
        if not n.id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="批量创建需要为每个节点提供 id")
        if len(n.id) > 191:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="节点 ID 过长")
        if n.id in seen_ids:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="批量创建中存在重复节点 ID")
        seen_ids.add(n.id)
        ids.append(n.id)

    # 校验 id 是否已存在
    existing = db.query(KnowledgeNode.id).filter(KnowledgeNode.id.in_(ids)).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="节点 ID 已存在")

    # 预取外部父节点（不在本次 batch 内的 parentId）
    external_parent_ids = sorted({n.parentId for n in nodes_data if n.parentId and n.parentId not in seen_ids})
    external_parent_topics: dict[str, str | None] = {}
    if external_parent_ids:
        rows = (
            db.query(KnowledgeNode.id, KnowledgeNode.topicId)
            .filter(KnowledgeNode.userId == current_user.id, KnowledgeNode.id.in_(external_parent_ids))
            .all()
        )
        external_parent_topics = {rid: topic_id for rid, topic_id in rows}
        missing = [pid for pid in external_parent_ids if pid not in external_parent_topics]
        if missing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="父节点不存在")

    # topicId 校验缓存：同一批次里通常都属于同一 Topic
    topic_exists_cache: dict[str, bool] = {}
    created_topic_by_id: dict[str, str | None] = {}
    created_nodes: list[KnowledgeNode] = []

    try:
        for node_data in nodes_data:
            name = node_data.name.strip() if node_data.name else ""
            if not name:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="节点名称不能为空")

            if node_data.parentId and node_data.parentId == node_data.id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="父节点不能是自身")

            parent_topic_id: str | None = None
            if node_data.parentId:
                if node_data.parentId in created_topic_by_id:
                    parent_topic_id = created_topic_by_id[node_data.parentId]
                else:
                    parent_topic_id = external_parent_topics.get(node_data.parentId)

            effective_topic_id = node_data.topicId or parent_topic_id

            if parent_topic_id and effective_topic_id and parent_topic_id != effective_topic_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="子节点 topicId 必须与父节点一致")

            if effective_topic_id:
                if effective_topic_id not in topic_exists_cache:
                    topic = (
                        db.query(Topic.id)
                        .filter(Topic.id == effective_topic_id, Topic.userId == current_user.id)
                        .first()
                    )
                    topic_exists_cache[effective_topic_id] = bool(topic)
                if not topic_exists_cache[effective_topic_id]:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="主题不存在")

            node_kwargs = {
                "id": node_data.id,
                "userId": current_user.id,
                "topicId": effective_topic_id,
                "name": name,
                "description": node_data.description,
                "parentId": node_data.parentId,
                "knowledgeType": node_data.knowledgeType,
                "difficulty": node_data.difficulty,
                "estimatedMinutes": node_data.estimatedMinutes,
                "learningObjectives": node_data.learningObjectives,
                "keyConcepts": node_data.keyConcepts,
                "prerequisites": node_data.prerequisites,
                "questionPatterns": node_data.questionPatterns,
                "commonMistakes": node_data.commonMistakes,
                "children": node_data.children,
                "materialIds": node_data.materialIds,
                "questionIds": node_data.questionIds,
                "userNotes": node_data.userNotes,
                "aiInferredGoals": node_data.aiInferredGoals,
                "source": node_data.source,
                "mastery": node_data.mastery,
                "questionCount": node_data.questionCount,
                "correctCount": node_data.correctCount,
            }
            if node_data.createdAt is not None:
                node_kwargs["createdAt"] = node_data.createdAt
                node_kwargs["updatedAt"] = node_data.updatedAt or node_data.createdAt
            elif node_data.updatedAt is not None:
                node_kwargs["updatedAt"] = node_data.updatedAt

            node = KnowledgeNode(**node_kwargs)
            db.add(node)
            db.flush()  # 确保同批次子节点 parentId 可引用
            created_nodes.append(node)
            created_topic_by_id[node.id] = effective_topic_id

        db.commit()
        for node in created_nodes:
            db.refresh(node)
        return created_nodes
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        logger.exception("批量创建节点失败: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="批量创建节点失败") from exc


@router.get("/{node_id}", response_model=KnowledgeNodeResponse)
async def get_node(
    node_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取单个节点"""
    node = db.query(KnowledgeNode).filter(
        KnowledgeNode.id == node_id,
        KnowledgeNode.userId == current_user.id
    ).first()

    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="节点不存在"
        )
    return node


@router.post("", response_model=KnowledgeNodeResponse)
async def create_node(
    node_data: KnowledgeNodeCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建节点"""
    name = node_data.name.strip() if node_data.name else ""
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="节点名称不能为空")

    # 如果前端传了自定义 id（用于与前端本地状态对齐），需要校验长度与唯一性
    if node_data.id:
        if len(node_data.id) > 191:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="节点 ID 过长"
            )
        existing_id = db.query(KnowledgeNode).filter(KnowledgeNode.id == node_data.id).first()
        if existing_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="节点 ID 已存在"
            )

    if node_data.id and node_data.parentId and node_data.id == node_data.parentId:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="父节点不能是自身")

    # 如果指定了 parentId，验证父节点是否存在（并用于推断 topicId）
    parent = None
    if node_data.parentId:
        parent = db.query(KnowledgeNode).filter(
            KnowledgeNode.id == node_data.parentId,
            KnowledgeNode.userId == current_user.id
        ).first()
        if not parent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="父节点不存在"
            )

    # 推断/校验 topicId：子节点默认继承父节点的 topicId（前端可能不传）
    effective_topic_id = node_data.topicId
    if not effective_topic_id and parent:
        effective_topic_id = parent.topicId

    if parent and parent.topicId and effective_topic_id and parent.topicId != effective_topic_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="子节点 topicId 必须与父节点一致"
        )

    # 如果最终有 topicId，验证主题是否存在且属于当前用户
    if effective_topic_id:
        topic = db.query(Topic).filter(
            Topic.id == effective_topic_id,
            Topic.userId == current_user.id
        ).first()
        if not topic:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="主题不存在"
            )

    node_kwargs = {
        "userId": current_user.id,
        "topicId": effective_topic_id,
        "name": name,
        "description": node_data.description,
        "parentId": node_data.parentId,
        "knowledgeType": node_data.knowledgeType,
        "difficulty": node_data.difficulty,
        "estimatedMinutes": node_data.estimatedMinutes,
        "learningObjectives": node_data.learningObjectives,
        "keyConcepts": node_data.keyConcepts,
        "prerequisites": node_data.prerequisites,
        "questionPatterns": node_data.questionPatterns,
        "commonMistakes": node_data.commonMistakes,
        "children": node_data.children,
        "materialIds": node_data.materialIds,
        "questionIds": node_data.questionIds,
        "userNotes": node_data.userNotes,
        "aiInferredGoals": node_data.aiInferredGoals,
        "source": node_data.source,
        "mastery": node_data.mastery,
        "questionCount": node_data.questionCount,
        "correctCount": node_data.correctCount,
    }
    if node_data.id:
        node_kwargs["id"] = node_data.id
    if node_data.createdAt is not None:
        node_kwargs["createdAt"] = node_data.createdAt
        node_kwargs["updatedAt"] = node_data.updatedAt or node_data.createdAt
    elif node_data.updatedAt is not None:
        node_kwargs["updatedAt"] = node_data.updatedAt

    node = KnowledgeNode(**node_kwargs)
    db.add(node)
    db.commit()
    db.refresh(node)
    return node


@router.patch("/{node_id}", response_model=KnowledgeNodeResponse)
async def update_node(
    node_id: str,
    node_data: KnowledgeNodeUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新节点"""
    node = db.query(KnowledgeNode).filter(
        KnowledgeNode.id == node_id,
        KnowledgeNode.userId == current_user.id
    ).first()

    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="节点不存在"
        )

    payload = node_data.model_dump(exclude_unset=True)
    if "name" in payload and payload.get("name") is not None:
        next_name = str(payload.get("name") or "").strip()
        if not next_name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="节点名称不能为空")
        payload["name"] = next_name

    # parentId / topicId 可能需要一致性校验
    next_parent_id = payload.get("parentId", node.parentId) if "parentId" in payload else node.parentId
    next_topic_id = payload.get("topicId", node.topicId) if "topicId" in payload else node.topicId

    parent = None
    if next_parent_id:
        if next_parent_id == node_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="父节点不能是自身")

        parent = db.query(KnowledgeNode).filter(
            KnowledgeNode.id == next_parent_id,
            KnowledgeNode.userId == current_user.id,
        ).first()
        if not parent:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="父节点不存在")

        if _would_create_cycle(db, current_user.id, node_id, next_parent_id):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不能将节点移动到其子节点下（会形成循环）")

        if parent.topicId and next_topic_id and parent.topicId != next_topic_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="子节点 topicId 必须与父节点一致")
        if parent.topicId and "topicId" in payload and next_topic_id is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="子节点 topicId 必须与父节点一致")

        # 未显式传 topicId 时：默认继承父节点 topicId
        if "topicId" not in payload and parent.topicId and next_topic_id is None:
            next_topic_id = parent.topicId

    if next_topic_id:
        topic = db.query(Topic).filter(
            Topic.id == next_topic_id,
            Topic.userId == current_user.id,
        ).first()
        if not topic:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="主题不存在")

    # 应用 parentId / topicId
    if "parentId" in payload:
        node.parentId = next_parent_id
    if "topicId" in payload or ("topicId" not in payload and parent and parent.topicId and node.topicId is None):
        node.topicId = next_topic_id

    # 其余字段直接赋值
    for field, value in payload.items():
        if field in {"parentId", "topicId"}:
            continue
        setattr(node, field, value)

    db.commit()
    db.refresh(node)
    return node


@router.delete("/{node_id}")
async def delete_node(
    node_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除节点"""
    node = db.query(KnowledgeNode).filter(
        KnowledgeNode.id == node_id,
        KnowledgeNode.userId == current_user.id
    ).first()

    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="节点不存在"
        )

    # 递归删除：避免只删父节点导致子节点变成“孤儿节点”在前端重新出现
    to_delete: list[str] = []
    try:
        rows = db.execute(
            text(
                """
                WITH RECURSIVE descendants(id) AS (
                    SELECT id
                    FROM knowledgenode
                    WHERE id = :node_id AND userId = :user_id
                    UNION
                    SELECT kn.id
                    FROM knowledgenode kn
                    JOIN descendants d ON kn.parentId = d.id
                    WHERE kn.userId = :user_id
                )
                SELECT id FROM descendants
                """
            ),
            {"node_id": node_id, "user_id": current_user.id},
        ).fetchall()
        to_delete = [row[0] for row in rows if row and row[0]]
    except Exception as exc:
        logger.warning("递归 CTE 获取子孙节点失败，回退到逐层查询: %s", exc)
        seen: set[str] = set()
        queue: list[str] = [node_id]

        while queue:
            current_id = queue.pop()
            if current_id in seen:
                continue
            seen.add(current_id)
            to_delete.append(current_id)

            children = (
                db.query(KnowledgeNode.id)
                .filter(
                    KnowledgeNode.userId == current_user.id,
                    KnowledgeNode.parentId == current_id,
                )
                .all()
            )
            queue.extend([cid for (cid,) in children if cid not in seen])

    # 清理：删除这些节点关联笔记里的文件（/api/files/<id> 或旧的 OSS 直链 node images）
    note_contents = [
        content
        for (content,) in db.query(UserNote.content)
        .filter(UserNote.userId == current_user.id, UserNote.nodeId.in_(to_delete))
        .all()
        if content
    ]

    proxy_file_ids: set[str] = set()
    for content in note_contents:
        proxy_file_ids.update(extract_proxy_file_ids_from_text(content))

    if proxy_file_ids:
        ensure_user_file_schema(db)
        for file_id in sorted(proxy_file_ids):
            proxy_url = f"/api/files/{file_id}"

            # 如果还被其他资料/笔记引用，则不删底层文件
            material_refs = (
                db.query(Material)
                .filter(Material.userId == current_user.id, Material.url == proxy_url)
                .count()
            )
            note_refs = (
                db.query(UserNote)
                .filter(
                    UserNote.userId == current_user.id,
                    ~UserNote.nodeId.in_(to_delete),
                    UserNote.content.contains(proxy_url),
                )
                .count()
            )
            if material_refs or note_refs:
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
    for content in note_contents:
        for url in extract_http_urls_from_text(content):
            key = normalize_oss_key(url)
            if not key:
                continue
            if not key.startswith(f"{node_images_prefix}/"):
                continue
            try:
                delete_file_by_key(key)
            except Exception as exc:
                logger.warning("删除旧 node image 失败 key=%s err=%s", key, exc)

    # 清理资料的 nodeIds（避免残留已删除的节点 ID）
    materials = (
        db.query(Material)
        .filter(Material.userId == current_user.id, Material.nodeIds.isnot(None))
        .all()
    )
    node_id_set = set(to_delete)
    for material in materials:
        if not isinstance(material.nodeIds, list) or not material.nodeIds:
            continue
        filtered = [nid for nid in material.nodeIds if nid not in node_id_set]
        if filtered != material.nodeIds:
            material.nodeIds = filtered

    # 先删关联数据，再删节点本身（兼容不同数据库的外键/级联行为）
    db.query(AnswerRecord).filter(
        AnswerRecord.userId == current_user.id,
        AnswerRecord.nodeId.in_(to_delete),
    ).delete(synchronize_session=False)

    db.query(UserNote).filter(
        UserNote.userId == current_user.id,
        UserNote.nodeId.in_(to_delete),
    ).delete(synchronize_session=False)

    db.query(Question).filter(
        Question.userId == current_user.id,
        Question.nodeId.in_(to_delete),
    ).delete(synchronize_session=False)

    db.query(KnowledgeNode).filter(
        KnowledgeNode.userId == current_user.id,
        KnowledgeNode.id.in_(to_delete),
    ).delete(synchronize_session=False)

    db.commit()
    return {"success": True, "deleted": len(to_delete)}


@router.delete("")
async def delete_nodes_by_topic(
    topicId: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除主题下的所有节点（包含关联题目/笔记/记录与笔记图片清理）。"""
    node_ids = [
        nid
        for (nid,) in db.query(KnowledgeNode.id)
        .filter(KnowledgeNode.userId == current_user.id, KnowledgeNode.topicId == topicId)
        .all()
    ]
    if not node_ids:
        return {"deleted": 0}

    # 清理：删除这些节点关联笔记里的文件（/api/files/<id> 或旧的 OSS 直链 node images）
    note_contents = [
        content
        for (content,) in db.query(UserNote.content)
        .filter(UserNote.userId == current_user.id, UserNote.nodeId.in_(node_ids))
        .all()
        if content
    ]

    proxy_file_ids: set[str] = set()
    for content in note_contents:
        proxy_file_ids.update(extract_proxy_file_ids_from_text(content))

    if proxy_file_ids:
        ensure_user_file_schema(db)
        for file_id in sorted(proxy_file_ids):
            proxy_url = f"/api/files/{file_id}"

            material_refs = (
                db.query(Material)
                .filter(Material.userId == current_user.id, Material.url == proxy_url)
                .count()
            )
            note_refs = (
                db.query(UserNote)
                .filter(
                    UserNote.userId == current_user.id,
                    ~UserNote.nodeId.in_(node_ids),
                    UserNote.content.contains(proxy_url),
                )
                .count()
            )
            if material_refs or note_refs:
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

    node_images_prefix = get_oss_path(OSSPath.NODE_IMAGES, current_user.id).rstrip("/")
    for content in note_contents:
        for url in extract_http_urls_from_text(content):
            key = normalize_oss_key(url)
            if not key:
                continue
            if not key.startswith(f"{node_images_prefix}/"):
                continue
            try:
                delete_file_by_key(key)
            except Exception as exc:
                logger.warning("删除旧 node image 失败 key=%s err=%s", key, exc)

    # 清理资料的 nodeIds（避免残留已删除的节点 ID）
    materials = (
        db.query(Material)
        .filter(Material.userId == current_user.id, Material.nodeIds.isnot(None))
        .all()
    )
    node_id_set = set(node_ids)
    for material in materials:
        if not isinstance(material.nodeIds, list) or not material.nodeIds:
            continue
        filtered = [nid for nid in material.nodeIds if nid not in node_id_set]
        if filtered != material.nodeIds:
            material.nodeIds = filtered

    deleted_answer_records = (
        db.query(AnswerRecord)
        .filter(AnswerRecord.userId == current_user.id, AnswerRecord.nodeId.in_(node_ids))
        .delete(synchronize_session=False)
    )
    deleted_notes = (
        db.query(UserNote)
        .filter(UserNote.userId == current_user.id, UserNote.nodeId.in_(node_ids))
        .delete(synchronize_session=False)
    )
    deleted_questions = (
        db.query(Question)
        .filter(Question.userId == current_user.id, Question.nodeId.in_(node_ids))
        .delete(synchronize_session=False)
    )
    deleted_nodes = (
        db.query(KnowledgeNode)
        .filter(KnowledgeNode.userId == current_user.id, KnowledgeNode.id.in_(node_ids))
        .delete(synchronize_session=False)
    )

    db.commit()
    return {
        "deleted": deleted_nodes,
        "deletedNotes": deleted_notes,
        "deletedQuestions": deleted_questions,
        "deletedAnswerRecords": deleted_answer_records,
    }
