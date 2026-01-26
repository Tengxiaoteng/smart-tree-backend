from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.file_refs import extract_file_id_from_proxy_url, extract_http_urls_from_text, extract_proxy_file_ids_from_text
from app.core.oss import OSSPath, delete_file_by_key, get_oss_path, normalize_oss_key
from app.core.schema import ensure_user_file_schema
from app.core.security import get_current_user
from app.core.user_files import (
    delete_user_file_record_and_maybe_blob,
    ensure_user_files_copied,
    replace_proxy_file_ids_in_json,
    replace_proxy_file_ids_in_text,
)
from app.models import AnswerRecord, KnowledgeNode, Material, Question, Topic, User, UserNote, UserFile
from app.schemas.topic import TopicCloneRequest, TopicCreate, TopicUpdate, TopicResponse
import uuid
from pydantic import BaseModel

router = APIRouter()


class TopicStatsResponse(BaseModel):
    totalTopics: int
    totalNodes: int
    nodesByTopic: dict[str, int]
    orphanNodes: int
    includeNoTopic: bool

def _normalize_topic_name(name: str | None) -> str:
    if not name:
        return ""
    return name.strip()


def _generate_unique_topic_name(db: Session, user_id: str, desired_name: str) -> str:
    base = _normalize_topic_name(desired_name) or "未命名知识树"

    exists = db.query(Topic).filter(Topic.userId == user_id, Topic.name == base).first()
    if not exists:
        return base

    # 自动加 “（副本）/（副本 2）…” 直到不冲突
    for i in range(1, 1000):
        candidate = f"{base}（副本）" if i == 1 else f"{base}（副本 {i}）"
        if not db.query(Topic).filter(Topic.userId == user_id, Topic.name == candidate).first():
            return candidate

    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="无法生成唯一的知识树名称")


def _remap_str_list(values: list[str] | None, id_map: dict[str, str]) -> list[str] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        return values
    return [id_map.get(v, v) for v in values]


@router.get("", response_model=list[TopicResponse])
async def get_topics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取当前用户的所有主题"""
    topics = db.query(Topic).filter(Topic.userId == current_user.id).all()
    return topics


@router.get("/stats", response_model=TopicStatsResponse)
async def get_topic_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取知识树统计（用于知识森林页展示节点数，避免必须先加载每棵树的全量数据）。"""
    topic_ids = [tid for (tid,) in db.query(Topic.id).filter(Topic.userId == current_user.id).all()]
    total_topics = len(topic_ids)
    include_no_topic = total_topics == 1

    nodes_by_topic: dict[str, int] = {tid: 0 for tid in topic_ids}
    orphan_nodes = 0

    rows = (
        db.query(KnowledgeNode.topicId, func.count(KnowledgeNode.id))
        .filter(KnowledgeNode.userId == current_user.id)
        .group_by(KnowledgeNode.topicId)
        .all()
    )

    for topic_id, count in rows:
        if topic_id is None:
            orphan_nodes += int(count or 0)
            continue
        if topic_id in nodes_by_topic:
            nodes_by_topic[topic_id] = int(count or 0)
        else:
            # 兜底：topic 已删除但节点残留，归入 orphan 统计
            orphan_nodes += int(count or 0)

    if include_no_topic and topic_ids:
        nodes_by_topic[topic_ids[0]] = nodes_by_topic.get(topic_ids[0], 0) + orphan_nodes
        orphan_nodes = 0

    total_nodes = sum(nodes_by_topic.values()) + orphan_nodes

    return TopicStatsResponse(
        totalTopics=total_topics,
        totalNodes=total_nodes,
        nodesByTopic=nodes_by_topic,
        orphanNodes=orphan_nodes,
        includeNoTopic=include_no_topic,
    )


@router.get("/{topic_id}", response_model=TopicResponse)
async def get_topic(
    topic_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取单个主题"""
    topic = db.query(Topic).filter(
        Topic.id == topic_id,
        Topic.userId == current_user.id
    ).first()

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="主题不存在"
        )
    return topic


@router.post("/{topic_id}/clone", response_model=TopicResponse)
async def clone_topic(
    topic_id: str,
    payload: TopicCloneRequest | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """复制一份指定知识树到当前用户账号（深拷贝节点，原树保持隔离）。"""
    source_topic = db.query(Topic).filter(Topic.id == topic_id).first()
    if not source_topic:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="未找到该知识树，请检查 ID 是否正确")

    if source_topic.userId != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权复制该知识树")

    desired_name = _normalize_topic_name(payload.name) if payload else ""
    if not desired_name:
        desired_name = source_topic.name

    new_topic = Topic(
        id=str(uuid.uuid4()),
        userId=current_user.id,
        name=_generate_unique_topic_name(db, current_user.id, desired_name),
        description=source_topic.description,
        scope=source_topic.scope,
        keywords=source_topic.keywords,
        rootNodeId=None,
    )
    db.add(new_topic)
    db.flush()

    # 兼容旧数据：如果源账号只有一个 Topic，则 topicId 为空的节点也算在该树下
    source_topic_count = db.query(Topic).filter(Topic.userId == source_topic.userId).count()
    include_no_topic = source_topic_count == 1

    # 先取源节点（用于后续拷贝资料/笔记/题目与文件引用）
    node_query = db.query(KnowledgeNode).filter(KnowledgeNode.userId == source_topic.userId)
    if include_no_topic:
        node_query = node_query.filter(or_(KnowledgeNode.topicId == topic_id, KnowledgeNode.topicId.is_(None)))
    else:
        node_query = node_query.filter(KnowledgeNode.topicId == topic_id)

    source_nodes = node_query.all()
    source_node_ids = [n.id for n in source_nodes]
    id_map: dict[str, str] = {n.id: str(uuid.uuid4()) for n in source_nodes}

    # 源资料：
    # - 新数据：material.topicId 可能为空（前端很多场景只写 nodeIds 不写 topicId）
    # - 旧数据：与 include_no_topic 的节点兼容逻辑保持一致
    material_query = db.query(Material).filter(Material.userId == source_topic.userId)
    if include_no_topic:
        source_materials = material_query.filter(or_(Material.topicId == topic_id, Material.topicId.is_(None))).all()
    else:
        candidate_materials = material_query.all()
        source_node_id_set = set(source_node_ids)
        material_ids_from_nodes: set[str] = set()
        for n in source_nodes:
            if isinstance(n.materialIds, list) and n.materialIds:
                material_ids_from_nodes.update([mid for mid in n.materialIds if mid])

        seen_material_ids: set[str] = set()
        source_materials = []
        for m in candidate_materials:
            if m.id in seen_material_ids:
                continue

            include = False
            if m.topicId == topic_id:
                include = True
            elif m.id in material_ids_from_nodes:
                include = True
            elif isinstance(m.nodeIds, list) and m.nodeIds:
                include = bool(source_node_id_set.intersection(m.nodeIds))

            if include:
                source_materials.append(m)
                seen_material_ids.add(m.id)

    source_questions: list[Question] = []
    source_notes: list[UserNote] = []
    if source_node_ids:
        source_questions = (
            db.query(Question)
            .filter(Question.userId == source_topic.userId, Question.nodeId.in_(source_node_ids))
            .all()
        )
        source_notes = (
            db.query(UserNote)
            .filter(UserNote.userId == source_topic.userId, UserNote.nodeId.in_(source_node_ids))
            .all()
        )

    # ===== 文件“逻辑拷贝”=====
    # 复制：为当前用户创建/复用新的 UserFile 记录（ossPath 共享同一对象，不重复占用存储）
    source_file_ids: set[str] = set()
    for n in source_nodes:
        source_file_ids.update(extract_proxy_file_ids_from_text(n.description))
    for m in source_materials:
        fid = extract_file_id_from_proxy_url(m.url)
        if fid:
            source_file_ids.add(fid)
        source_file_ids.update(extract_proxy_file_ids_from_text(m.content))
        source_file_ids.update(extract_proxy_file_ids_from_text(m.organizedContent))
        source_file_ids.update(extract_proxy_file_ids_from_text(m.aiSummary))
        if m.structuredContent is not None:
            source_file_ids.update(extract_proxy_file_ids_from_text(str(m.structuredContent)))
    for q in source_questions:
        source_file_ids.update(extract_proxy_file_ids_from_text(q.content))
        source_file_ids.update(extract_proxy_file_ids_from_text(q.answer))
        source_file_ids.update(extract_proxy_file_ids_from_text(q.explanation))
        if q.options is not None:
            source_file_ids.update(extract_proxy_file_ids_from_text(str(q.options)))
        if q.hints is not None:
            source_file_ids.update(extract_proxy_file_ids_from_text(str(q.hints)))
    for note in source_notes:
        source_file_ids.update(extract_proxy_file_ids_from_text(note.content))

    file_id_map = ensure_user_files_copied(db, source_file_ids, current_user.id)

    clones_by_old_id: dict[str, KnowledgeNode] = {}

    # 第一阶段：先插入所有节点（parentId 先置空，避免自引用外键顺序问题）
    for n in source_nodes:
        clone = KnowledgeNode(
            id=id_map[n.id],
            userId=current_user.id,
            topicId=new_topic.id,
            name=n.name,
            description=replace_proxy_file_ids_in_text(n.description, file_id_map),
            learningObjectives=n.learningObjectives,
            keyConcepts=n.keyConcepts,
            knowledgeType=n.knowledgeType,
            difficulty=n.difficulty,
            estimatedMinutes=n.estimatedMinutes,
            prerequisites=_remap_str_list(n.prerequisites, id_map),
            questionPatterns=n.questionPatterns,
            commonMistakes=n.commonMistakes,
            children=_remap_str_list(n.children, id_map) or [],
            materialIds=[],
            questionIds=[],
            userNotes=[],
            aiInferredGoals=n.aiInferredGoals,
            source=n.source,
            parentId=None,
            mastery=0,
            questionCount=0,
            correctCount=0,
        )
        clones_by_old_id[n.id] = clone
        db.add(clone)

    db.flush()

    # 第二阶段：回填 parentId（此时新节点已存在）
    for n in source_nodes:
        if not n.parentId:
            continue
        if n.parentId not in id_map:
            continue
        clones_by_old_id[n.id].parentId = id_map[n.parentId]

    # rootNodeId 映射
    if source_topic.rootNodeId and source_topic.rootNodeId in id_map:
        new_topic.rootNodeId = id_map[source_topic.rootNodeId]

    # ===== 复制资料（并重写 nodeIds / 文件引用）=====
    material_id_map: dict[str, str] = {}
    for m in source_materials:
        new_material_id = str(uuid.uuid4())
        material_id_map[m.id] = new_material_id

        mapped_node_ids: list[str] = []
        if isinstance(m.nodeIds, list) and m.nodeIds:
            mapped_node_ids = [id_map[nid] for nid in m.nodeIds if nid in id_map]

        next_url = m.url
        fid = extract_file_id_from_proxy_url(m.url)
        if fid and fid in file_id_map:
            next_url = f"/api/files/{file_id_map[fid]}"

        material = Material(
            id=new_material_id,
            userId=current_user.id,
            topicId=new_topic.id,
            name=m.name,
            type=m.type,
            content=replace_proxy_file_ids_in_text(m.content, file_id_map),
            url=next_url,
            fileSize=m.fileSize,
            nodeIds=mapped_node_ids,
            tags=m.tags,
            organizedContent=replace_proxy_file_ids_in_text(m.organizedContent, file_id_map),
            aiSummary=replace_proxy_file_ids_in_text(m.aiSummary, file_id_map),
            extractedConcepts=replace_proxy_file_ids_in_json(m.extractedConcepts, file_id_map),
            isOrganized=m.isOrganized,
            structuredContent=replace_proxy_file_ids_in_json(m.structuredContent, file_id_map),
            isStructured=m.isStructured,
        )
        db.add(material)

    # ===== 复制题目（不复制做题记录）=====
    question_id_map: dict[str, str] = {}
    cloned_questions_by_old_id: dict[str, Question] = {}
    for q in source_questions:
        if q.nodeId not in id_map:
            continue
        new_question_id = str(uuid.uuid4())
        question_id_map[q.id] = new_question_id

        cloned = Question(
            id=new_question_id,
            userId=current_user.id,
            nodeId=id_map[q.nodeId],
            type=q.type,
            difficulty=q.difficulty,
            content=replace_proxy_file_ids_in_text(q.content, file_id_map) or q.content,
            options=replace_proxy_file_ids_in_json(q.options, file_id_map),
            answer=replace_proxy_file_ids_in_text(q.answer, file_id_map),
            explanation=replace_proxy_file_ids_in_text(q.explanation, file_id_map),
            hints=replace_proxy_file_ids_in_json(q.hints, file_id_map),
            relatedConcepts=replace_proxy_file_ids_in_json(q.relatedConcepts, file_id_map),
            targetGoalIds=q.targetGoalIds,
            targetGoalNames=q.targetGoalNames,
            isFavorite=False,
            userNotes=replace_proxy_file_ids_in_text(q.userNotes, file_id_map),
            tags=replace_proxy_file_ids_in_json(q.tags, file_id_map),
            source=q.source,
            sourceMaterialIds=_remap_str_list(q.sourceMaterialIds, material_id_map),
            sourceMaterialNames=q.sourceMaterialNames,
            sourceContext=replace_proxy_file_ids_in_text(q.sourceContext, file_id_map),
            difficultyReason=replace_proxy_file_ids_in_text(q.difficultyReason, file_id_map),
            derivedFromQuestionId=None,
            derivedFromRecordId=None,
            isDerivedQuestion=bool(getattr(q, "isDerivedQuestion", False)),
            parentQuestionId=None,
            derivationType=q.derivationType,
        )
        cloned_questions_by_old_id[q.id] = cloned
        db.add(cloned)

    # 回填 parentQuestionId / derivedFromQuestionId（同一节点内的衍生题关系）
    for q in source_questions:
        src_parent_id = q.parentQuestionId or q.derivedFromQuestionId
        if not src_parent_id:
            continue
        new_parent_id = question_id_map.get(src_parent_id)
        cloned = cloned_questions_by_old_id.get(q.id)
        if not new_parent_id or not cloned:
            continue
        cloned.parentQuestionId = new_parent_id
        cloned.derivedFromQuestionId = new_parent_id
        cloned.isDerivedQuestion = True
        if not cloned.derivationType:
            cloned.derivationType = "derived"

    # ===== 复制笔记（并重写 nodeId / questionId / 文件引用）=====
    note_id_map: dict[str, str] = {}
    for note in source_notes:
        if note.nodeId not in id_map:
            continue
        new_note_id = str(uuid.uuid4())
        note_id_map[note.id] = new_note_id
        cloned_note = UserNote(
            id=new_note_id,
            userId=current_user.id,
            nodeId=id_map[note.nodeId],
            questionId=question_id_map.get(note.questionId) if note.questionId else None,
            content=replace_proxy_file_ids_in_text(note.content, file_id_map) or note.content,
            source=note.source,
            tags=note.tags,
        )
        db.add(cloned_note)

    # ===== 回填节点关联字段（materialIds/questionIds/userNotes）=====
    for n in source_nodes:
        clone = clones_by_old_id.get(n.id)
        if not clone:
            continue
        clone.materialIds = _remap_str_list(n.materialIds, material_id_map) or []
        clone.questionIds = _remap_str_list(n.questionIds, question_id_map) or []
        clone.userNotes = _remap_str_list(n.userNotes, note_id_map) or []

    db.commit()
    db.refresh(new_topic)
    return new_topic


@router.post("", response_model=TopicResponse)
async def create_topic(
    topic_data: TopicCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建新主题"""
    # 如果前端传了自定义 id（用于与前端本地状态对齐），需要校验长度与唯一性
    if topic_data.id:
        if len(topic_data.id) > 191:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="主题 ID 过长"
            )
        existing_id = db.query(Topic).filter(Topic.id == topic_data.id).first()
        if existing_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="主题 ID 已存在"
            )

    # 检查主题名称是否已存在（同一用户）
    existing = db.query(Topic).filter(
        Topic.userId == current_user.id,
        Topic.name == topic_data.name
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="主题名称已存在"
        )

    if topic_data.rootNodeId is not None and topic_data.rootNodeId and len(topic_data.rootNodeId) > 191:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="根节点 ID 过长"
        )

    topic_kwargs = {
        "userId": current_user.id,
        "name": topic_data.name,
        "description": topic_data.description,
        "scope": topic_data.scope,
        "keywords": topic_data.keywords,
        "rootNodeId": topic_data.rootNodeId,
    }
    if topic_data.id:
        topic_kwargs["id"] = topic_data.id

    topic = Topic(**topic_kwargs)
    db.add(topic)
    db.commit()
    db.refresh(topic)
    return topic


@router.patch("/{topic_id}", response_model=TopicResponse)
async def update_topic(
    topic_id: str,
    topic_data: TopicUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新主题"""
    topic = db.query(Topic).filter(
        Topic.id == topic_id,
        Topic.userId == current_user.id
    ).first()

    if not topic:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="主题不存在"
        )

    # 更新字段
    if topic_data.name is not None:
        topic.name = topic_data.name
    if topic_data.description is not None:
        topic.description = topic_data.description
    if topic_data.scope is not None:
        topic.scope = topic_data.scope
    if topic_data.keywords is not None:
        topic.keywords = topic_data.keywords
    if topic_data.rootNodeId is not None:
        topic.rootNodeId = topic_data.rootNodeId

    db.commit()
    db.refresh(topic)
    return topic


@router.delete("/{topic_id}")
async def delete_topic(
    topic_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除主题（默认使用级联删除，避免残留孤儿数据/文件）。"""
    return await delete_topic_cascade(
        topic_id=topic_id,
        includeNoTopic=False,
        deleteMaterials=True,
        deleteFiles=True,
        db=db,
        current_user=current_user,
    )


@router.delete("/{topic_id}/cascade")
async def delete_topic_cascade(
    topic_id: str,
    includeNoTopic: bool = Query(False, description="兼容旧数据：同时删除 topicId 为空的节点"),
    deleteMaterials: bool = Query(True, description="删除该主题下的所有资料"),
    deleteFiles: bool = Query(True, description="删除资料对应的 OSS 文件（仅限 /api/files/<id> 代理文件）"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除主题（级联删除该主题下的所有节点及关联数据）。"""
    topic = db.query(Topic).filter(
        Topic.id == topic_id,
        Topic.userId == current_user.id,
    ).first()

    if not topic:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="主题不存在")

    node_id_query = db.query(KnowledgeNode.id).filter(KnowledgeNode.userId == current_user.id)
    if includeNoTopic:
        node_id_query = node_id_query.filter(or_(KnowledgeNode.topicId == topic_id, KnowledgeNode.topicId.is_(None)))
    else:
        node_id_query = node_id_query.filter(KnowledgeNode.topicId == topic_id)

    node_ids = [nid for (nid,) in node_id_query.all()]

    deleted_answer_records = 0
    deleted_notes = 0
    deleted_questions = 0
    deleted_nodes = 0
    updated_materials = 0
    deleted_materials = 0
    deleted_files = 0

    # 资料删除/清理（与是否存在节点无关）
    if deleteMaterials:
        materials_to_delete = (
            db.query(Material)
            .filter(Material.userId == current_user.id, Material.topicId == topic_id)
            .all()
        )
        material_ids = {m.id for m in materials_to_delete}
        file_ids: set[str] = set()
        url_by_file_id: dict[str, str] = {}
        direct_keys_to_delete: set[str] = set()
        for m in materials_to_delete:
            fid = extract_file_id_from_proxy_url(m.url)
            if fid:
                file_ids.add(fid)
                url_by_file_id[fid] = m.url or ""
            else:
                key = normalize_oss_key(m.url)
                if key:
                    allowed_prefixes = {
                        get_oss_path(OSSPath.MATERIAL_IMAGES, current_user.id).rstrip("/"),
                        get_oss_path(OSSPath.MATERIAL_DOCUMENTS, current_user.id).rstrip("/"),
                        get_oss_path(OSSPath.MATERIAL_AUDIO, current_user.id).rstrip("/"),
                    }
                    if any(key.startswith(f"{prefix}/") for prefix in allowed_prefixes):
                        direct_keys_to_delete.add(key)

        if deleteFiles and file_ids:
            ensure_user_file_schema(db)
            for fid in file_ids:
                url = url_by_file_id.get(fid, "")
                shared_count = (
                    db.query(Material)
                    .filter(
                        Material.userId == current_user.id,
                        Material.url == url,
                        ~Material.id.in_(material_ids),
                    )
                    .count()
                )
                if shared_count != 0:
                    continue

                file_record = db.query(UserFile).filter(
                    UserFile.id == fid,
                    UserFile.userId == current_user.id,
                ).first()
                if not file_record:
                    continue
                ok = delete_user_file_record_and_maybe_blob(db, file_record)
                if not ok:
                    raise HTTPException(status_code=502, detail="OSS 文件删除失败")
                deleted_files += 1

        if deleteFiles and direct_keys_to_delete:
            for key in sorted(direct_keys_to_delete):
                delete_file_by_key(key)

        for m in materials_to_delete:
            db.delete(m)
        deleted_materials = len(materials_to_delete)

    if node_ids:
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

                # 如果还被其他资料/笔记引用，则不删底层文件
                material_query = db.query(Material).filter(
                    Material.userId == current_user.id,
                    Material.url == proxy_url,
                )
                if deleteMaterials:
                    material_query = material_query.filter(Material.topicId != topic_id)
                material_refs = material_query.count()

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
                deleted_files += 1

        # 兼容旧数据：node/image 直链写进笔记时，尝试删除 nodes/<user_id>/images 下的 OSS 对象（best-effort）
        node_images_prefix = get_oss_path(OSSPath.NODE_IMAGES, current_user.id).rstrip("/")
        for content in note_contents:
            for url in extract_http_urls_from_text(content):
                key = normalize_oss_key(url)
                if not key:
                    continue
                if not key.startswith(f"{node_images_prefix}/"):
                    continue
                delete_file_by_key(key)

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

        if not deleteMaterials:
            # 清理资料的 nodeIds（避免残留已删除的节点 ID）
            node_id_set = set(node_ids)
            materials = (
                db.query(Material)
                .filter(Material.userId == current_user.id, Material.nodeIds.isnot(None))
                .all()
            )
            for material in materials:
                if not isinstance(material.nodeIds, list) or not material.nodeIds:
                    continue
                filtered = [nid for nid in material.nodeIds if nid not in node_id_set]
                if filtered != material.nodeIds:
                    material.nodeIds = filtered
                    updated_materials += 1

    # 删除与该主题关联的订阅记录（如果该主题是从分享复制而来的）
    from app.models import TreeSubscription
    deleted_subscriptions = (
        db.query(TreeSubscription)
        .filter(TreeSubscription.localTopicId == topic_id, TreeSubscription.subscriberId == current_user.id)
        .delete(synchronize_session=False)
    )

    db.delete(topic)
    db.commit()

    return {
        "success": True,
        "deletedNodes": deleted_nodes,
        "deletedQuestions": deleted_questions,
        "deletedAnswerRecords": deleted_answer_records,
        "deletedNotes": deleted_notes,
        "updatedMaterials": updated_materials,
        "deletedMaterials": deleted_materials,
        "deletedFiles": deleted_files,
    }
