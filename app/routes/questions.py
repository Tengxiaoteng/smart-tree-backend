from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
import json
from sqlalchemy import or_
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User, Question, KnowledgeNode
from app.schemas.question import QuestionCreate, QuestionUpdate, QuestionResponse
from app.services.question_generation import (
    generate_questions_batch,
    generate_questions_batch_stream,
    QuestionBatchRequest,
    QuestionGenerationResult,
    QuestionProgressEvent,
    MaterialInfo,
    LearningGoalInfo,
    QuestionSourceConfig,
    DifficultyDistribution,
)

router = APIRouter()


def _map_difficulty_to_db(value: str | None) -> str | None:
    if value is None:
        return None
    if value in {"beginner", "intermediate", "advanced"}:
        return value
    return {
        "easy": "beginner",
        "medium": "intermediate",
        "hard": "advanced",
    }.get(value, value)


def _format_answer_for_db(answer: object | None) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, list):
        import json

        return json.dumps(answer, ensure_ascii=False)
    if isinstance(answer, str):
        return answer
    return str(answer)


def _format_jsonish_for_db(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        import json

        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.lower() in {"null", "none"}:
            return None
        return value
    return str(value)


@router.get("", response_model=list[QuestionResponse])
async def get_questions(
    nodeId: str | None = Query(None),
    topicId: str | None = Query(None),
    includeNoTopic: bool | None = Query(False),
    isFavorite: bool | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取练习题列表"""
    query = db.query(Question).filter(Question.userId == current_user.id)
    if nodeId:
        query = query.filter(Question.nodeId == nodeId)
    if topicId:
        query = query.join(KnowledgeNode, KnowledgeNode.id == Question.nodeId).filter(
            KnowledgeNode.userId == current_user.id
        )
        if includeNoTopic:
            query = query.filter(or_(KnowledgeNode.topicId == topicId, KnowledgeNode.topicId.is_(None)))
        else:
            query = query.filter(KnowledgeNode.topicId == topicId)
    if isFavorite is not None:
        query = query.filter(Question.isFavorite == isFavorite)
    return query.order_by(Question.createdAt.desc()).all()


@router.get("/{question_id}", response_model=QuestionResponse)
async def get_question(
    question_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取单个题目"""
    question = db.query(Question).filter(
        Question.id == question_id,
        Question.userId == current_user.id,
    ).first()

    if not question:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="题目不存在")
    return question


@router.post("", response_model=QuestionResponse)
async def create_question(
    data: QuestionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建题目"""
    if data.id:
        if len(data.id) > 191:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="题目 ID 过长")
        existing = db.query(Question).filter(Question.id == data.id).first()
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="题目 ID 已存在")

    node = db.query(KnowledgeNode).filter(
        KnowledgeNode.id == data.nodeId,
        KnowledgeNode.userId == current_user.id,
    ).first()
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识点不存在")

    difficulty_db = _map_difficulty_to_db(data.difficulty) or "beginner"

    parent_question_id = data.parentQuestionId or data.derivedFromQuestionId
    derived_from_question_id = data.derivedFromQuestionId or data.parentQuestionId
    is_derived = bool(data.isDerivedQuestion) if data.isDerivedQuestion is not None else bool(parent_question_id)
    derivation_type = data.derivationType
    if is_derived and not derivation_type:
        derivation_type = "derived"

    question_kwargs = {
        "userId": current_user.id,
        "nodeId": data.nodeId,
        "type": data.type,
        "content": data.content,
        "options": data.options,
        "answer": _format_jsonish_for_db(data.answer),
        "explanation": data.explanation,
        "difficulty": difficulty_db,
        "hints": _format_jsonish_for_db(data.hints),
        "relatedConcepts": _format_jsonish_for_db(data.relatedConcepts),
        "targetGoalIds": data.targetGoalIds,
        "targetGoalNames": data.targetGoalNames,
        "isFavorite": bool(data.isFavorite) if data.isFavorite is not None else False,
        "userNotes": data.userNotes,
        "tags": data.tags,
        # 题目来源
        "source": data.source,
        "sourceMaterialIds": data.sourceMaterialIds,
        "sourceMaterialNames": data.sourceMaterialNames,
        "sourceContext": data.sourceContext,
        "difficultyReason": data.difficultyReason,
        # 衍生题关联
        "derivedFromQuestionId": derived_from_question_id,
        "derivedFromRecordId": data.derivedFromRecordId,
        "isDerivedQuestion": is_derived,
        "parentQuestionId": parent_question_id,
        "derivationType": derivation_type,
    }
    if data.createdAt is not None:
        question_kwargs["createdAt"] = data.createdAt
        question_kwargs["updatedAt"] = data.createdAt
    if data.id:
        question_kwargs["id"] = data.id

    question = Question(**question_kwargs)
    db.add(question)
    db.commit()
    db.refresh(question)
    return question


@router.patch("/{question_id}", response_model=QuestionResponse)
async def update_question(
    question_id: str,
    data: QuestionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新题目"""
    question = db.query(Question).filter(
        Question.id == question_id,
        Question.userId == current_user.id,
    ).first()
    if not question:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="题目不存在")

    if data.type is not None:
        question.type = data.type
    if data.content is not None:
        question.content = data.content
    if data.options is not None:
        question.options = data.options
    if data.answer is not None:
        question.answer = _format_jsonish_for_db(data.answer)
    if data.explanation is not None:
        question.explanation = data.explanation
    if data.difficulty is not None:
        question.difficulty = _map_difficulty_to_db(data.difficulty)

    if data.isFavorite is not None:
        question.isFavorite = bool(data.isFavorite)

    if data.targetGoalIds is not None:
        question.targetGoalIds = data.targetGoalIds
    if data.hints is not None:
        question.hints = _format_jsonish_for_db(data.hints)
    if data.relatedConcepts is not None:
        question.relatedConcepts = _format_jsonish_for_db(data.relatedConcepts)

    if data.parentQuestionId is not None or data.derivedFromQuestionId is not None:
        parent_id = data.parentQuestionId or data.derivedFromQuestionId
        question.parentQuestionId = parent_id
        if hasattr(question, "derivedFromQuestionId"):
            question.derivedFromQuestionId = parent_id
        question.isDerivedQuestion = bool(parent_id)
        if question.isDerivedQuestion and not question.derivationType:
            question.derivationType = "derived"

    if data.isDerivedQuestion is not None:
        question.isDerivedQuestion = bool(data.isDerivedQuestion)
        if question.isDerivedQuestion:
            question.derivationType = data.derivationType or question.derivationType or "derived"
        else:
            question.derivationType = None
    elif data.derivationType is not None:
        question.derivationType = data.derivationType
        question.isDerivedQuestion = True

    db.commit()
    db.refresh(question)
    return question


@router.delete("/{question_id}")
async def delete_question(
    question_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除题目"""
    question = db.query(Question).filter(
        Question.id == question_id,
        Question.userId == current_user.id,
    ).first()
    if not question:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="题目不存在")

    db.delete(question)
    db.commit()
    return {"success": True}


# ==================== 并行生成题目 API ====================

class MaterialInfoRequest(BaseModel):
    """资料信息请求"""
    id: str
    name: str
    content: str = ""
    contentDigest: str = ""
    keyTopics: list[str] = Field(default_factory=list)
    structuredSummary: str = ""
    structuredKeyPoints: list[dict] = Field(default_factory=list)
    questionablePoints: list[str] = Field(default_factory=list)


class LearningGoalRequest(BaseModel):
    """学习目标请求"""
    id: str
    goal: str
    importance: str = "medium"
    subGoals: list[str] = Field(default_factory=list)


class QuestionSourceConfigRequest(BaseModel):
    """出题来源配置"""
    useNodeContent: bool = True
    useMaterials: bool = True
    useLearningGoals: bool = False
    selectedMaterialIds: list[str] = Field(default_factory=list)
    selectedGoalIds: list[str] = Field(default_factory=list)


class DifficultyDistributionRequest(BaseModel):
    """难度分布配置"""
    easy: int = 30
    medium: int = 50
    hard: int = 20


class GenerateQuestionsRequest(BaseModel):
    """并行生成题目请求 - 支持多来源"""
    # 节点信息
    nodeId: str
    nodeName: str
    nodeDescription: str = ""
    knowledgeType: str = "concept"
    nodeDifficulty: str = "beginner"
    learningObjectives: list[str] = Field(default_factory=list)
    keyConcepts: list[str] = Field(default_factory=list)
    commonMistakes: list[str] = Field(default_factory=list)

    # 资料列表
    materials: list[MaterialInfoRequest] = Field(default_factory=list)

    # 学习目标列表
    learningGoals: list[LearningGoalRequest] = Field(default_factory=list)

    # 出题配置
    sources: QuestionSourceConfigRequest = Field(default_factory=QuestionSourceConfigRequest)
    difficulty: DifficultyDistributionRequest = Field(default_factory=DifficultyDistributionRequest)
    count: int = 5
    questionTypes: list[str] = Field(default_factory=lambda: ["single", "judge"])
    mode: str = "normal"  # normal / review / advance

    # 🎯 用户学习画像（用于个性化出题）
    learnerProfilePrompt: str = ""

    # 兼容旧字段
    materialsContent: str = ""

    # API 选择
    useSystemKey: bool | None = None


@router.post("/generate", response_model=QuestionGenerationResult)
async def generate_questions(
    payload: GenerateQuestionsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    并行生成题目

    - 官方 API (useSystemKey=true): 使用两阶段并行架构（规划 + 并发生成）
    - 用户自配 API (useSystemKey=false): 使用单次生成方式
    """
    # 验证节点存在
    node = db.query(KnowledgeNode).filter(
        KnowledgeNode.id == payload.nodeId,
        KnowledgeNode.userId == current_user.id,
    ).first()
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识点不存在")

    # 转换资料列表
    materials = [
        MaterialInfo(
            id=m.id,
            name=m.name,
            content=m.content,
            contentDigest=m.contentDigest,
            keyTopics=m.keyTopics,
            structuredSummary=m.structuredSummary,
            structuredKeyPoints=m.structuredKeyPoints,
            questionablePoints=m.questionablePoints,
        )
        for m in payload.materials
    ]

    # 转换学习目标列表
    learning_goals = [
        LearningGoalInfo(
            id=g.id,
            goal=g.goal,
            importance=g.importance,
            subGoals=g.subGoals,
        )
        for g in payload.learningGoals
    ]

    # 转换出题配置
    sources = QuestionSourceConfig(
        useNodeContent=payload.sources.useNodeContent,
        useMaterials=payload.sources.useMaterials,
        useLearningGoals=payload.sources.useLearningGoals,
        selectedMaterialIds=payload.sources.selectedMaterialIds,
        selectedGoalIds=payload.sources.selectedGoalIds,
    )

    difficulty = DifficultyDistribution(
        easy=payload.difficulty.easy,
        medium=payload.difficulty.medium,
        hard=payload.difficulty.hard,
    )

    request = QuestionBatchRequest(
        nodeId=payload.nodeId,
        nodeName=payload.nodeName,
        nodeDescription=payload.nodeDescription,
        knowledgeType=payload.knowledgeType,
        nodeDifficulty=payload.nodeDifficulty,
        learningObjectives=payload.learningObjectives,
        keyConcepts=payload.keyConcepts,
        commonMistakes=payload.commonMistakes,
        materials=materials,
        learningGoals=learning_goals,
        sources=sources,
        difficulty=difficulty,
        count=payload.count,
        questionTypes=payload.questionTypes,
        mode=payload.mode,
        learnerProfilePrompt=payload.learnerProfilePrompt,  # 🎯 用户学习画像
        materialsContent=payload.materialsContent,
    )

    result = await generate_questions_batch(
        db=db,
        user_id=current_user.id,
        request=request,
        use_system=payload.useSystemKey,
    )

    return result


@router.post("/generate/stream")
async def generate_questions_stream(
    payload: GenerateQuestionsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    流式并行生成题目（带进度反馈）

    返回 SSE 流，包含进度事件和最终结果
    """
    # 验证节点存在
    node = db.query(KnowledgeNode).filter(
        KnowledgeNode.id == payload.nodeId,
        KnowledgeNode.userId == current_user.id,
    ).first()
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识点不存在")

    # 转换资料列表
    materials = [
        MaterialInfo(
            id=m.id,
            name=m.name,
            content=m.content,
            contentDigest=m.contentDigest,
            keyTopics=m.keyTopics,
            structuredSummary=m.structuredSummary,
            structuredKeyPoints=m.structuredKeyPoints,
            questionablePoints=m.questionablePoints,
        )
        for m in payload.materials
    ]

    # 转换学习目标列表
    learning_goals = [
        LearningGoalInfo(
            id=g.id,
            goal=g.goal,
            importance=g.importance,
            subGoals=g.subGoals,
        )
        for g in payload.learningGoals
    ]

    # 转换出题配置
    sources = QuestionSourceConfig(
        useNodeContent=payload.sources.useNodeContent,
        useMaterials=payload.sources.useMaterials,
        useLearningGoals=payload.sources.useLearningGoals,
        selectedMaterialIds=payload.sources.selectedMaterialIds,
        selectedGoalIds=payload.sources.selectedGoalIds,
    )

    difficulty = DifficultyDistribution(
        easy=payload.difficulty.easy,
        medium=payload.difficulty.medium,
        hard=payload.difficulty.hard,
    )

    request = QuestionBatchRequest(
        nodeId=payload.nodeId,
        nodeName=payload.nodeName,
        nodeDescription=payload.nodeDescription,
        knowledgeType=payload.knowledgeType,
        nodeDifficulty=payload.nodeDifficulty,
        learningObjectives=payload.learningObjectives,
        keyConcepts=payload.keyConcepts,
        commonMistakes=payload.commonMistakes,
        materials=materials,
        learningGoals=learning_goals,
        sources=sources,
        difficulty=difficulty,
        count=payload.count,
        questionTypes=payload.questionTypes,
        mode=payload.mode,
        learnerProfilePrompt=payload.learnerProfilePrompt,  # 🎯 用户学习画像
        materialsContent=payload.materialsContent,
    )

    async def event_generator():
        async for event in generate_questions_batch_stream(
            db=db,
            user_id=current_user.id,
            request=request,
            use_system=payload.useSystemKey,
        ):
            if isinstance(event, QuestionProgressEvent):
                yield f"data: {json.dumps({'type': 'progress', 'data': event.model_dump()}, ensure_ascii=False)}\n\n"
            elif isinstance(event, QuestionGenerationResult):
                yield f"data: {json.dumps({'type': 'result', 'data': event.model_dump()}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
