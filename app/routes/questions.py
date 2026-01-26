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
from app.services.smart_question_service import (
    SmartQuestionService,
    SmartQuestionRequest,
    LearnerProfile,
    LearningGoal as SmartLearningGoal,
    MaterialInfo as SmartMaterialInfo,
    ConceptMastery,
    GeneratedQuestion,
)
from app.core.config import settings

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


# ==================== 智能出题 API (V2) ====================

class ConceptMasteryRequest(BaseModel):
    """概念掌握度"""
    concept: str
    masteryRate: float = Field(ge=0, le=100, alias="mastery_rate")
    isWeak: bool = Field(alias="is_weak")
    totalAttempts: int = Field(default=0, alias="total_attempts")

    class Config:
        populate_by_name = True


class LearnerProfileRequest(BaseModel):
    """用户学习画像"""
    totalQuestions: int = Field(default=0, alias="total_questions")
    correctRate: float = Field(default=0, ge=0, le=100, alias="correct_rate")
    weakConcepts: list[ConceptMasteryRequest] = Field(default_factory=list, alias="weak_concepts")
    masteredConcepts: list[str] = Field(default_factory=list, alias="mastered_concepts")

    class Config:
        populate_by_name = True


class SmartLearningGoalRequest(BaseModel):
    """学习目标请求"""
    id: str
    goal: str
    importance: str = "should"
    relatedConcepts: list[str] = Field(default_factory=list, alias="related_concepts")
    masteryScore: float = Field(default=0, ge=0, le=100, alias="mastery_score")

    class Config:
        populate_by_name = True


class SmartMaterialRequest(BaseModel):
    """资料请求"""
    id: str
    name: str
    contentDigest: str = Field(default="", alias="content_digest")
    keyTopics: list[str] = Field(default_factory=list, alias="key_topics")

    class Config:
        populate_by_name = True


class SmartGenerateRequest(BaseModel):
    """智能出题请求"""
    # 节点信息
    nodeId: str = Field(alias="node_id")
    nodeName: str = Field(alias="node_name")
    nodeDescription: str = Field(default="", alias="node_description")
    keyConcepts: list[str] = Field(default_factory=list, alias="key_concepts")

    # 出题配置
    count: int = Field(default=5, ge=1, le=20)
    mode: str = "normal"  # normal / review / advance
    questionTypes: list[str] = Field(default_factory=lambda: ["single", "judge"], alias="question_types")

    # 难度分布（百分比）
    difficultyEasy: int = Field(default=30, ge=0, le=100, alias="difficulty_easy")
    difficultyMedium: int = Field(default=50, ge=0, le=100, alias="difficulty_medium")
    difficultyHard: int = Field(default=20, ge=0, le=100, alias="difficulty_hard")

    # 出题来源
    useNodeContent: bool = Field(default=True, alias="use_node_content")
    useMaterials: bool = Field(default=True, alias="use_materials")
    useLearningGoals: bool = Field(default=True, alias="use_learning_goals")

    # 资料列表
    materials: list[SmartMaterialRequest] = Field(default_factory=list)

    # 学习目标列表
    learningGoals: list[SmartLearningGoalRequest] = Field(default_factory=list, alias="learning_goals")

    # 用户画像
    learnerProfile: LearnerProfileRequest | None = Field(default=None, alias="learner_profile")

    class Config:
        populate_by_name = True


class SmartGenerateResponse(BaseModel):
    """智能出题响应"""
    success: bool = True
    questions: list[dict] = Field(default_factory=list)
    total: int = 0
    mode: str = "normal"
    profile_used: bool = False
    goals_targeted: int = 0
    weaknesses_targeted: int = 0


@router.post("/generate/smart", response_model=SmartGenerateResponse)
async def generate_smart_questions(
    payload: SmartGenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    智能出题 V2 API

    特性：
    1. 根据用户画像过滤已掌握知识点
    2. 针对薄弱点优先出题（约70%）
    3. 强制关联学习目标
    4. 并发生成提高效率
    5. 使用 LangChain with_structured_output 确保输出格式
    """
    # 验证节点存在
    node = db.query(KnowledgeNode).filter(
        KnowledgeNode.id == payload.nodeId,
        KnowledgeNode.userId == current_user.id,
    ).first()
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识点不存在")

    # 获取 API 配置（使用系统内置 LLM 配置）
    api_key = settings.SYSTEM_LLM_API_KEY or settings.DASHSCOPE_API_KEY
    base_url = settings.SYSTEM_LLM_BASE_URL
    model_id = settings.SYSTEM_LLM_MODEL

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="未配置 AI API，请联系管理员"
        )

    # 转换用户画像
    learner_profile = None
    if payload.learnerProfile:
        weak_concepts = [
            ConceptMastery(
                concept=w.concept,
                mastery_rate=w.masteryRate,
                is_weak=w.isWeak,
                total_attempts=w.totalAttempts,
            )
            for w in payload.learnerProfile.weakConcepts
        ]
        learner_profile = LearnerProfile(
            total_questions=payload.learnerProfile.totalQuestions,
            correct_rate=payload.learnerProfile.correctRate,
            weak_concepts=weak_concepts,
            mastered_concepts=payload.learnerProfile.masteredConcepts,
        )

    # 转换学习目标
    learning_goals = [
        SmartLearningGoal(
            id=g.id,
            goal=g.goal,
            importance=g.importance,
            related_concepts=g.relatedConcepts,
            mastery_score=g.masteryScore,
        )
        for g in payload.learningGoals
    ]
    print(f"[SmartQuestion API] 接收到 {len(learning_goals)} 个学习目标: {[g.goal[:20] for g in learning_goals]}")

    # 转换资料
    materials = [
        SmartMaterialInfo(
            id=m.id,
            name=m.name,
            content_digest=m.contentDigest,
            key_topics=m.keyTopics,
        )
        for m in payload.materials
    ]

    # 构建服务请求
    request = SmartQuestionRequest(
        node_id=payload.nodeId,
        node_name=payload.nodeName,
        node_description=payload.nodeDescription,
        key_concepts=payload.keyConcepts,
        count=payload.count,
        mode=payload.mode,
        question_types=payload.questionTypes,
        difficulty_easy=payload.difficultyEasy,
        difficulty_medium=payload.difficultyMedium,
        difficulty_hard=payload.difficultyHard,
        use_node_content=payload.useNodeContent,
        use_materials=payload.useMaterials,
        use_learning_goals=payload.useLearningGoals,
        materials=materials,
        learning_goals=learning_goals,
        learner_profile=learner_profile,
    )

    # 调用智能出题服务
    service = SmartQuestionService(api_key, base_url, model_id)
    generated_questions = await service.generate_questions(request)

    # 统计
    goals_targeted = len(set(q.target_goal_id for q in generated_questions if q.target_goal_id))
    weaknesses_targeted = len(set(q.target_weakness for q in generated_questions if q.target_weakness))

    # 转换为响应格式
    questions_data = []
    for q in generated_questions:
        questions_data.append({
            "type": q.type,
            "difficulty": q.difficulty,
            "content": q.content,
            "options": q.options,
            "answer": q.answer,
            "explanation": q.explanation,
            "targetGoalIds": [q.target_goal_id] if q.target_goal_id else [],
            "targetGoalNames": [q.target_goal_name] if q.target_goal_name else [],
            "targetWeakness": q.target_weakness,
            "trace": q.trace.model_dump() if q.trace else None,
            "tags": q.tags.model_dump() if q.tags else None,
            "nodeId": payload.nodeId,
        })

    return SmartGenerateResponse(
        success=True,
        questions=questions_data,
        total=len(questions_data),
        mode=payload.mode,
        profile_used=learner_profile is not None,
        goals_targeted=goals_targeted,
        weaknesses_targeted=weaknesses_targeted,
    )


# ==================== V3 智能出题 API (基于RAG和Embedding) ====================

class V3LearningGoalRequest(BaseModel):
    """V3 学习目标请求"""
    id: str
    goal: str
    importance: str = "should"
    relatedConcepts: list[str] = Field(default_factory=list)
    masteryScore: float = Field(default=0, ge=0, le=100)


class V3LearnerProfileRequest(BaseModel):
    """V3 用户画像请求"""
    totalQuestions: int = Field(default=0)
    correctRate: float = Field(default=0, ge=0, le=100)
    weakConcepts: list[dict] = Field(default_factory=list)
    masteredConcepts: list[str] = Field(default_factory=list)


class V3GenerateRequest(BaseModel):
    """V3 智能出题请求"""
    nodeId: str
    materialIds: list[str] = Field(default_factory=list)
    learningGoals: list[V3LearningGoalRequest] = Field(default_factory=list)
    learnerProfile: V3LearnerProfileRequest | None = None

    # 出题配置
    count: int = Field(default=5, ge=1, le=20)
    mode: str = "normal"
    questionTypes: list[str] = Field(default_factory=lambda: ["single", "multiple"])

    # 难度分布
    difficultyEasy: int = Field(default=30, ge=0, le=100)
    difficultyMedium: int = Field(default=50, ge=0, le=100)
    difficultyHard: int = Field(default=20, ge=0, le=100)


class V3GenerateResponse(BaseModel):
    """V3 智能出题响应"""
    success: bool = True
    questions: list[dict] = Field(default_factory=list)
    total: int = 0
    mode: str = "normal"
    ragEnabled: bool = True
    goalsUsed: int = 0
    chunksRetrieved: int = 0


@router.post("/generate/v3", response_model=V3GenerateResponse)
async def generate_questions_v3(
    payload: V3GenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    V3 智能出题 API（基于RAG和Embedding）

    核心流程：
    1. 学习目标向量化
    2. RAG检索相关资料片段
    3. 构建精简上下文
    4. LLM生成题目（带检索trace）

    特性：
    - 使用阿里云 DashScope text-embedding-v4 进行向量化
    - 基于学习目标检索相关资料片段（而非全量资料）
    - 每道题都带有 ragTrace（检索追踪）
    - 上下文精简，token 消耗低
    """
    from app.services.question_generator_v3 import QuestionGeneratorV3

    # 验证节点存在
    node = db.query(KnowledgeNode).filter(
        KnowledgeNode.id == payload.nodeId,
        KnowledgeNode.userId == current_user.id,
    ).first()
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识点不存在")

    # 检查配置
    if not settings.DASHSCOPE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DASHSCOPE_API_KEY 未配置，无法使用 V3 出题"
        )

    # 转换学习目标
    goals = [
        {
            "id": g.id,
            "goal": g.goal,
            "importance": g.importance,
            "relatedConcepts": g.relatedConcepts,
            "masteryScore": g.masteryScore,
        }
        for g in payload.learningGoals
    ]

    # 转换用户画像
    learner_profile = None
    if payload.learnerProfile:
        learner_profile = {
            "totalQuestions": payload.learnerProfile.totalQuestions,
            "correctRate": payload.learnerProfile.correctRate,
            "weakConcepts": payload.learnerProfile.weakConcepts,
            "masteredConcepts": payload.learnerProfile.masteredConcepts,
        }

    # 构建配置
    config = {
        "count": payload.count,
        "mode": payload.mode,
        "questionTypes": payload.questionTypes,
        "difficulty": {
            "easy": payload.difficultyEasy,
            "medium": payload.difficultyMedium,
            "hard": payload.difficultyHard,
        },
    }

    try:
        # 调用 V3 生成器
        generator = QuestionGeneratorV3(db=db, user_id=current_user.id)
        questions = await generator.generate_questions_v3(
            node_id=payload.nodeId,
            goals=goals,
            material_ids=payload.materialIds,
            config=config,
            learner_profile=learner_profile,
        )

        # 统计
        chunks_retrieved = sum(
            len(q.get("ragTrace", {}).get("retrievedMaterialIds", []))
            for q in questions
        )

        return V3GenerateResponse(
            success=True,
            questions=questions,
            total=len(questions),
            mode=payload.mode,
            ragEnabled=True,
            goalsUsed=len(goals),
            chunksRetrieved=chunks_retrieved,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"V3 出题失败: {str(e)}"
        )
