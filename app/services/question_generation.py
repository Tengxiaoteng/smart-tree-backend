"""
题目并行生成服务 - 支持多来源出题

使用 LangChain + Pydantic 结构化输出，确保 AI 生成内容可追溯、可检索。

策略：
- 官方 API (system mode): 使用两阶段并行架构
  - 阶段1: 规划题目分布（题型、难度、考点）
  - 阶段2: 并发生成每道题目
- 用户自配 API (byok mode): 使用单次生成方式
"""
import asyncio
import json
import re
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.services.llm_context import resolve_llm_config


# ==================== AI 元数据 Schema（用于追溯和检索）====================

class AIGenerationMeta(BaseModel):
    """AI 生成元数据 - 记录生成过程信息，便于追溯和分析"""
    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        alias="generatedAt",
        description="生成时间 ISO 格式"
    )
    model_id: str = Field(default="", alias="modelId", description="使用的模型 ID")
    strategy: str = Field(default="parallel", description="生成策略: parallel / single")
    api_mode: str = Field(default="system", alias="apiMode", description="API 模式: system / byok")
    prompt_version: str = Field(default="v3.0", alias="promptVersion", description="Prompt 版本号")
    generation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        alias="generationId",
        description="本次生成的唯一标识"
    )

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class AIContentTrace(BaseModel):
    """AI 内容追溯信息 - 记录内容来源和生成依据"""
    source_type: str = Field(
        default="node_content", alias="sourceType",
        description="来源类型: node_content / user_material / learning_goals / mixed"
    )
    source_id: str = Field(default="", alias="sourceId", description="来源 ID")
    source_name: str = Field(default="", alias="sourceName", description="来源名称")
    source_context: str = Field(default="", alias="sourceContext", description="出题依据的上下文摘要")
    confidence: float = Field(default=0.8, description="AI 对该内容的置信度 0-1")

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class AISearchableTags(BaseModel):
    """AI 可检索标签 - 用于后续智能检索和推荐"""
    concepts: list[str] = Field(default_factory=list, description="涉及的核心概念")
    skills: list[str] = Field(default_factory=list, description="考察的能力/技能标签")
    bloom_level: str = Field(
        default="understand", alias="bloomLevel",
        description="布鲁姆认知层次: remember/understand/apply/analyze/evaluate/create"
    )
    keywords: list[str] = Field(default_factory=list, description="关键词列表，用于全文检索")
    similar_question_hints: list[str] = Field(
        default_factory=list, alias="similarQuestionHints",
        description="相似题目的出题方向提示"
    )

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


# ==================== 题目 Schema ====================

class QuestionSchema(BaseModel):
    """
    单道题目结构 - 包含完整的 AI 生成元数据

    设计原则：
    1. 所有 AI 生成的内容都留下可追溯的元数据
    2. 便于后续智能检索、推荐、变式题生成
    3. 使用 Pydantic 确保输出格式稳定
    """
    # 基础字段
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Literal["single", "multiple", "judge"] = "single"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    content: str = Field(description="题目内容，支持 LaTeX 公式")
    options: list[str] = Field(default_factory=list, description="选项列表")
    answer: str | list[str] = Field(default="", description="正确答案")
    explanation: str = Field(default="", description="答案解析")

    # AI 内容追溯
    trace: AIContentTrace = Field(
        default_factory=AIContentTrace,
        description="内容来源追溯信息"
    )

    # AI 可检索标签
    tags: AISearchableTags = Field(
        default_factory=AISearchableTags,
        description="可检索标签，用于智能推荐"
    )

    # AI 生成元数据
    ai_meta: AIGenerationMeta = Field(
        default_factory=AIGenerationMeta,
        alias="aiMeta",
        description="AI 生成过程元数据"
    )

    # 难度分析
    difficulty_reason: str = Field(default="", alias="difficultyReason", description="难度判定原因")

    # 学习目标关联
    target_goal_id: str | None = Field(None, alias="targetGoalId", description="关联的学习目标 ID")
    target_goal_name: str | None = Field(None, alias="targetGoalName", description="关联的学习目标名称")

    # 变式题提示（用于后续生成相似题）
    variation_hints: list[str] = Field(
        default_factory=list,
        alias="variationHints",
        description="变式题生成提示"
    )

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class QuestionPlanItem(BaseModel):
    """题目规划项"""
    index: int
    type: Literal["single", "multiple", "judge"] = "single"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    focus_point: str = Field(alias="focusPoint")  # 考察重点
    question_direction: str = Field(alias="questionDirection")  # 出题方向
    source_type: str = Field("node_content", alias="sourceType")  # 来源类型
    source_name: str = Field("", alias="sourceName")  # 来源名称

    model_config = ConfigDict(populate_by_name=True)


class MaterialInfo(BaseModel):
    """资料信息"""
    id: str
    name: str
    content: str = ""
    content_digest: str = Field("", alias="contentDigest")
    key_topics: list[str] = Field(default_factory=list, alias="keyTopics")
    # 结构化内容
    structured_summary: str = Field("", alias="structuredSummary")
    structured_key_points: list[dict] = Field(default_factory=list, alias="structuredKeyPoints")
    questionable_points: list[str] = Field(default_factory=list, alias="questionablePoints")

    model_config = ConfigDict(populate_by_name=True)


class LearningGoalInfo(BaseModel):
    """学习目标信息"""
    id: str
    goal: str
    importance: str = "medium"  # high/medium/low
    sub_goals: list[str] = Field(default_factory=list, alias="subGoals")

    model_config = ConfigDict(populate_by_name=True)


class QuestionSourceConfig(BaseModel):
    """出题来源配置"""
    use_node_content: bool = Field(True, alias="useNodeContent")
    use_materials: bool = Field(True, alias="useMaterials")
    use_learning_goals: bool = Field(False, alias="useLearningGoals")
    selected_material_ids: list[str] = Field(default_factory=list, alias="selectedMaterialIds")
    selected_goal_ids: list[str] = Field(default_factory=list, alias="selectedGoalIds")

    model_config = ConfigDict(populate_by_name=True)


class DifficultyDistribution(BaseModel):
    """难度分布配置"""
    easy: int = 30
    medium: int = 50
    hard: int = 20


class QuestionBatchRequest(BaseModel):
    """批量生成请求 - 支持多来源"""
    # 节点信息
    node_id: str = Field(alias="nodeId")
    node_name: str = Field(alias="nodeName")
    node_description: str = Field("", alias="nodeDescription")
    knowledge_type: str = Field("concept", alias="knowledgeType")
    node_difficulty: str = Field("beginner", alias="nodeDifficulty")
    learning_objectives: list[str] = Field(default_factory=list, alias="learningObjectives")
    key_concepts: list[str] = Field(default_factory=list, alias="keyConcepts")
    common_mistakes: list[str] = Field(default_factory=list, alias="commonMistakes")

    # 资料列表
    materials: list[MaterialInfo] = Field(default_factory=list)

    # 学习目标列表
    learning_goals: list[LearningGoalInfo] = Field(default_factory=list, alias="learningGoals")

    # 出题配置
    sources: QuestionSourceConfig = Field(default_factory=QuestionSourceConfig)
    difficulty: DifficultyDistribution = Field(default_factory=DifficultyDistribution)
    count: int = 5
    question_types: list[str] = Field(default_factory=lambda: ["single", "judge"], alias="questionTypes")
    mode: str = "normal"  # normal / review / advance

    # 🎯 用户学习画像（用于个性化出题）
    learner_profile_prompt: str = Field("", alias="learnerProfilePrompt")

    # 兼容旧字段
    materials_content: str = Field("", alias="materialsContent")

    model_config = ConfigDict(populate_by_name=True)


class QuestionProgressEvent(BaseModel):
    """进度事件"""
    stage: str  # planning / generating / done
    progress: float  # 0-100
    message: str
    current_question: int = 0
    total_questions: int = 0
    completed_questions: int = 0


class QuestionGenerationResult(BaseModel):
    """题目生成结果"""
    success: bool
    questions: list[QuestionSchema] = Field(default_factory=list)
    error: str | None = None
    mode: Literal["system", "byok"]
    strategy: Literal["parallel", "single_request"]
    processing_time_ms: float
    model_used: str


# ==================== LLM Client ====================

class LLMClient:
    """统一的 LLM 调用客户端"""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        if "chat/completions" not in self.base_url:
            self.chat_url = f"{self.base_url}/chat/completions"
        else:
            self.chat_url = self.base_url

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        timeout = httpx.Timeout(120.0, connect=10.0)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                self.chat_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json=payload,
            )

        if resp.status_code >= 400:
            raise Exception(f"LLM API 请求失败 ({resp.status_code}): {resp.text}")

        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def _extract_json(content: str) -> dict | list | None:
    """从 LLM 响应中提取 JSON"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 尝试提取 markdown 代码块中的 JSON
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试提取裸 JSON 对象或数组
    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        match = re.search(pattern, content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return None


# ==================== 上下文构建辅助函数 ====================

def _build_context_from_request(request: QuestionBatchRequest) -> tuple[str, list[dict]]:
    """
    根据请求配置构建出题上下文

    Returns:
        (context_text, source_info_list)
    """
    context_parts: list[str] = []
    source_info: list[dict] = []

    # 1. 节点知识内容
    if request.sources.use_node_content:
        node_context = f"""## 节点知识：{request.node_name}
{request.node_description or '无描述'}

知识类型：{request.knowledge_type}
难度等级：{request.node_difficulty}
学习目标：{', '.join(request.learning_objectives) or '无'}
关键概念：{', '.join(request.key_concepts) or '无'}
常见错误：{', '.join(request.common_mistakes) or '无'}"""
        context_parts.append(node_context)
        source_info.append({"type": "node_content", "name": request.node_name})

    # 2. 上传的资料
    if request.sources.use_materials and request.materials:
        selected_ids = set(request.sources.selected_material_ids) if request.sources.selected_material_ids else None
        selected_materials = [m for m in request.materials if selected_ids is None or m.id in selected_ids]

        for mat in selected_materials:
            if mat.structured_key_points:
                # 使用结构化内容
                key_points_text = "\n".join([
                    f"- {kp.get('title', '')}: {kp.get('content', '')}"
                    for kp in mat.structured_key_points[:5]
                ])
                mat_content = f"""## 资料：{mat.name}
摘要：{mat.structured_summary or mat.content_digest}

关键知识点：
{key_points_text}

可出题点：{', '.join(mat.questionable_points[:5]) if mat.questionable_points else '无'}"""
            else:
                # 使用原始内容
                mat_content = f"""## 资料：{mat.name}
{mat.content_digest or mat.content[:1500] if mat.content else '无内容'}"""

            context_parts.append(mat_content)
            source_info.append({"type": "user_material", "name": mat.name, "id": mat.id})

    # 3. 学习目标
    if request.sources.use_learning_goals and request.learning_goals:
        selected_ids = set(request.sources.selected_goal_ids) if request.sources.selected_goal_ids else None
        selected_goals = [g for g in request.learning_goals if selected_ids is None or g.id in selected_ids]

        if selected_goals:
            goals_text = "\n".join([f"- {g.goal} (重要程度: {g.importance})" for g in selected_goals])
            goals_context = f"""## 用户想掌握的学习目标
{goals_text}"""
            context_parts.append(goals_context)
            source_info.append({"type": "learning_goals", "name": "学习目标"})

    # 4. 🎯 用户学习画像（个性化出题核心）
    if request.learner_profile_prompt:
        context_parts.append(f"""---

{request.learner_profile_prompt}

**重要：请根据上述用户画像，优先针对薄弱概念出题，避免重复考察已掌握的内容。**""")
        source_info.append({"type": "learner_profile", "name": "用户学习画像"})

    # 兼容旧的 materials_content 字段
    if not context_parts and request.materials_content:
        context_parts.append(f"## 参考资料\n{request.materials_content[:3000]}")

    return "\n\n---\n\n".join(context_parts), source_info


# ==================== V3 两阶段并行策略（官方 API 专用）====================

class QuestionPlannerAgent:
    """规划 Agent - 规划题目分布（题型、难度、考点、来源）"""

    SYSTEM_PROMPT = """你是教育测评规划专家。根据学习内容，规划一组练习题的分布。

要求：
- 根据难度分布配置分配题目难度
- 每道题有明确的考察重点和来源
- 考察点要覆盖提供的各类学习内容

返回 JSON:
{"plans":[{"index":0,"type":"single","difficulty":"easy","focusPoint":"考察重点","questionDirection":"出题方向","sourceType":"node_content|user_material|learning_goals","sourceName":"来源名称"}]}

type 只能是: single(单选), judge(判断)
difficulty 只能是: easy, medium, hard"""

    def __init__(self, client: LLMClient):
        self.client = client
        self.model = "qwen-plus"

    async def plan(self, request: QuestionBatchRequest, context: str, source_info: list[dict]) -> list[QuestionPlanItem]:
        # 计算难度分布
        easy_count = round(request.count * request.difficulty.easy / 100)
        medium_count = round(request.count * request.difficulty.medium / 100)
        hard_count = request.count - easy_count - medium_count

        mode_label = {"normal": "综合练习", "review": "复习巩固", "advance": "进阶挑战"}.get(request.mode, "综合练习")

        user_prompt = f"""## 学习内容
{context[:4000]}

## 出题配置
- 生成数量：{request.count} 道题目
- 允许题型：{', '.join(request.question_types)}
- 出题模式：{mode_label}
- 难度分布：简单 {easy_count} 道，中等 {medium_count} 道，困难 {hard_count} 道
- 可用来源：{', '.join([s.get('name', '') for s in source_info])}

请规划题目分布，确保考察点覆盖各个来源的核心内容。"""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        result = await self.client.chat(self.model, messages, max_tokens=1500, temperature=0.3)
        parsed = _extract_json(result)

        if not parsed or not isinstance(parsed, dict):
            return self._default_plan(request, source_info)

        plans = parsed.get("plans", [])
        if not plans:
            return self._default_plan(request, source_info)

        return [
            QuestionPlanItem(
                index=p.get("index", i),
                type=p.get("type", "single") if p.get("type") in ["single", "judge"] else "single",
                difficulty=p.get("difficulty", "medium") if p.get("difficulty") in ["easy", "medium", "hard"] else "medium",
                focus_point=p.get("focusPoint", request.node_name),
                question_direction=p.get("questionDirection", f"考察{request.node_name}"),
                source_type=p.get("sourceType", "node_content"),
                source_name=p.get("sourceName", request.node_name),
            )
            for i, p in enumerate(plans[:request.count])
        ]

    def _default_plan(self, request: QuestionBatchRequest, source_info: list[dict]) -> list[QuestionPlanItem]:
        """生成默认规划"""
        plans = []
        # 根据难度分布计算
        easy_count = round(request.count * request.difficulty.easy / 100)
        medium_count = round(request.count * request.difficulty.medium / 100)

        for i in range(request.count):
            if i < easy_count:
                diff = "easy"
            elif i < easy_count + medium_count:
                diff = "medium"
            else:
                diff = "hard"

            # 轮流使用不同来源
            source = source_info[i % len(source_info)] if source_info else {"type": "node_content", "name": request.node_name}

            plans.append(QuestionPlanItem(
                index=i,
                type="single" if i % 2 == 0 else "judge",
                difficulty=diff,
                focus_point=request.key_concepts[i % len(request.key_concepts)] if request.key_concepts else request.node_name,
                question_direction=f"考察{request.node_name}的理解",
                source_type=source.get("type", "node_content"),
                source_name=source.get("name", request.node_name),
            ))
        return plans


class QuestionGeneratorAgent:
    """内容生成 Agent - 并发生成单道题目（带完整 AI 元数据）"""

    SYSTEM_PROMPT = """你是专业的教育测评专家，擅长根据学生的学习情况进行个性化出题。

## 题型说明
- single: 单选题，4个选项，只有1个正确答案
- judge: 判断题，选项为 ["A. 正确", "B. 错误"]

## 数学公式格式
- 使用 LaTeX 格式，行内公式用 $...$

## 个性化出题原则
- 如果有用户画像，优先针对薄弱概念出题
- 对已掌握的概念可以提高难度或减少出题
- 根据用户的错误模式设计针对性题目

## 输出格式（严格 JSON，包含完整元数据）
{
  "type": "single",
  "content": "题目内容",
  "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
  "answer": "A",
  "explanation": "【答案解析】...\\n【易错分析】...",
  "difficulty": "easy/medium/hard",
  "difficultyReason": "难度判定原因",
  "trace": {
    "sourceContext": "这道题基于哪部分内容出题",
    "confidence": 0.9
  },
  "tags": {
    "concepts": ["核心概念1", "核心概念2"],
    "skills": ["考察的能力"],
    "bloomLevel": "understand/apply/analyze",
    "keywords": ["关键词1", "关键词2"],
    "similarQuestionHints": ["可以从XX角度出变式题", "可以考察YY"]
  },
  "variationHints": ["变式1：改变条件...", "变式2：反向提问..."],
  "targetWeakness": "针对的薄弱点（如果有）"
}

只输出 JSON，不要其他文字。"""

    PROMPT_VERSION = "v3.2"  # 用于追踪 prompt 迭代（加入个性化出题）

    def __init__(self, client: LLMClient, max_concurrent: int = 8):
        self.client = client
        self.model = "qwen-max"
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.generation_id = str(uuid.uuid4())[:8]  # 本批次生成 ID

    async def generate_question(
        self,
        plan: QuestionPlanItem,
        request: QuestionBatchRequest,
        context: str,
    ) -> QuestionSchema:
        """生成单道题目（带限流和完整元数据）"""
        async with self.semaphore:
            mode_hint = {
                "review": "复习巩固模式：基础优先、易错点优先、避免过度刁钻",
                "advance": "进阶挑战模式：综合应用、多步骤推理、可包含变式题",
                "normal": "综合练习模式：覆盖核心知识点",
            }.get(request.mode, "")

            user_prompt = f"""## 学习内容
{context[:3000]}

## 出题要求
- 题型：{plan.type}
- 难度：{plan.difficulty}
- 考察重点：{plan.focus_point}
- 出题方向：{plan.question_direction}
- 来源：{plan.source_name}
- 模式提示：{mode_hint}

请基于上述学习内容，生成一道符合要求的题目。
要求：
1. 题目必须有明确依据，不要凭空编造
2. 填写完整的 tags 信息，便于后续检索
3. 提供 variationHints，便于生成变式题"""

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            try:
                result = await self.client.chat(self.model, messages, max_tokens=1200, temperature=0.7)
                parsed = _extract_json(result)

                if parsed and isinstance(parsed, dict):
                    return self._build_question_from_response(parsed, plan, request)
            except Exception as e:
                print(f"[QuestionGenerator] 生成失败: {e}")

            return self._default_question(plan, request)

    def _build_question_from_response(
        self,
        parsed: dict,
        plan: QuestionPlanItem,
        request: QuestionBatchRequest,
    ) -> QuestionSchema:
        """从 AI 响应构建完整的 QuestionSchema"""
        # 解析 trace
        trace_data = parsed.get("trace", {})
        trace = AIContentTrace(
            source_type=plan.source_type,
            source_id=trace_data.get("sourceId", ""),
            source_name=plan.source_name,
            source_context=trace_data.get("sourceContext", parsed.get("sourceContext", "")),
            confidence=trace_data.get("confidence", 0.8),
        )

        # 解析 tags
        tags_data = parsed.get("tags", {})
        tags = AISearchableTags(
            concepts=tags_data.get("concepts", parsed.get("relatedConcepts", [])),
            skills=tags_data.get("skills", []),
            bloom_level=tags_data.get("bloomLevel", "understand"),
            keywords=tags_data.get("keywords", []),
            similar_question_hints=tags_data.get("similarQuestionHints", []),
        )

        # 构建 AI 元数据
        ai_meta = AIGenerationMeta(
            model_id=self.model,
            strategy="parallel",
            api_mode="system",
            prompt_version=self.PROMPT_VERSION,
            generation_id=self.generation_id,
        )

        return QuestionSchema(
            id=str(uuid.uuid4()),
            type=parsed.get("type", plan.type),
            difficulty=parsed.get("difficulty", plan.difficulty),
            content=parsed.get("content", ""),
            options=parsed.get("options", []),
            answer=parsed.get("answer", ""),
            explanation=parsed.get("explanation", ""),
            trace=trace,
            tags=tags,
            ai_meta=ai_meta,
            difficulty_reason=parsed.get("difficultyReason", ""),
            target_goal_name=plan.focus_point,
            variation_hints=parsed.get("variationHints", []),
        )

    def _default_question(self, plan: QuestionPlanItem, request: QuestionBatchRequest) -> QuestionSchema:
        """生成默认题目（带完整元数据）"""
        # 构建默认元数据
        trace = AIContentTrace(
            source_type=plan.source_type,
            source_name=plan.source_name,
            source_context="默认生成",
            confidence=0.5,
        )
        tags = AISearchableTags(
            concepts=[plan.focus_point],
            keywords=[request.node_name, plan.focus_point],
        )
        ai_meta = AIGenerationMeta(
            model_id=self.model,
            strategy="parallel",
            api_mode="system",
            prompt_version=self.PROMPT_VERSION,
            generation_id=self.generation_id,
        )

        if plan.type == "judge":
            return QuestionSchema(
                id=str(uuid.uuid4()),
                type="judge",
                difficulty=plan.difficulty,
                content=f"判断：关于{request.node_name}，{plan.focus_point}的说法是正确的。",
                options=["A. 正确", "B. 错误"],
                answer="A",
                explanation=f"这道题考察{plan.focus_point}的理解。",
                trace=trace,
                tags=tags,
                ai_meta=ai_meta,
                target_goal_name=plan.focus_point,
            )
        return QuestionSchema(
            id=str(uuid.uuid4()),
            type="single",
            difficulty=plan.difficulty,
            content=f"关于{request.node_name}中的{plan.focus_point}，以下说法正确的是？",
            options=["A. 选项A", "B. 选项B", "C. 选项C", "D. 选项D"],
            answer="A",
            explanation=f"这道题考察{plan.focus_point}的理解。",
            trace=trace,
            tags=tags,
            ai_meta=ai_meta,
            target_goal_name=plan.focus_point,
        )



# ==================== 并行生成主函数 ====================

async def generate_questions_parallel(
    client: LLMClient,
    request: QuestionBatchRequest,
) -> list[QuestionSchema]:
    """V3 两阶段并行策略生成题目"""
    planner = QuestionPlannerAgent(client)
    generator = QuestionGeneratorAgent(client)

    # 构建上下文
    context, source_info = _build_context_from_request(request)

    # 阶段1: 规划题目分布
    plans = await planner.plan(request, context, source_info)

    # 阶段2: 并发生成每道题目
    async def generate_one(plan: QuestionPlanItem) -> tuple[int, QuestionSchema]:
        question = await generator.generate_question(plan, request, context)
        return plan.index, question

    tasks = [generate_one(plan) for plan in plans]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 组装结果（按顺序）
    questions_map = {}
    for result in results:
        if isinstance(result, tuple):
            idx, question = result
            questions_map[idx] = question

    return [questions_map[i] for i in sorted(questions_map.keys())]


async def generate_questions_parallel_stream(
    client: LLMClient,
    request: QuestionBatchRequest,
) -> AsyncGenerator[QuestionProgressEvent | list[QuestionSchema], None]:
    """V3 两阶段并行策略 - 流式版本"""
    planner = QuestionPlannerAgent(client)
    generator = QuestionGeneratorAgent(client)

    # 构建上下文
    context, source_info = _build_context_from_request(request)

    # 阶段1: 规划
    yield QuestionProgressEvent(
        stage="planning",
        progress=5,
        message="正在规划题目分布...",
        total_questions=request.count,
    )

    plans = await planner.plan(request, context, source_info)
    total = len(plans)

    yield QuestionProgressEvent(
        stage="planning",
        progress=15,
        message=f"规划完成，共 {total} 道题目",
        total_questions=total,
    )

    # 阶段2: 并发生成（带进度）
    completed = 0
    results = {}
    lock = asyncio.Lock()

    async def generate_with_progress(plan: QuestionPlanItem) -> tuple[int, QuestionSchema]:
        nonlocal completed
        question = await generator.generate_question(plan, request, context)
        async with lock:
            completed += 1
            results[plan.index] = question
        return plan.index, question

    tasks = [generate_with_progress(plan) for plan in plans]

    for coro in asyncio.as_completed(tasks):
        try:
            idx, _ = await coro
            progress = 15 + (completed / total) * 80
            yield QuestionProgressEvent(
                stage="generating",
                progress=progress,
                message=f"已生成第 {completed}/{total} 道题目",
                current_question=idx + 1,
                total_questions=total,
                completed_questions=completed,
            )
        except Exception as e:
            print(f"[QuestionStream] 生成失败: {e}")

    yield QuestionProgressEvent(
        stage="done",
        progress=100,
        message="题目生成完成",
        total_questions=total,
        completed_questions=total,
    )

    # 返回排序后的题目列表
    yield [results[i] for i in sorted(results.keys())]


# ==================== 单次生成策略（用户自配 API）====================

SINGLE_REQUEST_PROMPT = """你是专业的教育测评专家，擅长根据学生的学习情况进行个性化出题。

## 题型说明
- single: 单选题，4个选项，只有1个正确答案
- judge: 判断题，选项为 ["A. 正确", "B. 错误"]

## 数学公式格式
- 使用 LaTeX 格式，行内公式用 $...$

## 个性化出题原则
- 如果有用户画像，优先针对薄弱概念出题（约70%）
- 对已掌握的概念可以提高难度或减少出题
- 根据用户的错误模式设计针对性题目

## 输出格式（严格JSON数组，包含完整元数据）
[
  {
    "type": "single",
    "content": "题目内容",
    "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
    "answer": "A",
    "explanation": "答案解析",
    "difficulty": "easy|medium|hard",
    "difficultyReason": "难度判定原因",
    "trace": {
      "sourceContext": "出题依据",
      "confidence": 0.9
    },
    "tags": {
      "concepts": ["核心概念"],
      "skills": ["考察能力"],
      "bloomLevel": "understand",
      "keywords": ["关键词"]
    },
    "variationHints": ["变式题提示"],
    "targetWeakness": "针对的薄弱点（如果有）"
  }
]

只输出 JSON 数组，不要其他文字。"""

SINGLE_REQUEST_PROMPT_VERSION = "v3.2"


async def generate_questions_single_request(
    client: LLMClient,
    request: QuestionBatchRequest,
    model_id: str,
) -> list[QuestionSchema]:
    """单次请求策略（用户自配 API）- 带完整 AI 元数据"""
    # 构建上下文
    context, source_info = _build_context_from_request(request)
    generation_id = str(uuid.uuid4())[:8]

    mode_label = {"normal": "综合练习", "review": "复习巩固", "advance": "进阶挑战"}.get(request.mode, "综合练习")

    # 计算难度分布
    easy_count = round(request.count * request.difficulty.easy / 100)
    medium_count = round(request.count * request.difficulty.medium / 100)
    hard_count = request.count - easy_count - medium_count

    user_prompt = f"""## 学习内容
{context[:5000]}

## 出题要求
- 生成数量：{request.count} 道题目
- 题型包括：{', '.join(request.question_types)}
- 出题模式：{mode_label}
- 难度分布：简单 {easy_count} 道，中等 {medium_count} 道，困难 {hard_count} 道

请基于上述学习内容生成题目：
1. 每道题必须有明确依据，不要凭空编造
2. 填写完整的 tags 信息，便于后续检索
3. 提供 variationHints，便于生成变式题"""

    messages = [
        {"role": "system", "content": SINGLE_REQUEST_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    result = await client.chat(model_id, messages, max_tokens=8000, temperature=0.7)
    parsed = _extract_json(result)

    if not parsed or not isinstance(parsed, list):
        return []

    questions = []
    for i, q in enumerate(parsed[:request.count]):
        # 解析 trace
        trace_data = q.get("trace", {})
        source = source_info[i % len(source_info)] if source_info else {"type": "node_content", "name": request.node_name}
        trace = AIContentTrace(
            source_type=source.get("type", "node_content"),
            source_id=source.get("id", ""),
            source_name=source.get("name", request.node_name),
            source_context=trace_data.get("sourceContext", q.get("sourceContext", "")),
            confidence=trace_data.get("confidence", 0.8),
        )

        # 解析 tags
        tags_data = q.get("tags", {})
        tags = AISearchableTags(
            concepts=tags_data.get("concepts", q.get("relatedConcepts", [])),
            skills=tags_data.get("skills", []),
            bloom_level=tags_data.get("bloomLevel", "understand"),
            keywords=tags_data.get("keywords", []),
            similar_question_hints=tags_data.get("similarQuestionHints", []),
        )

        # 构建 AI 元数据
        ai_meta = AIGenerationMeta(
            model_id=model_id,
            strategy="single",
            api_mode="byok",
            prompt_version=SINGLE_REQUEST_PROMPT_VERSION,
            generation_id=generation_id,
        )

        questions.append(QuestionSchema(
            id=str(uuid.uuid4()),
            type=q.get("type", "single"),
            difficulty=q.get("difficulty", "medium"),
            content=q.get("content", ""),
            options=q.get("options", []),
            answer=q.get("answer", ""),
            explanation=q.get("explanation", ""),
            trace=trace,
            tags=tags,
            ai_meta=ai_meta,
            difficulty_reason=q.get("difficultyReason", ""),
            target_goal_name=q.get("targetGoalName"),
            variation_hints=q.get("variationHints", []),
        ))

    return questions


# ==================== 主入口 ====================

async def generate_questions_batch(
    db: Session,
    user_id: str,
    request: QuestionBatchRequest,
    use_system: bool | None = None,
) -> QuestionGenerationResult:
    """
    批量生成题目

    Args:
        db: 数据库会话
        user_id: 用户 ID
        request: 批量生成请求
        use_system: 是否使用系统 API（None 表示自动判断）

    Returns:
        QuestionGenerationResult: 生成结果
    """
    start_time = time.time()

    try:
        resolved = resolve_llm_config(
            db,
            user_id=user_id,
            requested_use_system=use_system,
            override_api_key=None,
            override_base_url=None,
            override_model_id=None,
            override_routing=None,
        )
    except Exception as e:
        return QuestionGenerationResult(
            success=False,
            error=str(e),
            mode="byok",
            strategy="single_request",
            processing_time_ms=(time.time() - start_time) * 1000,
            model_used="",
        )

    client = LLMClient(resolved.api_key, resolved.base_url)

    if resolved.mode == "system":
        # 官方 API: 使用两阶段并行策略
        try:
            questions = await generate_questions_parallel(client, request)
            return QuestionGenerationResult(
                success=True,
                questions=questions,
                mode="system",
                strategy="parallel",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="qwen-plus (规划) + qwen-max (生成)",
            )
        except Exception as e:
            return QuestionGenerationResult(
                success=False,
                error=str(e),
                mode="system",
                strategy="parallel",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="qwen-plus + qwen-max",
            )
    else:
        # 用户自配 API: 使用单次生成策略
        model_id = resolved.model_id or "gpt-4"
        try:
            questions = await generate_questions_single_request(client, request, model_id)
            return QuestionGenerationResult(
                success=True,
                questions=questions,
                mode="byok",
                strategy="single_request",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used=model_id,
            )
        except Exception as e:
            return QuestionGenerationResult(
                success=False,
                error=str(e),
                mode="byok",
                strategy="single_request",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used=model_id,
            )



async def generate_questions_batch_stream(
    db: Session,
    user_id: str,
    request: QuestionBatchRequest,
    use_system: bool | None = None,
) -> AsyncGenerator[QuestionProgressEvent | QuestionGenerationResult, None]:
    """
    流式批量生成题目（带进度反馈）
    """
    start_time = time.time()

    try:
        resolved = resolve_llm_config(
            db,
            user_id=user_id,
            requested_use_system=use_system,
            override_api_key=None,
            override_base_url=None,
            override_model_id=None,
            override_routing=None,
        )
    except Exception as e:
        yield QuestionGenerationResult(
            success=False,
            error=str(e),
            mode="byok",
            strategy="single_request",
            processing_time_ms=(time.time() - start_time) * 1000,
            model_used="",
        )
        return

    client = LLMClient(resolved.api_key, resolved.base_url)

    if resolved.mode == "system":
        # 官方 API: 使用流式并行策略
        try:
            questions = []
            async for event in generate_questions_parallel_stream(client, request):
                if isinstance(event, QuestionProgressEvent):
                    yield event
                elif isinstance(event, list):
                    questions = event

            yield QuestionGenerationResult(
                success=True,
                questions=questions,
                mode="system",
                strategy="parallel",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="qwen-plus (规划) + qwen-max (生成)",
            )
        except Exception as e:
            yield QuestionGenerationResult(
                success=False,
                error=str(e),
                mode="system",
                strategy="parallel",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="qwen-plus + qwen-max",
            )
    else:
        # 用户自配 API: 单次请求（无流式）
        model_id = resolved.model_id or "gpt-4"
        yield QuestionProgressEvent(
            stage="generating",
            progress=10,
            message="正在生成题目...",
            total_questions=request.count,
        )
        try:
            questions = await generate_questions_single_request(client, request, model_id)
            yield QuestionProgressEvent(
                stage="done",
                progress=100,
                message="题目生成完成",
                total_questions=len(questions),
                completed_questions=len(questions),
            )
            yield QuestionGenerationResult(
                success=True,
                questions=questions,
                mode="byok",
                strategy="single_request",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used=model_id,
            )
        except Exception as e:
            yield QuestionGenerationResult(
                success=False,
                error=str(e),
                mode="byok",
                strategy="single_request",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used=model_id,
            )