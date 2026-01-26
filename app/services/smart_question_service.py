"""
智能出题服务 V2 - 基于 LangChain + Pydantic 结构化输出

核心特性：
1. 使用 LangChain with_structured_output 确保输出格式
2. 集成用户画像，过滤已掌握知识点，针对薄弱点出题
3. 强制关联学习目标
4. 并发生成题目（asyncio.gather）
5. 支持多种出题来源：知识点内容、资料、学习目标
"""
import asyncio
import uuid
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ==================== Pydantic Schema ====================

class QuestionTrace(BaseModel):
    """题目来源追溯"""
    source_type: Literal["node_content", "material", "learning_goal", "weak_point"] = Field(
        description="来源类型"
    )
    source_name: str = Field(description="来源名称")
    source_context: str = Field(description="出题依据的具体内容")
    confidence: float = Field(default=0.8, ge=0, le=1, description="置信度")


class QuestionTags(BaseModel):
    """可检索标签"""
    concepts: list[str] = Field(default_factory=list, description="涉及的核心概念")
    skills: list[str] = Field(default_factory=list, description="考察的能力")
    bloom_level: Literal["remember", "understand", "apply", "analyze", "evaluate", "create"] = Field(
        default="understand", description="布鲁姆认知层次"
    )
    keywords: list[str] = Field(default_factory=list, description="关键词")


class GeneratedQuestion(BaseModel):
    """生成的题目"""
    type: Literal["single", "judge"] = Field(description="题型：single=单选, judge=判断")
    difficulty: Literal["easy", "medium", "hard"] = Field(description="难度")
    content: str = Field(description="题目内容，支持 LaTeX 公式 $...$")
    options: list[str] = Field(description="选项列表，格式如 ['A. xxx', 'B. xxx']")
    answer: str = Field(description="正确答案，如 'A' 或 'B'")
    explanation: str = Field(description="答案解析")
    target_goal_id: Optional[str] = Field(None, description="关联的学习目标 ID")
    target_goal_name: Optional[str] = Field(None, description="关联的学习目标名称")
    target_weakness: Optional[str] = Field(None, description="针对的薄弱点")
    trace: QuestionTrace = Field(description="来源追溯")
    tags: QuestionTags = Field(default_factory=QuestionTags, description="可检索标签")


class QuestionBatch(BaseModel):
    """批量生成结果"""
    questions: list[GeneratedQuestion] = Field(description="生成的题目列表")


# ==================== 用户画像 Schema ====================

class ConceptMastery(BaseModel):
    """概念掌握度"""
    concept: str
    mastery_rate: float = Field(ge=0, le=100, description="掌握率 0-100%")
    is_weak: bool = Field(description="是否为薄弱点")
    total_attempts: int = Field(default=0, description="尝试次数")


class LearnerProfile(BaseModel):
    """学习者画像"""
    total_questions: int = Field(default=0, description="已做题目总数")
    correct_rate: float = Field(default=0, ge=0, le=100, description="总体正确率")
    weak_concepts: list[ConceptMastery] = Field(default_factory=list, description="薄弱概念")
    mastered_concepts: list[str] = Field(default_factory=list, description="已掌握概念（正确率>80%）")


class LearningGoal(BaseModel):
    """学习目标"""
    id: str
    goal: str
    importance: Literal["must", "should", "could"] = "should"
    related_concepts: list[str] = Field(default_factory=list)
    mastery_score: float = Field(default=0, ge=0, le=100)


class MaterialInfo(BaseModel):
    """资料信息"""
    id: str
    name: str
    content_digest: str = ""
    key_topics: list[str] = Field(default_factory=list)


# ==================== 出题请求 Schema ====================

class SmartQuestionRequest(BaseModel):
    """智能出题请求"""
    # 节点信息
    node_id: str
    node_name: str
    node_description: str = ""
    key_concepts: list[str] = Field(default_factory=list)

    # 出题配置
    count: int = Field(default=5, ge=1, le=20)
    mode: Literal["normal", "review", "advance"] = "normal"
    question_types: list[str] = Field(default_factory=lambda: ["single", "judge"])

    # 难度分布（百分比）
    difficulty_easy: int = Field(default=30, ge=0, le=100)
    difficulty_medium: int = Field(default=50, ge=0, le=100)
    difficulty_hard: int = Field(default=20, ge=0, le=100)

    # 出题来源
    use_node_content: bool = True
    use_materials: bool = True
    use_learning_goals: bool = True

    # 资料列表
    materials: list[MaterialInfo] = Field(default_factory=list)

    # 学习目标列表
    learning_goals: list[LearningGoal] = Field(default_factory=list)

    # 用户画像
    learner_profile: Optional[LearnerProfile] = None


# ==================== 智能出题服务 ====================

class SmartQuestionService:
    """
    智能出题服务

    特性：
    1. 根据用户画像过滤已掌握知识点
    2. 针对薄弱点优先出题
    3. 强制关联学习目标
    4. 并发生成提高效率
    """

    SYSTEM_PROMPT = """你是专业的教育测评专家，擅长根据学生的学习情况进行个性化出题。

## 核心原则
1. **精准出题**：每道题必须有明确的知识点依据，不可凭空编造
2. **个性化**：根据用户画像，针对薄弱点出题，避免重复考察已掌握内容
3. **目标导向**：每道题都要关联学习目标
4. **难度适配**：根据用户表现调整难度

## 题型规范
- single: 单选题，4个选项，只有1个正确答案
- judge: 判断题，选项为 ["A. 正确", "B. 错误"]

## LaTeX 公式
- 行内公式用 $...$，如 $E=mc^2$
- 独立公式用 $$...$$

## 重要：输出格式
你必须直接输出一个 JSON 对象（不要用 markdown 代码块包裹），包含以下字段：
{
  "type": "single 或 judge",
  "difficulty": "easy/medium/hard",
  "content": "题目内容（具体、有意义的问题）",
  "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
  "answer": "正确答案字母如 A",
  "explanation": "详细的答案解析",
  "trace": {"source_type": "node_content", "source_name": "来源", "source_context": "依据", "confidence": 0.8},
  "tags": {"concepts": ["概念"], "skills": [], "bloom_level": "understand", "keywords": []}
}

注意：content 必须是具体的、有教育意义的问题，不能是通用模板如"以下哪个说法最准确"。"""

    def __init__(self, api_key: str, base_url: str, model_id: str = "qwen-max"):
        """
        初始化服务

        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            model_id: 模型 ID
        """
        # 确保 base_url 格式正确
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        if not base_url.endswith("/v1"):
            if "/chat/completions" in base_url:
                base_url = base_url.replace("/chat/completions", "")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1" if "dashscope" in base_url else base_url

        self.llm = ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=0.7,
            max_tokens=2000,
        )
        self.model_id = model_id

    def _build_context(self, request: SmartQuestionRequest) -> str:
        """构建出题上下文"""
        parts = []

        # 1. 节点知识内容
        if request.use_node_content:
            parts.append(f"""## 知识点：{request.node_name}
{request.node_description or '无描述'}
关键概念：{', '.join(request.key_concepts) or '无'}""")

        # 2. 资料内容
        if request.use_materials and request.materials:
            for mat in request.materials[:3]:  # 最多3份资料
                parts.append(f"""## 资料：{mat.name}
{mat.content_digest[:500] if mat.content_digest else '无内容摘要'}
关键主题：{', '.join(mat.key_topics[:5]) if mat.key_topics else '无'}""")

        # 3. 学习目标
        if request.use_learning_goals and request.learning_goals:
            goals_text = "\n".join([
                f"- [{g.importance}] {g.goal}（掌握度: {g.mastery_score:.0f}%）"
                for g in request.learning_goals
            ])
            parts.append(f"""## 用户学习目标（请每道题关联一个目标）
{goals_text}""")

        return "\n\n---\n\n".join(parts)

    def _build_learner_context(self, profile: Optional[LearnerProfile]) -> str:
        """构建用户画像上下文"""
        if not profile or profile.total_questions == 0:
            return "用户首次练习，暂无历史数据。建议从基础题开始。"

        parts = [
            f"## 用户学习画像",
            f"- 已做题目: {profile.total_questions} 道",
            f"- 总体正确率: {profile.correct_rate:.0f}%",
        ]

        # 薄弱点（重点）
        if profile.weak_concepts:
            parts.append("\n### ⚠️ 薄弱概念（请重点出题，约占70%）")
            for weak in profile.weak_concepts[:5]:
                parts.append(f"- 「{weak.concept}」正确率 {weak.mastery_rate:.0f}%")

        # 已掌握（减少或跳过）
        if profile.mastered_concepts:
            parts.append("\n### ✅ 已掌握概念（可少出或提高难度）")
            parts.append(f"- {', '.join(profile.mastered_concepts[:5])}")

        parts.append("\n**重要：优先针对薄弱概念出题，已掌握的概念可以跳过或提高难度。**")

        return "\n".join(parts)

    def _filter_mastered_goals(self, goals: list[LearningGoal]) -> list[LearningGoal]:
        """过滤已掌握的学习目标"""
        # 保留掌握度低于 90% 的目标
        return [g for g in goals if g.mastery_score < 90]

    async def generate_single_question(
        self,
        index: int,
        context: str,
        learner_context: str,
        request: SmartQuestionRequest,
        plan: dict,
    ) -> GeneratedQuestion:
        """
        生成单道题目

        Args:
            index: 题目索引
            context: 知识上下文
            learner_context: 用户画像上下文
            request: 出题请求
            plan: 出题计划（包含题型、难度、目标等）
        """
        # 构建 prompt
        target_goal = plan.get("target_goal")
        target_weakness = plan.get("target_weakness")

        goal_hint = ""
        if target_goal:
            goal_hint = f"\n关联学习目标：{target_goal['goal']}（ID: {target_goal['id']}）"

        weakness_hint = ""
        if target_weakness:
            weakness_hint = f"\n针对薄弱点：{target_weakness}"

        user_prompt = f"""{context}

{learner_context}

---

## 出题要求（第 {index + 1} 题）
- 题型：{plan['type']}
- 难度：{plan['difficulty']}
- 模式：{request.mode}{goal_hint}{weakness_hint}

请生成一道符合要求的题目。必须包含完整的 trace 和 tags 信息。"""

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            # 直接调用 LLM（兼容不支持 structured_output 的 API）
            response = await self.llm.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # 手动解析 JSON
            parsed = self._extract_json(response_text)
            
            if parsed and isinstance(parsed, dict):
                result = GeneratedQuestion(
                    type=parsed.get('type', plan['type']),
                    difficulty=parsed.get('difficulty', plan['difficulty']),
                    content=parsed.get('content', ''),
                    options=parsed.get('options', []),
                    answer=parsed.get('answer', 'A'),
                    explanation=parsed.get('explanation', ''),
                    target_goal_id=target_goal['id'] if target_goal else parsed.get('target_goal_id'),
                    target_goal_name=target_goal['goal'] if target_goal else parsed.get('target_goal_name'),
                    target_weakness=target_weakness or parsed.get('target_weakness'),
                    trace=QuestionTrace(
                        source_type=parsed.get('trace', {}).get('source_type', 'node_content'),
                        source_name=parsed.get('trace', {}).get('source_name', request.node_name),
                        source_context=parsed.get('trace', {}).get('source_context', ''),
                        confidence=parsed.get('trace', {}).get('confidence', 0.8),
                    ),
                    tags=QuestionTags(
                        concepts=parsed.get('tags', {}).get('concepts', request.key_concepts[:2]),
                        skills=parsed.get('tags', {}).get('skills', []),
                        bloom_level=parsed.get('tags', {}).get('bloom_level', 'understand'),
                        keywords=parsed.get('tags', {}).get('keywords', [request.node_name]),
                    ),
                )
                print(f"[SmartQuestionService] 题目 {index + 1} 生成成功，关联目标ID: {result.target_goal_id}")
                return result
            else:
                print(f"[SmartQuestionService] 题目 {index + 1} JSON 解析失败，返回内容: {response_text[:200]}")
                return self._default_question(plan, request)

        except Exception as e:
            print(f"[SmartQuestionService] 生成题目 {index + 1} 失败: {e}")
            # 返回默认题目
            return self._default_question(plan, request)


    def _extract_json(self, text: str):
        """从 AI 响应中提取 JSON"""
        import json as json_module
        import re as re_module
        
        if not text:
            return None
        
        # 尝试直接解析
        try:
            return json_module.loads(text)
        except:
            pass
        
        # 尝试提取 ```json ... ``` 块
        json_match = re_module.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            try:
                return json_module.loads(json_match.group(1).strip())
            except:
                pass
        
        # 尝试提取 {...} 块
        brace_match = re_module.search(r'\{[\s\S]*\}', text)
        if brace_match:
            try:
                return json_module.loads(brace_match.group())
            except:
                pass
        
        return None

    def _default_question(self, plan: dict, request: SmartQuestionRequest) -> GeneratedQuestion:
        """生成默认题目"""
        if plan["type"] == "judge":
            return GeneratedQuestion(
                type="judge",
                difficulty=plan["difficulty"],
                content=f"判断：关于{request.node_name}的说法是正确的。",
                options=["A. 正确", "B. 错误"],
                answer="A",
                explanation=f"这是一道基础理解题。{request.node_name}是本知识点的核心内容，需要全面理解其概念和应用场景。如果题目质量不理想，建议点击刷新重新生成。",
                target_goal_id=plan.get("target_goal", {}).get("id"),
                target_goal_name=plan.get("target_goal", {}).get("goal"),
                target_weakness=plan.get("target_weakness"),
                trace=QuestionTrace(
                    source_type="node_content",
                    source_name=request.node_name,
                    source_context="默认生成",
                    confidence=0.5,
                ),
                tags=QuestionTags(
                    concepts=request.key_concepts[:2],
                    keywords=[request.node_name],
                ),
            )
        else:
            return GeneratedQuestion(
                type="single",
                difficulty=plan["difficulty"],
                content=f"关于「{request.node_name}」的学习，以下哪种方法最有效？",
                options=[
                    f"A. 结合实际案例理解{request.node_name}的应用场景",
                    f"B. 只需要记住{request.node_name}的定义即可",
                    f"C. {request.node_name}不重要，可以跳过",
                    f"D. 孤立学习{request.node_name}，不需要联系其他知识"
                ],
                answer="A",
                explanation=f"学习{request.node_name}最有效的方法是结合实际案例理解其应用场景，这样可以加深理解并提高实践能力。建议点击重新生成以获取更具针对性的练习题。",
                target_goal_id=plan.get("target_goal", {}).get("id"),
                target_goal_name=plan.get("target_goal", {}).get("goal"),
                target_weakness=plan.get("target_weakness"),
                trace=QuestionTrace(
                    source_type="node_content",
                    source_name=request.node_name,
                    source_context="默认生成",
                    confidence=0.5,
                ),
                tags=QuestionTags(
                    concepts=request.key_concepts[:2],
                    keywords=[request.node_name],
                ),
            )

    def _create_question_plans(self, request: SmartQuestionRequest) -> list[dict]:
        """
        创建出题计划

        根据用户画像和学习目标分配题目
        """
        plans = []
        count = request.count

        # 计算难度分布
        easy_count = round(count * request.difficulty_easy / 100)
        medium_count = round(count * request.difficulty_medium / 100)
        hard_count = count - easy_count - medium_count

        # 准备可用目标（过滤已掌握）
        available_goals = self._filter_mastered_goals(request.learning_goals) if request.learning_goals else []
        print(f"[SmartQuestionService] 原始目标数: {len(request.learning_goals) if request.learning_goals else 0}, 过滤后: {len(available_goals)}"); print(f"[SmartQuestionService] 各目标mastery_score: {[(g.goal[:15], g.mastery_score) for g in (request.learning_goals or [])]}")
        if available_goals:
            print(f"[SmartQuestionService] 可用目标: {[g.goal[:20] for g in available_goals]}")

        # 准备薄弱点
        weak_concepts = []
        if request.learner_profile and request.learner_profile.weak_concepts:
            weak_concepts = [w.concept for w in request.learner_profile.weak_concepts]

        # 分配题目
        for i in range(count):
            # 确定难度
            if i < easy_count:
                difficulty = "easy"
            elif i < easy_count + medium_count:
                difficulty = "medium"
            else:
                difficulty = "hard"

            # 确定题型（交替）
            q_type = "single" if i % 2 == 0 else "judge"
            if "single" not in request.question_types:
                q_type = "judge"
            if "judge" not in request.question_types:
                q_type = "single"

            # 分配学习目标
            target_goal = None
            if available_goals:
                target_goal = {
                    "id": available_goals[i % len(available_goals)].id,
                    "goal": available_goals[i % len(available_goals)].goal,
                }

            # 分配薄弱点（优先针对薄弱点）
            target_weakness = None
            if weak_concepts and i < len(weak_concepts):
                # 前 70% 的题目针对薄弱点
                if i < count * 0.7:
                    target_weakness = weak_concepts[i % len(weak_concepts)]

            plans.append({
                "index": i,
                "type": q_type,
                "difficulty": difficulty,
                "target_goal": target_goal,
                "target_weakness": target_weakness,
            })

        for p in plans: print(f"[Plan] 题目{p['index']+1}: 目标={p.get('target_goal', {}).get('goal', '无')[:20]}")
        return plans

    async def generate_questions(self, request: SmartQuestionRequest) -> list[GeneratedQuestion]:
        """
        并发生成题目

        Args:
            request: 出题请求

        Returns:
            生成的题目列表
        """
        # 构建上下文
        context = self._build_context(request)
        learner_context = self._build_learner_context(request.learner_profile)

        # 创建出题计划
        plans = self._create_question_plans(request)

        # 并发生成（限制并发数）
        semaphore = asyncio.Semaphore(5)  # 最多5个并发

        async def generate_with_semaphore(plan: dict) -> tuple[int, GeneratedQuestion]:
            async with semaphore:
                question = await self.generate_single_question(
                    plan["index"], context, learner_context, request, plan
                )
                return plan["index"], question

        # 并发执行
        tasks = [generate_with_semaphore(plan) for plan in plans]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 组装结果（保持顺序）
        questions_map = {}
        for result in results:
            if isinstance(result, tuple):
                idx, question = result
                questions_map[idx] = question
            elif isinstance(result, Exception):
                print(f"[SmartQuestionService] 生成异常: {result}")

        return [questions_map[i] for i in sorted(questions_map.keys()) if i in questions_map]


# ==================== 便捷函数 ====================

async def generate_smart_questions(
    api_key: str,
    base_url: str,
    model_id: str,
    request: SmartQuestionRequest,
) -> list[GeneratedQuestion]:
    """
    生成智能题目的便捷函数

    Args:
        api_key: API 密钥
        base_url: API 基础 URL
        model_id: 模型 ID
        request: 出题请求

    Returns:
        生成的题目列表
    """
    service = SmartQuestionService(api_key, base_url, model_id)
    return await service.generate_questions(request)
