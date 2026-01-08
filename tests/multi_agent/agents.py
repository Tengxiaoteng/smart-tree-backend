"""
多 Agent 协同处理核心模块

架构设计：
1. IntentAgent (意图识别) - 使用 qwen-turbo (快速便宜)
   - 判断任务难度和类型
   - 推荐合适的模型

2. AnalysisAgent (图片/文档分析) - 使用 qwen-vl-plus (视觉模型)
   - 处理图片和 PDF
   - 提取结构化内容

3. TreeAgent (知识树生成) - 根据难度动态选择模型
   - simple: qwen-turbo
   - medium: qwen-plus
   - complex: qwen-max

4. RouterAgent (路由协调) - 协调各 Agent 工作
"""
import json
import time
import re
from typing import Any, Optional

import httpx

from .schemas import (
    IntentResult,
    ImageAnalysisResult,
    KnowledgeTree,
    KnowledgeNode,
    MultiAgentResult,
    TaskDifficulty,
    TaskType,
)


# DashScope 模型配置
MODEL_CONFIG = {
    "fast": "qwen-turbo",           # 快速模型，用于意图识别
    "standard": "qwen-plus",        # 标准模型，中等难度任务
    "advanced": "qwen-max",         # 高级模型，复杂任务
    "vision": "qwen-vl-plus",       # 视觉模型，图片分析
}

# 难度到模型的映射
DIFFICULTY_MODEL_MAP = {
    TaskDifficulty.SIMPLE: MODEL_CONFIG["fast"],
    TaskDifficulty.MEDIUM: MODEL_CONFIG["standard"],
    TaskDifficulty.COMPLEX: MODEL_CONFIG["advanced"],
}


class LLMClient:
    """统一的 LLM 调用客户端"""

    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
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
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
    ) -> str:
        """调用 LLM API"""
        timeout = httpx.Timeout(120.0, connect=10.0)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # DashScope 支持 JSON mode
        if response_format:
            payload["response_format"] = response_format

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


def _extract_json(content: str) -> dict | None:
    """从 LLM 响应中提取 JSON"""
    # 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试提取 {...}
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


class IntentAgent:
    """意图识别 Agent - 使用快速模型判断任务难度和类型"""

    SYSTEM_PROMPT = """你是一个智能任务分析助手。你的职责是分析用户的输入，判断任务类型和难度。

任务类型：
- text_analysis: 文本分析、理解、提取信息
- image_analysis: 图片内容识别和分析
- tree_generation: 生成知识树/思维导图结构
- qa: 问答任务
- summary: 摘要任务

难度等级：
- simple: 简单任务，直接回答即可，无需复杂推理
- medium: 中等任务，需要一定推理或结构化输出
- complex: 复杂任务，需要深度分析、多步推理、专业知识

模型推荐：
- simple 难度 → qwen-turbo（快速便宜）
- medium 难度 → qwen-plus（平衡性能）
- complex 难度 → qwen-max（最强能力）
- 图片任务 → qwen-vl-plus（视觉模型）

请严格按照以下 JSON 格式返回：
{
    "task_type": "类型",
    "difficulty": "难度",
    "recommended_model": "模型名",
    "reasoning": "判断理由"
}"""

    def __init__(self, client: LLMClient):
        self.client = client
        self.model = MODEL_CONFIG["fast"]  # 使用快速模型做意图识别

    async def analyze(self, user_input: str, has_image: bool = False) -> IntentResult:
        """分析用户意图"""
        context = f"用户输入: {user_input}"
        if has_image:
            context += "\n注意：用户上传了图片"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        start = time.time()
        result = await self.client.chat(self.model, messages, temperature=0.1)
        elapsed = (time.time() - start) * 1000

        print(f"[IntentAgent] 耗时: {elapsed:.0f}ms, 模型: {self.model}")

        parsed = _extract_json(result)
        if not parsed:
            # 默认返回
            return IntentResult(
                task_type=TaskType.TEXT_ANALYSIS,
                difficulty=TaskDifficulty.MEDIUM,
                recommended_model=MODEL_CONFIG["standard"],
                reasoning="JSON 解析失败，使用默认配置",
            )

        return IntentResult(
            task_type=parsed.get("task_type", "text_analysis"),
            difficulty=parsed.get("difficulty", "medium"),
            recommended_model=parsed.get("recommended_model", MODEL_CONFIG["standard"]),
            reasoning=parsed.get("reasoning", ""),
        )


class AnalysisAgent:
    """图片/文档分析 Agent - 使用视觉模型"""

    SYSTEM_PROMPT = """你是学习资料分析助手。请分析图片/文档内容，提取学习相关的关键信息。

要求：
1. 尽量逐字转写图片中的文字/公式
2. 数学公式必须使用 LaTeX，用 $...$ 或 $$...$$ 包裹
3. 用 Markdown 给出解释与学习要点

请严格按照以下 JSON 格式返回：
{
    "content": "Markdown 格式的完整内容",
    "summary": "一句话摘要",
    "concepts": ["关键概念1", "关键概念2"],
    "has_math": true/false,
    "has_code": true/false
}"""

    def __init__(self, client: LLMClient):
        self.client = client
        self.model = MODEL_CONFIG["vision"]

    async def analyze_image(self, image_base64: str, mime_type: str = "image/jpeg") -> ImageAnalysisResult:
        """分析图片内容"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                {"type": "text", "text": "请分析这张图片的内容，提取学习相关的关键信息。"},
            ]},
        ]

        start = time.time()
        result = await self.client.chat(self.model, messages)
        elapsed = (time.time() - start) * 1000

        print(f"[AnalysisAgent] 耗时: {elapsed:.0f}ms, 模型: {self.model}")

        parsed = _extract_json(result)
        if not parsed:
            return ImageAnalysisResult(
                content=result,
                summary="图片内容分析",
                concepts=[],
            )

        return ImageAnalysisResult(
            content=parsed.get("content", result),
            summary=parsed.get("summary", ""),
            concepts=parsed.get("concepts", []),
            has_math=parsed.get("has_math", False),
            has_code=parsed.get("has_code", False),
        )


class TreeAgent:
    """知识树生成 Agent - 根据难度动态选择模型"""

    SYSTEM_PROMPT = """你是知识结构化专家。请将输入内容整理成清晰的知识树结构。

要求：
1. 识别主题和核心概念
2. 构建层级分明的知识节点
3. 每个节点包含简洁的描述

请严格按照以下 JSON 格式返回：
{
    "topic": "主题名称",
    "summary": "主题摘要",
    "nodes": [
        {"id": "1", "name": "节点名", "description": "描述", "parent_id": null, "order": 0},
        {"id": "1.1", "name": "子节点", "description": "描述", "parent_id": "1", "order": 0}
    ],
    "concepts": ["核心概念1", "核心概念2"]
}"""

    def __init__(self, client: LLMClient):
        self.client = client

    async def generate(self, content: str, difficulty: TaskDifficulty) -> KnowledgeTree:
        """生成知识树"""
        model = DIFFICULTY_MODEL_MAP.get(difficulty, MODEL_CONFIG["standard"])

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"请将以下内容整理成知识树：\n\n{content}"},
        ]

        start = time.time()
        result = await self.client.chat(model, messages)
        elapsed = (time.time() - start) * 1000

        print(f"[TreeAgent] 耗时: {elapsed:.0f}ms, 模型: {model}, 难度: {difficulty}")

        parsed = _extract_json(result)
        if not parsed:
            return KnowledgeTree(
                topic="知识树",
                summary="",
                nodes=[KnowledgeNode(id="1", name="根节点", description=content[:100])],
            )

        nodes = []
        for n in parsed.get("nodes", []):
            nodes.append(KnowledgeNode(
                id=str(n.get("id", len(nodes) + 1)),
                name=n.get("name", ""),
                description=n.get("description"),
                parent_id=n.get("parent_id"),
                order=n.get("order", 0),
            ))

        return KnowledgeTree(
            topic=parsed.get("topic", "知识树"),
            summary=parsed.get("summary", ""),
            nodes=nodes,
            concepts=parsed.get("concepts", []),
        )


class MultiAgentOrchestrator:
    """多 Agent 协调器 - 协调各 Agent 协同工作"""

    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = LLMClient(api_key, base_url)
        self.intent_agent = IntentAgent(self.client)
        self.analysis_agent = AnalysisAgent(self.client)
        self.tree_agent = TreeAgent(self.client)

    async def process(
        self,
        user_input: str,
        image_base64: Optional[str] = None,
        mime_type: str = "image/jpeg",
    ) -> MultiAgentResult:
        """协调多 Agent 处理请求"""
        total_start = time.time()

        # Step 1: 意图识别
        has_image = image_base64 is not None
        intent = await self.intent_agent.analyze(user_input, has_image)
        print(f"[Orchestrator] 意图识别完成: {intent.task_type}, 难度: {intent.difficulty}")

        result: dict[str, Any] = {}
        model_used = intent.recommended_model

        # Step 2: 根据意图路由到对应 Agent
        if intent.task_type == TaskType.IMAGE_ANALYSIS and image_base64:
            # 图片分析
            analysis = await self.analysis_agent.analyze_image(image_base64, mime_type)
            result = analysis.model_dump()
            model_used = MODEL_CONFIG["vision"]

        elif intent.task_type == TaskType.TREE_GENERATION:
            # 知识树生成
            tree = await self.tree_agent.generate(user_input, TaskDifficulty(intent.difficulty))
            result = tree.model_dump()
            model_used = DIFFICULTY_MODEL_MAP.get(TaskDifficulty(intent.difficulty), MODEL_CONFIG["standard"])

        else:
            # 其他任务：直接用推荐模型处理
            model = intent.recommended_model
            messages = [
                {"role": "system", "content": "你是一个智能学习助手，请帮助用户解答问题。"},
                {"role": "user", "content": user_input},
            ]
            response = await self.client.chat(model, messages)
            result = {"content": response}
            model_used = model

        total_time = (time.time() - total_start) * 1000

        return MultiAgentResult(
            intent=intent,
            processing_time_ms=total_time,
            model_used=model_used,
            result=result,
        )
