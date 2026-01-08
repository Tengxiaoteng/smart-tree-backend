"""
多 Agent 架构 v2 - 使用 LangChain + 并行处理

改进：
1. 知识树生成三层，内容更丰富
2. 意图识别与内容预处理并行
3. 使用 LangChain 结构化输出
"""
import asyncio
import json
import time
import re
from typing import Any, Optional
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, Field

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
    "fast": "qwen-turbo",
    "standard": "qwen-plus",
    "advanced": "qwen-max",
    "vision": "qwen-vl-plus",
}

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
        max_tokens: int = 4000,
    ) -> str:
        timeout = httpx.Timeout(180.0, connect=10.0)

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


def _extract_json(content: str) -> dict | None:
    """从 LLM 响应中提取 JSON"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


class IntentAgentV2:
    """意图识别 Agent v2 - 更快速的意图判断"""

    SYSTEM_PROMPT = """你是任务分析专家。快速判断任务类型和难度。

任务类型判断规则：
- tree_generation: 包含"知识树"、"整理"、"结构化"、"体系"等关键词
- text_analysis: 分析、理解、提取信息
- qa: 问答任务
- summary: 摘要、总结

难度判断：
- simple: 直接回答，无需复杂推理
- medium: 需要推理或结构化
- complex: 深度分析，多步推理

模型映射: simple→qwen-turbo, medium→qwen-plus, complex→qwen-max

返回 JSON: {"task_type": "...", "difficulty": "...", "recommended_model": "...", "reasoning": "..."}"""

    def __init__(self, client: LLMClient):
        self.client = client
        self.model = MODEL_CONFIG["fast"]

    async def analyze(self, user_input: str, has_image: bool = False) -> IntentResult:
        # 快速关键词检测
        input_lower = user_input.lower()
        if any(kw in input_lower for kw in ["知识树", "整理", "结构", "体系", "树形", "层级"]):
            print(f"[IntentAgent v2] 关键词检测: tree_generation")
            return IntentResult(
                task_type=TaskType.TREE_GENERATION,
                difficulty=TaskDifficulty.MEDIUM,
                recommended_model=MODEL_CONFIG["standard"],
                reasoning="包含知识树/整理相关关键词",
            )

        context = f"输入: {user_input[:200]}"
        if has_image:
            context += "\n[有图片]"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        start = time.time()
        result = await self.client.chat(self.model, messages, temperature=0.1, max_tokens=500)
        elapsed = (time.time() - start) * 1000

        print(f"[IntentAgent v2] 耗时: {elapsed:.0f}ms")

        parsed = _extract_json(result)
        if not parsed:
            return IntentResult(
                task_type=TaskType.TEXT_ANALYSIS,
                difficulty=TaskDifficulty.MEDIUM,
                recommended_model=MODEL_CONFIG["standard"],
                reasoning="默认配置",
            )

        return IntentResult(
            task_type=parsed.get("task_type", "text_analysis"),
            difficulty=parsed.get("difficulty", "medium"),
            recommended_model=parsed.get("recommended_model", MODEL_CONFIG["standard"]),
            reasoning=parsed.get("reasoning", ""),
        )


class TreeAgentV2:
    """知识树生成 Agent v2 - 生成三层丰富内容"""

    SYSTEM_PROMPT = """你是知识结构化专家。将内容整理成三层知识树。

要求：
- 第一层: 2-4个核心主题
- 第二层: 每个主题下3-5个关键概念
- 第三层: 每个概念下2-3个具体知识点
- 每个节点有30-80字描述

只返回 JSON，不要其他文字：
{"topic":"主题名","summary":"概述","nodes":[{"id":"1","name":"名称","description":"描述","parent_id":null,"order":0},{"id":"1.1","name":"名称","description":"描述","parent_id":"1","order":0}],"concepts":["概念1","概念2"]}"""

    def __init__(self, client: LLMClient):
        self.client = client

    async def generate(self, content: str, difficulty: TaskDifficulty) -> KnowledgeTree:
        model = DIFFICULTY_MODEL_MAP.get(difficulty, MODEL_CONFIG["standard"])

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"整理成三层知识树：\n{content}"},
        ]

        start = time.time()
        result = await self.client.chat(model, messages, max_tokens=4000)
        elapsed = (time.time() - start) * 1000

        print(f"[TreeAgent v2] 耗时: {elapsed:.0f}ms, 模型: {model}")
        print(f"[TreeAgent v2] 响应长度: {len(result)}, 前200字: {result[:200]}...")

        parsed = _extract_json(result)
        if not parsed:
            print(f"[TreeAgent v2] JSON 解析失败!")
            return KnowledgeTree(
                topic="知识树",
                summary="",
                nodes=[KnowledgeNode(id="1", name="根节点", description=content[:100])],
            )

        print(f"[TreeAgent v2] 解析成功, 节点数: {len(parsed.get('nodes', []))}")

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


class ParallelOrchestrator:
    """并行处理协调器 - 意图识别与预处理并行"""

    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = LLMClient(api_key, base_url)
        self.intent_agent = IntentAgentV2(self.client)
        self.tree_agent = TreeAgentV2(self.client)

    async def _quick_preprocess(self, user_input: str) -> dict:
        """快速预处理 - 与意图识别并行"""
        # 简单的关键词提取和长度分析
        word_count = len(user_input)
        has_formula = bool(re.search(r'[\$\=\+\-\*\/\^]', user_input))
        has_code = bool(re.search(r'```|def |class |function|import ', user_input))

        return {
            "word_count": word_count,
            "has_formula": has_formula,
            "has_code": has_code,
        }

    async def process(
        self,
        user_input: str,
        image_base64: Optional[str] = None,
        mime_type: str = "image/jpeg",
    ) -> MultiAgentResult:
        """并行处理：意图识别 + 预处理同时进行"""
        total_start = time.time()
        has_image = image_base64 is not None

        # 并行执行：意图识别 + 预处理
        print("[Parallel] 开始并行处理：意图识别 + 预处理")
        intent_task = asyncio.create_task(self.intent_agent.analyze(user_input, has_image))
        preprocess_task = asyncio.create_task(self._quick_preprocess(user_input))

        intent, preprocess_info = await asyncio.gather(intent_task, preprocess_task)

        parallel_time = (time.time() - total_start) * 1000
        print(f"[Parallel] 并行阶段完成: {parallel_time:.0f}ms")
        print(f"[Parallel] 意图: {intent.task_type}, 难度: {intent.difficulty}")
        print(f"[Parallel] 预处理: {preprocess_info}")

        result: dict[str, Any] = {}
        model_used = intent.recommended_model

        # 根据意图路由处理
        if intent.task_type == TaskType.TREE_GENERATION:
            tree = await self.tree_agent.generate(user_input, TaskDifficulty(intent.difficulty))
            result = tree.model_dump()
            model_used = DIFFICULTY_MODEL_MAP.get(TaskDifficulty(intent.difficulty), MODEL_CONFIG["standard"])
        else:
            model = intent.recommended_model
            messages = [
                {"role": "system", "content": "你是智能学习助手。"},
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


# ============== LangChain 版本 ==============

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class LangChainOrchestrator:
    """使用 LangChain 的多 Agent 协调器"""

    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed. Run: pip install langchain langchain-openai")

        self.api_key = api_key
        self.base_url = base_url

        # 创建不同能力的模型
        self.fast_model = ChatOpenAI(
            model="qwen-turbo",
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
            max_tokens=500,
        )

        self.standard_model = ChatOpenAI(
            model="qwen-plus",
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
            max_tokens=4000,
        )

        self.advanced_model = ChatOpenAI(
            model="qwen-max",
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
            max_tokens=4000,
        )

        self.json_parser = JsonOutputParser()

    async def _analyze_intent(self, user_input: str) -> IntentResult:
        """使用 LangChain 进行意图分析"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """任务分析专家。判断任务类型和难度。
类型: text_analysis | tree_generation | qa | summary
难度: simple | medium | complex
返回 JSON: {{"task_type": "...", "difficulty": "...", "recommended_model": "qwen-turbo/qwen-plus/qwen-max", "reasoning": "..."}}"""),
            ("human", "输入: {input}")
        ])

        chain = prompt | self.fast_model | self.json_parser

        start = time.time()
        try:
            result = await chain.ainvoke({"input": user_input[:200]})
            elapsed = (time.time() - start) * 1000
            print(f"[LangChain Intent] 耗时: {elapsed:.0f}ms")

            return IntentResult(
                task_type=result.get("task_type", "text_analysis"),
                difficulty=result.get("difficulty", "medium"),
                recommended_model=result.get("recommended_model", "qwen-plus"),
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            print(f"[LangChain Intent] 解析失败: {e}")
            return IntentResult(
                task_type=TaskType.TEXT_ANALYSIS,
                difficulty=TaskDifficulty.MEDIUM,
                recommended_model="qwen-plus",
                reasoning="默认",
            )

    async def _generate_tree(self, content: str, difficulty: TaskDifficulty) -> KnowledgeTree:
        """使用 LangChain 生成知识树"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """知识结构化专家。生成三层知识树。
第一层: 核心主题（1-3个）
第二层: 关键概念（每主题2-5个）
第三层: 具体知识点（每概念2-4个）
每节点描述30-100字。

返回 JSON:
{{
    "topic": "总主题",
    "summary": "概述",
    "nodes": [
        {{"id": "1", "name": "名称", "description": "描述", "parent_id": null, "order": 0}},
        {{"id": "1.1", "name": "名称", "description": "描述", "parent_id": "1", "order": 0}}
    ],
    "concepts": ["概念1", "概念2"]
}}"""),
            ("human", "整理成三层知识树:\n\n{content}")
        ])

        # 根据难度选择模型
        model = {
            TaskDifficulty.SIMPLE: self.fast_model,
            TaskDifficulty.MEDIUM: self.standard_model,
            TaskDifficulty.COMPLEX: self.advanced_model,
        }.get(difficulty, self.standard_model)

        chain = prompt | model | self.json_parser

        start = time.time()
        try:
            result = await chain.ainvoke({"content": content})
            elapsed = (time.time() - start) * 1000
            print(f"[LangChain Tree] 耗时: {elapsed:.0f}ms, 难度: {difficulty}")

            nodes = [
                KnowledgeNode(
                    id=str(n.get("id")),
                    name=n.get("name", ""),
                    description=n.get("description"),
                    parent_id=n.get("parent_id"),
                    order=n.get("order", 0),
                )
                for n in result.get("nodes", [])
            ]

            return KnowledgeTree(
                topic=result.get("topic", "知识树"),
                summary=result.get("summary", ""),
                nodes=nodes,
                concepts=result.get("concepts", []),
            )
        except Exception as e:
            print(f"[LangChain Tree] 生成失败: {e}")
            return KnowledgeTree(
                topic="知识树",
                summary="",
                nodes=[KnowledgeNode(id="1", name="根节点", description=content[:100])],
            )

    async def process(self, user_input: str) -> MultiAgentResult:
        """LangChain 并行处理"""
        total_start = time.time()

        # 意图识别
        intent = await self._analyze_intent(user_input)
        print(f"[LangChain] 意图: {intent.task_type}, 难度: {intent.difficulty}")

        result: dict[str, Any] = {}
        model_used = intent.recommended_model

        if intent.task_type == TaskType.TREE_GENERATION:
            tree = await self._generate_tree(user_input, TaskDifficulty(intent.difficulty))
            result = tree.model_dump()
        else:
            # 其他任务使用推荐模型
            model = {
                "qwen-turbo": self.fast_model,
                "qwen-plus": self.standard_model,
                "qwen-max": self.advanced_model,
            }.get(intent.recommended_model, self.standard_model)

            response = await model.ainvoke([
                SystemMessage(content="你是智能学习助手。"),
                HumanMessage(content=user_input),
            ])
            result = {"content": response.content}

        total_time = (time.time() - total_start) * 1000

        return MultiAgentResult(
            intent=intent,
            processing_time_ms=total_time,
            model_used=model_used,
            result=result,
        )
