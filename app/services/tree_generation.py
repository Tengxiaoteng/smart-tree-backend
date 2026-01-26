"""
知识树生成服务

策略：
- 官方 API (system mode): 使用 V3 两阶段并行架构
- 用户自配 API (byok mode): 使用单次生成方式
"""
import asyncio
import json
import re
import time
import uuid
from typing import Any, AsyncGenerator, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.llm_context import resolve_llm_config, ResolvedLLMConfig
from app.services import credits as credits_service
from app.services.credits_pricing import FeatureType, calculate_fixed_points


# ==================== Pydantic Schema ====================

class KnowledgeNodeSchema(BaseModel):
    """知识节点"""
    id: str
    name: str
    description: str | None = None
    parent_id: str | None = Field(None, alias="parentId")
    order: int = 0
    # 学习属性
    learning_objectives: list[str] = Field(default_factory=list, alias="learningObjectives")
    key_concepts: list[str] = Field(default_factory=list, alias="keyConcepts")
    knowledge_type: str | None = Field(None, alias="knowledgeType")  # concept/principle/procedure/application
    difficulty: str | None = None  # beginner/intermediate/advanced
    estimated_minutes: int | None = Field(None, alias="estimatedMinutes")
    # 出题相关
    question_patterns: list[str] = Field(default_factory=list, alias="questionPatterns")
    common_mistakes: list[str] = Field(default_factory=list, alias="commonMistakes")

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class KnowledgeTreeSchema(BaseModel):
    """知识树结构"""
    topic: str
    summary: str = ""
    nodes: list[KnowledgeNodeSchema] = []
    concepts: list[str] = []


class TreeGenerationResult(BaseModel):
    """知识树生成结果"""
    success: bool
    tree: KnowledgeTreeSchema | None = None
    error: str | None = None
    mode: Literal["system", "byok"]
    strategy: Literal["v3_parallel", "single_request"]
    processing_time_ms: float
    model_used: str


class ProgressEvent(BaseModel):
    """进度事件"""
    stage: str  # planning / filling / done
    progress: float  # 0-100
    message: str
    current_node: str | None = None
    total_nodes: int = 0
    completed_nodes: int = 0


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


# ==================== V3 两阶段并行策略（官方 API 专用）====================

class PlannerAgent:
    """规划 Agent - 生成知识树骨架"""

    SYSTEM_PROMPT = """你是知识结构规划专家。生成三层知识树骨架。

要求：
- 第一层: 2-4个核心主题
- 第二层: 每个主题下2-4个关键概念
- 第三层: 每个概念下2-3个知识点
- 只需要节点名称

返回 JSON:
{"topic":"主题","nodes":[{"id":"1","name":"一级","parent_id":null},{"id":"1.1","name":"二级","parent_id":"1"},{"id":"1.1.1","name":"三级","parent_id":"1.1"}]}"""

    def __init__(self, client: LLMClient):
        self.client = client
        self.model = "deepseek-chat"

    async def plan(self, content: str) -> dict:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"为以下内容规划知识树：\n{content}"},
        ]

        result = await self.client.chat(self.model, messages, max_tokens=1500)
        parsed = _extract_json(result)

        if not parsed:
            return {"topic": "知识树", "nodes": [{"id": "1", "name": "根节点", "parent_id": None}]}

        return parsed


class ContentAgent:
    """内容生成 Agent - 并发填充节点内容（包含学习属性）"""

    SYSTEM_PROMPT = """你是知识内容专家。为知识点生成完整的学习信息。

要求：
- description: 50-100字描述
- learningObjectives: 2-3个学习目标，以"能够"开头
- keyConcepts: 2-4个关键概念/术语
- knowledgeType: 知识类型（concept/principle/procedure/application）
- difficulty: 难度（beginner/intermediate/advanced）
- estimatedMinutes: 预计学习时间（分钟）
- questionPatterns: 1-2个出题方向
- commonMistakes: 1-2个常见错误

返回 JSON:
{"description":"描述","learningObjectives":["能够..."],"keyConcepts":["概念1","概念2"],"knowledgeType":"concept","difficulty":"beginner","estimatedMinutes":10,"questionPatterns":["题型"],"commonMistakes":["错误"]}"""

    def __init__(self, client: LLMClient, max_concurrent: int = 10):
        self.client = client
        self.model = "deepseek-chat"
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_content(self, node_name: str, context: str, node_level: int) -> dict:
        """为节点生成完整内容（带限流）"""
        async with self.semaphore:
            # 根据层级调整难度提示
            level_hint = {
                1: "这是顶层核心主题",
                2: "这是中层关键概念",
                3: "这是底层具体知识点",
            }.get(node_level, "")

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"主题背景: {context}\n层级说明: {level_hint}\n\n请为「{node_name}」生成学习信息："},
            ]

            try:
                result = await self.client.chat(self.model, messages, max_tokens=500)
                parsed = _extract_json(result)
                if parsed:
                    return parsed
            except Exception as e:
                print(f"[ContentAgent] 生成失败 ({node_name}): {e}")

            # 返回默认值
            return {
                "description": f"{node_name}的相关内容",
                "learningObjectives": [f"能够理解{node_name}的基本概念"],
                "keyConcepts": [node_name],
                "knowledgeType": "concept",
                "difficulty": "beginner",
                "estimatedMinutes": 10,
                "questionPatterns": [],
                "commonMistakes": [],
            }


async def generate_tree_v3_parallel(
    client: LLMClient,
    content: str,
    progress_callback: AsyncGenerator[ProgressEvent, None] | None = None,
) -> KnowledgeTreeSchema:
    """V3 两阶段并行策略"""
    planner = PlannerAgent(client)
    content_agent = ContentAgent(client)

    # 阶段1: 规划骨架
    skeleton = await planner.plan(content)
    nodes = skeleton.get("nodes", [])
    topic = skeleton.get("topic", "知识树")

    total_nodes = len(nodes)

    # 阶段2: 并发填充内容
    async def fill_node(i: int, node: dict) -> tuple[int, dict]:
        node_id = str(node.get("id", i + 1))
        node_level = len(node_id.split("."))
        result = await content_agent.generate_content(
            node.get("name", ""),
            f"{topic}: {content[:150]}",
            node_level
        )
        return i, result

    tasks = [fill_node(i, node) for i, node in enumerate(nodes)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 组装结果
    node_contents = {}
    for result in results:
        if isinstance(result, tuple):
            i, content_data = result
            node_contents[i] = content_data

    final_nodes = []
    for i, node in enumerate(nodes):
        content_data = node_contents.get(i, {})
        if isinstance(content_data, Exception):
            content_data = {"description": f"{node.get('name', '')}的相关内容"}

        node_id = str(node.get("id", i + 1))
        node_level = len(node_id.split("."))

        final_nodes.append(KnowledgeNodeSchema(
            id=node_id,
            name=node.get("name", ""),
            description=content_data.get("description", ""),
            parent_id=node.get("parent_id"),
            order=i,
            learning_objectives=content_data.get("learningObjectives", []),
            key_concepts=content_data.get("keyConcepts", []),
            knowledge_type=content_data.get("knowledgeType", "concept"),
            difficulty=content_data.get("difficulty", "beginner"),
            estimated_minutes=content_data.get("estimatedMinutes", 10),
            question_patterns=content_data.get("questionPatterns", []),
            common_mistakes=content_data.get("commonMistakes", []),
        ))

    return KnowledgeTreeSchema(
        topic=topic,
        summary=f"共 {len(final_nodes)} 个节点",
        nodes=final_nodes,
        concepts=[n.name for n in final_nodes if not n.parent_id],
    )


# ==================== 流式进度生成（SSE）====================

async def generate_tree_v3_parallel_stream(
    client: LLMClient,
    content: str,
) -> AsyncGenerator[ProgressEvent | KnowledgeTreeSchema, None]:
    """V3 两阶段并行策略 - 流式版本"""
    planner = PlannerAgent(client)
    content_agent = ContentAgent(client)

    # 阶段1: 规划骨架
    yield ProgressEvent(
        stage="planning",
        progress=5,
        message="正在分析内容结构...",
        total_nodes=0,
        completed_nodes=0,
    )

    skeleton = await planner.plan(content)
    nodes = skeleton.get("nodes", [])
    topic = skeleton.get("topic", "知识树")
    total_nodes = len(nodes)

    yield ProgressEvent(
        stage="planning",
        progress=20,
        message=f"结构规划完成，共 {total_nodes} 个节点",
        total_nodes=total_nodes,
        completed_nodes=0,
    )

    # 阶段2: 并发填充内容（带进度）
    completed = 0
    results = {}
    lock = asyncio.Lock()

    async def fill_node_with_progress(i: int, node: dict):
        nonlocal completed
        node_id = str(node.get("id", i + 1))
        node_level = len(node_id.split("."))
        result = await content_agent.generate_content(
            node.get("name", ""),
            f"{topic}: {content[:150]}",
            node_level
        )
        async with lock:
            completed += 1
            results[i] = result
        return i, result

    # 创建所有任务
    tasks = [fill_node_with_progress(i, node) for i, node in enumerate(nodes)]

    # 使用 as_completed 来跟踪进度
    for coro in asyncio.as_completed(tasks):
        try:
            i, _ = await coro
            node_name = nodes[i].get("name", "") if i < len(nodes) else ""
            progress = 20 + (completed / total_nodes) * 75
            yield ProgressEvent(
                stage="filling",
                progress=progress,
                message=f"已完成: {node_name}",
                current_node=node_name,
                total_nodes=total_nodes,
                completed_nodes=completed,
            )
        except Exception as e:
            print(f"[V3 Stream] 节点处理失败: {e}")

    # 组装结果
    final_nodes = []
    for i, node in enumerate(nodes):
        content_data = results.get(i, {})
        if isinstance(content_data, Exception):
            content_data = {"description": f"{node.get('name', '')}的相关内容"}

        node_id = str(node.get("id", i + 1))

        final_nodes.append(KnowledgeNodeSchema(
            id=node_id,
            name=node.get("name", ""),
            description=content_data.get("description", ""),
            parent_id=node.get("parent_id"),
            order=i,
            learning_objectives=content_data.get("learningObjectives", []),
            key_concepts=content_data.get("keyConcepts", []),
            knowledge_type=content_data.get("knowledgeType", "concept"),
            difficulty=content_data.get("difficulty", "beginner"),
            estimated_minutes=content_data.get("estimatedMinutes", 10),
            question_patterns=content_data.get("questionPatterns", []),
            common_mistakes=content_data.get("commonMistakes", []),
        ))

    yield ProgressEvent(
        stage="done",
        progress=100,
        message="知识树生成完成",
        total_nodes=total_nodes,
        completed_nodes=total_nodes,
    )

    # 最后返回完整的树
    yield KnowledgeTreeSchema(
        topic=topic,
        summary=f"共 {len(final_nodes)} 个节点",
        nodes=final_nodes,
        concepts=[n.name for n in final_nodes if not n.parent_id],
    )


# ==================== 单次生成策略（用户自配 API）====================

SINGLE_REQUEST_PROMPT = """你是知识结构化专家。将内容整理成三层知识树。

要求：
- 第一层: 2-4个核心主题
- 第二层: 每个主题下3-5个关键概念
- 第三层: 每个概念下2-3个具体知识点
- 每个节点需要包含完整的学习属性

返回 JSON:
{"topic":"主题名","summary":"概述","nodes":[{"id":"1","name":"名称","description":"描述","parent_id":null,"order":0,"learningObjectives":["能够..."],"knowledgeType":"concept","difficulty":"beginner","estimatedMinutes":10,"questionPatterns":["题型"],"commonMistakes":["错误"]}],"concepts":["概念1","概念2"]}"""


async def generate_tree_single_request(
    client: LLMClient,
    content: str,
    model_id: str,
) -> KnowledgeTreeSchema:
    """单次请求策略（用户自配 API）"""
    messages = [
        {"role": "system", "content": SINGLE_REQUEST_PROMPT},
        {"role": "user", "content": f"整理成三层知识树：\n{content}"},
    ]

    result = await client.chat(model_id, messages, max_tokens=4000)
    parsed = _extract_json(result)

    if not parsed:
        return KnowledgeTreeSchema(
            topic="知识树",
            summary="",
            nodes=[KnowledgeNodeSchema(id="1", name="根节点", description=content[:100])],
        )

    nodes = [
        KnowledgeNodeSchema(
            id=str(n.get("id", i + 1)),
            name=n.get("name", ""),
            description=n.get("description"),
            parent_id=n.get("parent_id"),
            order=n.get("order", i),
            learning_objectives=n.get("learningObjectives", []),
            key_concepts=n.get("keyConcepts", []),
            knowledge_type=n.get("knowledgeType"),
            difficulty=n.get("difficulty"),
            estimated_minutes=n.get("estimatedMinutes"),
            question_patterns=n.get("questionPatterns", []),
            common_mistakes=n.get("commonMistakes", []),
        )
        for i, n in enumerate(parsed.get("nodes", []))
    ]

    return KnowledgeTreeSchema(
        topic=parsed.get("topic", "知识树"),
        summary=parsed.get("summary", ""),
        nodes=nodes,
        concepts=parsed.get("concepts", []),
    )


# ==================== 主入口 ====================

async def generate_knowledge_tree(
    db: Session,
    user_id: str,
    content: str,
    use_system: bool | None = None,
) -> TreeGenerationResult:
    """
    生成知识树

    Args:
        db: 数据库会话
        user_id: 用户 ID
        content: 要整理的内容
        use_system: 是否使用系统 API（None 表示自动判断）

    Returns:
        TreeGenerationResult: 生成结果
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
        return TreeGenerationResult(
            success=False,
            error=str(e),
            mode="byok",
            strategy="single_request",
            processing_time_ms=(time.time() - start_time) * 1000,
            model_used="",
        )

    client = LLMClient(resolved.api_key, resolved.base_url)

    if resolved.mode == "system":
        # 官方 API: 使用 V3 两阶段并行策略
        # 积分扣除
        request_id = f"tree_generate:{uuid.uuid4()}"
        fixed_points = calculate_fixed_points(FeatureType.TREE_GENERATE)

        try:
            credits_service.reserve_points(
                db, user_id,
                request_id=request_id,
                points=fixed_points,
                meta={"feature": "tree_generate", "contentLength": len(content)},
            )
        except Exception as e:
            return TreeGenerationResult(
                success=False,
                error=f"积分不足: {str(e)}",
                mode="system",
                strategy="v3_parallel",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="",
            )

        try:
            tree = await generate_tree_v3_parallel(client, content)
            # 成功：确认扣费
            credits_service.finalize_reservation(
                db, user_id,
                request_id=request_id,
                reserved_points=fixed_points,
                actual_points=fixed_points,
                meta={"feature": "tree_generate", "success": True},
            )
            return TreeGenerationResult(
                success=True,
                tree=tree,
                mode="system",
                strategy="v3_parallel",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="deepseek-chat",
            )
        except Exception as e:
            # 失败：退还积分
            credits_service.finalize_reservation(
                db, user_id,
                request_id=request_id,
                reserved_points=fixed_points,
                actual_points=0,
                meta={"feature": "tree_generate", "error": str(e)},
            )
            return TreeGenerationResult(
                success=False,
                error=str(e),
                mode="system",
                strategy="v3_parallel",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="deepseek-chat",
            )
    else:
        # 用户自配 API: 使用单次生成策略（不扣积分）
        model_id = resolved.model_id or "gpt-4"
        try:
            tree = await generate_tree_single_request(client, content, model_id)
            return TreeGenerationResult(
                success=True,
                tree=tree,
                mode="byok",
                strategy="single_request",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used=model_id,
            )
        except Exception as e:
            return TreeGenerationResult(
                success=False,
                error=str(e),
                mode="byok",
                strategy="single_request",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used=model_id,
            )


async def generate_knowledge_tree_stream(
    db: Session,
    user_id: str,
    content: str,
    use_system: bool | None = None,
) -> AsyncGenerator[ProgressEvent | TreeGenerationResult, None]:
    """
    流式生成知识树（带进度反馈）

    Args:
        db: 数据库会话
        user_id: 用户 ID
        content: 要整理的内容
        use_system: 是否使用系统 API

    Yields:
        ProgressEvent 或 TreeGenerationResult
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
        yield TreeGenerationResult(
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
        # 官方 API: 使用 V3 流式版本
        # 积分扣除
        request_id = f"tree_generate_stream:{uuid.uuid4()}"
        fixed_points = calculate_fixed_points(FeatureType.TREE_GENERATE)

        try:
            credits_service.reserve_points(
                db, user_id,
                request_id=request_id,
                points=fixed_points,
                meta={"feature": "tree_generate_stream", "contentLength": len(content)},
            )
        except Exception as e:
            yield TreeGenerationResult(
                success=False,
                error=f"积分不足: {str(e)}",
                mode="system",
                strategy="v3_parallel",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="",
            )
            return

        try:
            tree = None
            async for event in generate_tree_v3_parallel_stream(client, content):
                if isinstance(event, ProgressEvent):
                    yield event
                elif isinstance(event, KnowledgeTreeSchema):
                    tree = event

            # 成功：确认扣费
            credits_service.finalize_reservation(
                db, user_id,
                request_id=request_id,
                reserved_points=fixed_points,
                actual_points=fixed_points,
                meta={"feature": "tree_generate_stream", "success": True},
            )
            yield TreeGenerationResult(
                success=True,
                tree=tree,
                mode="system",
                strategy="v3_parallel",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="deepseek-chat",
            )
        except Exception as e:
            # 失败：退还积分
            credits_service.finalize_reservation(
                db, user_id,
                request_id=request_id,
                reserved_points=fixed_points,
                actual_points=0,
                meta={"feature": "tree_generate_stream", "error": str(e)},
            )
            yield TreeGenerationResult(
                success=False,
                error=str(e),
                mode="system",
                strategy="v3_parallel",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="deepseek-chat",
            )
    else:
        # 用户自配 API: 单次请求（无流式）
        model_id = resolved.model_id or "gpt-4"
        yield ProgressEvent(
            stage="generating",
            progress=10,
            message="正在生成知识树...",
            total_nodes=0,
            completed_nodes=0,
        )
        try:
            tree = await generate_tree_single_request(client, content, model_id)
            yield ProgressEvent(
                stage="done",
                progress=100,
                message="知识树生成完成",
                total_nodes=len(tree.nodes),
                completed_nodes=len(tree.nodes),
            )
            yield TreeGenerationResult(
                success=True,
                tree=tree,
                mode="byok",
                strategy="single_request",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used=model_id,
            )
        except Exception as e:
            yield TreeGenerationResult(
                success=False,
                error=str(e),
                mode="byok",
                strategy="single_request",
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used=model_id,
            )
