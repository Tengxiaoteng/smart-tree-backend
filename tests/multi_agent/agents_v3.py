"""
多 Agent 架构 v3 - 两阶段并行生成

架构：
1. PlannerAgent: 快速生成知识树骨架（结构+节点名）
2. ContentAgents: 并发填充每个节点的详细内容

时间 = 规划时间 + max(并发生成时间)
"""
import asyncio
import json
import time
import re
from typing import Any, Optional

import httpx

from .schemas import (
    IntentResult,
    KnowledgeTree,
    KnowledgeNode,
    MultiAgentResult,
    TaskDifficulty,
    TaskType,
)


MODEL_CONFIG = {
    "fast": "qwen-turbo",
    "standard": "qwen-plus",
    "advanced": "qwen-max",
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


class PlannerAgent:
    """规划 Agent - 快速生成知识树骨架"""

    SYSTEM_PROMPT = """你是知识结构规划专家。快速生成三层知识树的骨架结构。

要求：
- 第一层: 2-4个核心主题
- 第二层: 每个主题下2-4个关键概念
- 第三层: 每个概念下2-3个知识点
- 只需要节点名称，不需要详细描述

返回 JSON:
{"topic":"主题","nodes":[{"id":"1","name":"一级节点","parent_id":null},{"id":"1.1","name":"二级节点","parent_id":"1"},{"id":"1.1.1","name":"三级节点","parent_id":"1.1"}]}"""

    def __init__(self, client: LLMClient):
        self.client = client
        self.model = MODEL_CONFIG["standard"]  # 规划用 qwen-plus（平衡速度和质量）

    async def plan(self, content: str) -> dict:
        """生成知识树骨架"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"为以下内容规划知识树结构：\n{content}"},
        ]

        start = time.time()
        result = await self.client.chat(self.model, messages, max_tokens=1500)
        elapsed = (time.time() - start) * 1000

        print(f"[PlannerAgent] 耗时: {elapsed:.0f}ms, 模型: {self.model}")

        parsed = _extract_json(result)
        if not parsed:
            print(f"[PlannerAgent] JSON 解析失败，返回默认结构")
            return {
                "topic": "知识树",
                "nodes": [{"id": "1", "name": "根节点", "parent_id": None}]
            }

        print(f"[PlannerAgent] 规划完成，节点数: {len(parsed.get('nodes', []))}")
        return parsed


class ContentAgent:
    """内容生成 Agent - 为单个节点生成详细内容"""

    SYSTEM_PROMPT = """你是知识内容专家。为给定的知识点生成详细描述。

要求：
- 描述 50-100 字
- 内容准确、专业
- 适合学习理解

只返回描述文本，不需要 JSON。"""

    def __init__(self, client: LLMClient, max_concurrent: int = 10):
        self.client = client
        self.model = MODEL_CONFIG["advanced"]  # 使用最强模型 qwen-max
        self.semaphore = asyncio.Semaphore(max_concurrent)  # 10 并发

    async def generate_content(self, node_name: str, context: str) -> str:
        """为节点生成内容（带限流）"""
        async with self.semaphore:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"主题背景: {context}\n\n请为「{node_name}」生成详细描述："},
            ]

            try:
                result = await self.client.chat(self.model, messages, max_tokens=200)
                return result.strip()
            except Exception as e:
                print(f"[ContentAgent] 生成失败 ({node_name}): {e}")
                return f"{node_name}的相关内容"


class TwoStageOrchestrator:
    """两阶段并行协调器"""

    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = LLMClient(api_key, base_url)
        self.planner = PlannerAgent(self.client)
        self.content_agent = ContentAgent(self.client)

    async def process(self, user_input: str) -> MultiAgentResult:
        """两阶段处理"""
        total_start = time.time()

        # ========== 阶段1: 规划 ==========
        print("\n[阶段1] 规划知识树结构...")
        plan_start = time.time()
        skeleton = await self.planner.plan(user_input)
        plan_time = (time.time() - plan_start) * 1000
        print(f"[阶段1] 完成，耗时: {plan_time:.0f}ms")

        nodes = skeleton.get("nodes", [])
        topic = skeleton.get("topic", "知识树")

        # ========== 阶段2: 并发填充内容 ==========
        print(f"\n[阶段2] 并发生成 {len(nodes)} 个节点的内容...")
        fill_start = time.time()

        # 创建所有内容生成任务
        tasks = []
        for node in nodes:
            task = self.content_agent.generate_content(
                node.get("name", ""),
                f"主题: {topic}, 内容: {user_input[:200]}"
            )
            tasks.append(task)

        # 并发执行所有任务
        descriptions = await asyncio.gather(*tasks, return_exceptions=True)
        fill_time = (time.time() - fill_start) * 1000
        print(f"[阶段2] 完成，耗时: {fill_time:.0f}ms (并发 {len(nodes)} 个请求)")

        # 组装最终结果
        final_nodes = []
        for i, node in enumerate(nodes):
            desc = descriptions[i] if i < len(descriptions) else ""
            if isinstance(desc, Exception):
                desc = f"{node.get('name', '')}的相关内容"

            final_nodes.append(KnowledgeNode(
                id=str(node.get("id", i + 1)),
                name=node.get("name", ""),
                description=str(desc),
                parent_id=node.get("parent_id"),
                order=i,
            ))

        # 统计层级
        level_counts = {}
        for node in final_nodes:
            level = len(str(node.id).split("."))
            level_counts[level] = level_counts.get(level, 0) + 1

        total_time = (time.time() - total_start) * 1000

        print(f"\n[总结] 总耗时: {total_time:.0f}ms = 规划 {plan_time:.0f}ms + 并发填充 {fill_time:.0f}ms")
        print(f"[总结] 层级分布: {level_counts}")

        tree = KnowledgeTree(
            topic=topic,
            summary=f"共 {len(final_nodes)} 个节点，{len(level_counts)} 层结构",
            nodes=final_nodes,
            concepts=[n.name for n in final_nodes if not n.parent_id],
        )

        return MultiAgentResult(
            intent=IntentResult(
                task_type=TaskType.TREE_GENERATION,
                difficulty=TaskDifficulty.MEDIUM,
                recommended_model="qwen-turbo",
                reasoning="两阶段并行生成",
            ),
            processing_time_ms=total_time,
            model_used="qwen-turbo (并行)",
            result=tree.model_dump(),
        )


# ========== 批量并发优化版 ==========

class BatchContentAgent:
    """批量内容生成 Agent - 一次请求生成多个节点内容"""

    SYSTEM_PROMPT = """你是知识内容专家。为多个知识点生成详细描述。

要求：
- 每个描述 50-80 字
- 内容准确专业

返回 JSON 数组: [{"id": "节点ID", "description": "描述内容"}, ...]"""

    def __init__(self, client: LLMClient):
        self.client = client
        self.model = MODEL_CONFIG["standard"]  # 批量用标准模型

    async def generate_batch(self, nodes: list[dict], context: str) -> list[dict]:
        """批量生成内容"""
        node_list = "\n".join([f"- {n['id']}: {n['name']}" for n in nodes])

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"主题: {context}\n\n节点列表:\n{node_list}\n\n请为每个节点生成描述："},
        ]

        result = await self.client.chat(self.model, messages, max_tokens=3000)
        parsed = _extract_json(result)

        if parsed and isinstance(parsed, list):
            return parsed
        return []


class TwoStageBatchOrchestrator:
    """两阶段批量并发协调器 - 更高效"""

    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = LLMClient(api_key, base_url)
        self.planner = PlannerAgent(self.client)
        self.batch_agent = BatchContentAgent(self.client)

    async def process(self, user_input: str) -> MultiAgentResult:
        """两阶段处理（批量版）"""
        total_start = time.time()

        # 阶段1: 规划
        print("\n[阶段1-Batch] 规划知识树结构...")
        plan_start = time.time()
        skeleton = await self.planner.plan(user_input)
        plan_time = (time.time() - plan_start) * 1000

        nodes = skeleton.get("nodes", [])
        topic = skeleton.get("topic", "知识树")
        print(f"[阶段1-Batch] 完成，规划 {len(nodes)} 个节点，耗时: {plan_time:.0f}ms")

        # 阶段2: 按层级分批并发生成
        print(f"\n[阶段2-Batch] 分批并发生成内容...")
        fill_start = time.time()

        # 按层级分组
        level_groups = {}
        for node in nodes:
            level = len(str(node.get("id", "1")).split("."))
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)

        # 并发生成各层级内容
        all_descriptions = {}
        tasks = []
        for level, level_nodes in level_groups.items():
            task = self.batch_agent.generate_batch(level_nodes, f"{topic}: {user_input[:150]}")
            tasks.append((level, task))

        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        for i, (level, _) in enumerate(tasks):
            if not isinstance(results[i], Exception) and results[i]:
                for item in results[i]:
                    all_descriptions[item.get("id")] = item.get("description", "")

        fill_time = (time.time() - fill_start) * 1000
        print(f"[阶段2-Batch] 完成，耗时: {fill_time:.0f}ms (并发 {len(level_groups)} 批)")

        # 组装结果
        final_nodes = []
        for i, node in enumerate(nodes):
            node_id = str(node.get("id", i + 1))
            desc = all_descriptions.get(node_id, f"{node.get('name', '')}的相关内容")

            final_nodes.append(KnowledgeNode(
                id=node_id,
                name=node.get("name", ""),
                description=desc,
                parent_id=node.get("parent_id"),
                order=i,
            ))

        total_time = (time.time() - total_start) * 1000

        print(f"\n[总结-Batch] 总耗时: {total_time:.0f}ms = 规划 {plan_time:.0f}ms + 批量填充 {fill_time:.0f}ms")

        tree = KnowledgeTree(
            topic=topic,
            summary=f"共 {len(final_nodes)} 个节点",
            nodes=final_nodes,
            concepts=[n.name for n in final_nodes if not n.parent_id],
        )

        return MultiAgentResult(
            intent=IntentResult(
                task_type=TaskType.TREE_GENERATION,
                difficulty=TaskDifficulty.MEDIUM,
                recommended_model="qwen-turbo + qwen-plus",
                reasoning="两阶段批量并发生成",
            ),
            processing_time_ms=total_time,
            model_used="qwen-turbo (规划) + qwen-plus (批量填充)",
            result=tree.model_dump(),
        )
