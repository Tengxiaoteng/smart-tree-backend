"""
智能资料分类服务 v4.2 - 并行优化版

核心改进：
- 关键词提取 LLM 和 资料 Embedding 并行执行
- 减少串行等待时间

流程：
1. Step 0: 并行执行
   - LLM 提取关键词
   - Embedding 向量化资料（直接用原文摘要）
2. Step 1: Embedding 选树（用并行结果）
3. Step 2: Embedding 全局 Top-K 检索
4. Step 3: LLM 最终决策
"""

import json
import time
import asyncio
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.schemas.material_classifier import (
    KeywordExtractionOutput,
    TreeMatch,
    TreeSelectionOutput,
    FinalDecisionOutput,
    MaterialClassificationResult,
    MockTree,
    MockNode,
)
from app.services.embedding_service import (
    EmbeddingService,
    SimpleVectorStore,
    build_node_text,
    build_tree_text,
)

logger = logging.getLogger(__name__)


class ClassifierConfigV4_2:
    """分类器配置 v4.2"""
    # Embedding 检索配置
    TREE_TOP_K = 3
    TREE_MIN_SIMILARITY = 0.2

    # 全局检索配置
    GLOBAL_TOP_K = 10           # 全局检索的候选节点数
    FINAL_TOP_K = 5             # 送给 LLM 的最终候选数
    NODE_MIN_SIMILARITY = 0.3

    # 树匹配阈值
    TREE_MATCH_THRESHOLD = 0.3


# ============== Prompts ==============

KEYWORD_EXTRACTION_PROMPT = """你是一个专业的文本分析助手。请从学习资料中提取关键信息。

请严格按照以下 JSON 格式输出：

{{
    "keywords": ["关键词1", "关键词2", ...],
    "domain": "学科领域",
    "content_type": "concept/case/formula/procedure/mixed",
    "summary": "100字以内的摘要",
    "detailed_description": "200-300字的详细描述"
}}

资料内容：
{content}"""

FINAL_DECISION_PROMPT = """你是一个知识管理助手。根据资料信息和候选节点，选择最匹配的节点。

资料信息：
- 关键词：{keywords}
- 领域：{domain}
- 摘要：{summary}

候选节点（按相似度排序）：
{candidates_description}

请输出 JSON 格式：
{{
    "action": "link_to_node|create_child_node|create_sibling_node",
    "target_node_id": "节点ID",
    "suggested_node_name": "新节点名称（create_* 时填）",
    "suggested_parent_id": "父节点ID（create_child_node 时填）",
    "confidence": 0.0-1.0,
    "reason": "决策理由",
    "extracted_concepts": ["概念1", "概念2"]
}}

决策规则：
1. link_to_node: 资料与某个节点完全匹配
2. create_child_node: 资料是某节点的子知识点
3. create_sibling_node: 资料是同级别的新知识点

只输出 JSON。"""


class MaterialClassifierV4_2:
    """
    智能资料分类器 v4.2 - 并行优化版
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        llm_model: str = "qwen-plus",
        embedding_api_key: Optional[str] = None,
        embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_model: str = "text-embedding-v4",
        embedding_dimension: int = 512,
    ):
        self.llm = ChatOpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=llm_model,
            temperature=0.3,
            max_tokens=2048,
        )

        self.embedding_service = EmbeddingService(
            api_key=embedding_api_key or llm_api_key,
            base_url=embedding_base_url,
            model=embedding_model,
            dimension=embedding_dimension,
        )

        self.vector_store = SimpleVectorStore()
        self._stats = {}

    def index_trees_and_nodes(
        self,
        trees: list[MockTree],
        nodes_by_tree: dict[str, list[MockNode]],
    ):
        """建立向量索引"""
        logger.info(f"开始建立向量索引: {len(trees)} 棵树")

        items_to_embed = []

        # 树
        for tree in trees:
            tree_nodes = nodes_by_tree.get(tree.id, [])
            root_nodes = [n for n in tree_nodes if n.parent_id is None]
            text = build_tree_text(tree, root_nodes)
            items_to_embed.append({
                "id": f"tree:{tree.id}",
                "text": text,
                "metadata": {"type": "tree", "tree_id": tree.id, "name": tree.name}
            })

        # 节点
        for tree_id, nodes in nodes_by_tree.items():
            for node in nodes:
                text = build_node_text(node)
                items_to_embed.append({
                    "id": f"node:{node.id}",
                    "text": text,
                    "metadata": {
                        "type": "node",
                        "node_id": node.id,
                        "tree_id": tree_id,
                        "name": node.name,
                        "parent_id": node.parent_id,
                    }
                })

        # 批量 Embedding
        texts = [item["text"] for item in items_to_embed]
        embeddings = self.embedding_service.embed_texts(texts)

        for item, embedding in zip(items_to_embed, embeddings):
            item["embedding"] = embedding
        self.vector_store.add_batch(items_to_embed)

        logger.info(f"向量索引建立完成，共 {len(self.vector_store)} 个向量")

    async def classify(
        self,
        content: str,
        trees: list[MockTree],
        nodes_by_tree: dict[str, list[MockNode]],
        materials_by_node: dict[str, list[str]],
    ) -> MaterialClassificationResult:
        """执行分类（并行优化版）"""
        start_time = time.time()
        self._stats = {"llm_calls": 0, "embedding_calls": 0, "tokens_used": 0}

        result = MaterialClassificationResult()

        # ========== Step 0: 并行执行 LLM 关键词提取 + Embedding 向量化 ==========
        logger.info("Step 0: 并行执行关键词提取和向量化")

        parallel_start = time.time()

        # 创建并行任务
        keyword_task = self._extract_keywords(content)

        # 用内容摘要做 Embedding（取前1000字）
        content_for_embedding = content[:1000]
        embedding_task = asyncio.get_event_loop().run_in_executor(
            None,
            self.embedding_service.embed_text,
            content_for_embedding
        )

        # 并行等待
        keyword_result, material_embedding = await asyncio.gather(
            keyword_task,
            embedding_task
        )

        parallel_time = time.time() - parallel_start
        logger.info(f"  并行步骤完成: {int(parallel_time*1000)}ms")

        self._stats["embedding_calls"] += 1

        result.keywords = keyword_result.keywords
        result.summary = keyword_result.summary
        result.detailed_description = keyword_result.detailed_description
        result.domain = keyword_result.domain
        result.content_type = keyword_result.content_type

        # ========== Step 1: Embedding 选树 ==========
        logger.info("Step 1: Embedding 选树")

        tree_candidates = self.vector_store.search_by_type(
            query_embedding=material_embedding,
            type_value="tree",
            top_k=ClassifierConfigV4_2.TREE_TOP_K,
            min_similarity=ClassifierConfigV4_2.TREE_MIN_SIMILARITY,
        )

        if not tree_candidates:
            result.suggest_new_tree = True
            result.suggested_tree_name = result.domain
            result.total_api_calls = self._stats["llm_calls"]
            result.total_tokens_used = self._stats["tokens_used"]
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            return result

        # 构建匹配的树
        matched_trees = []
        for tc in tree_candidates:
            if tc["similarity"] >= ClassifierConfigV4_2.TREE_MATCH_THRESHOLD:
                matched_trees.append(TreeMatch(
                    tree_id=tc["metadata"]["tree_id"],
                    tree_name=tc["metadata"]["name"],
                    confidence=tc["similarity"],
                    reason=f"向量相似度: {tc['similarity']:.2f}"
                ))

        result.matched_trees = matched_trees

        if not matched_trees:
            result.suggest_new_tree = True
            result.suggested_tree_name = result.domain

        # ========== Step 2: 全局 Top-K 检索 ==========
        for tree_match in matched_trees:
            tree_id = tree_match.tree_id
            tree_nodes = nodes_by_tree.get(tree_id, [])

            logger.info(f"Step 2: 在树「{tree_match.tree_name}」中全局检索")

            # 全局检索所有节点
            node_candidates = self.vector_store.search(
                query_embedding=material_embedding,
                top_k=ClassifierConfigV4_2.GLOBAL_TOP_K,
                filter_fn=lambda id, meta: meta.get("tree_id") == tree_id and meta.get("type") == "node",
                min_similarity=ClassifierConfigV4_2.NODE_MIN_SIMILARITY,
            )

            if not node_candidates:
                result.tree_node_decisions[tree_id] = self._default_decision(result.keywords)
                continue

            # ========== Step 3: LLM 最终决策 ==========
            logger.info(f"Step 3: LLM 最终决策 ({len(node_candidates)} 个候选)")

            # 构建候选描述
            candidates_desc = []
            for i, nc in enumerate(node_candidates[:ClassifierConfigV4_2.FINAL_TOP_K]):
                node_id = nc["metadata"]["node_id"]
                node = next((n for n in tree_nodes if n.id == node_id), None)
                if node:
                    # 获取路径
                    path = self._get_node_path(node, tree_nodes)
                    desc = f"{i+1}. [{node.id}] {' > '.join(path)}\n   描述: {node.description or '无'}\n   关键概念: {', '.join(node.key_concepts) if node.key_concepts else '无'}\n   相似度: {nc['similarity']:.2f}"
                    candidates_desc.append(desc)

            decision = await self._make_final_decision(
                keywords=result.keywords,
                domain=result.domain,
                summary=result.summary,
                candidates_description="\n".join(candidates_desc),
            )

            result.tree_node_decisions[tree_id] = decision

        result.total_api_calls = self._stats["llm_calls"]
        result.total_tokens_used = self._stats["tokens_used"]
        result.processing_time_ms = int((time.time() - start_time) * 1000)

        return result

    def _get_node_path(self, node: MockNode, all_nodes: list[MockNode]) -> list[str]:
        """获取节点的完整路径"""
        path = [node.name]
        current = node
        nodes_dict = {n.id: n for n in all_nodes}

        while current.parent_id:
            parent = nodes_dict.get(current.parent_id)
            if parent:
                path.insert(0, parent.name)
                current = parent
            else:
                break

        return path

    async def _extract_keywords(self, content: str) -> KeywordExtractionOutput:
        """提取关键词"""
        truncated = content[:6000]

        prompt = KEYWORD_EXTRACTION_PROMPT.format(content=truncated)

        response = await self.llm.ainvoke(prompt)
        self._stats["llm_calls"] += 1
        self._stats["tokens_used"] += response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0

        try:
            content_str = response.content
            if "```json" in content_str:
                content_str = content_str.split("```json")[1].split("```")[0]
            elif "```" in content_str:
                content_str = content_str.split("```")[1].split("```")[0]
            parsed = json.loads(content_str.strip())
            keywords = parsed.get("keywords", [])
            # 确保至少5个关键词
            if len(keywords) < 5:
                keywords.extend(content[:200].split()[:5-len(keywords)])
            return KeywordExtractionOutput(
                keywords=keywords[:20],
                key_sentences=parsed.get("key_sentences", []),
                domain=parsed.get("domain", "未知"),
                content_type=parsed.get("content_type", "mixed"),
                summary=parsed.get("summary", content[:100]),
                detailed_description=parsed.get("detailed_description", content[:300])
            )
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"关键词提取解析失败: {e}")
            # 从内容中提取关键词
            words = [w for w in content[:500].split() if len(w) > 1][:10]
            if len(words) < 5:
                words.extend(["关键词"] * (5 - len(words)))
            return KeywordExtractionOutput(
                keywords=words,
                key_sentences=[],
                domain="未知",
                content_type="mixed",
                summary=content[:100],
                detailed_description=content[:300]
            )

    async def _make_final_decision(
        self,
        keywords: list[str],
        domain: str,
        summary: str,
        candidates_description: str,
    ) -> FinalDecisionOutput:
        """最终决策"""
        prompt = FINAL_DECISION_PROMPT.format(
            keywords=", ".join(keywords[:10]),
            domain=domain,
            summary=summary,
            candidates_description=candidates_description,
        )

        response = await self.llm.ainvoke(prompt)
        self._stats["llm_calls"] += 1
        self._stats["tokens_used"] += response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0

        try:
            content_str = response.content
            if "```json" in content_str:
                content_str = content_str.split("```json")[1].split("```")[0]
            elif "```" in content_str:
                content_str = content_str.split("```")[1].split("```")[0]
            parsed = json.loads(content_str.strip())
            return FinalDecisionOutput(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"最终决策解析失败: {e}")
            return FinalDecisionOutput(
                action="link_to_node",
                target_node_id=None,
                confidence=0.3,
                reason="解析失败",
                extracted_concepts=[]
            )

    def _default_decision(self, keywords: list[str]) -> FinalDecisionOutput:
        """默认决策"""
        return FinalDecisionOutput(
            action="create_child_node",
            suggested_node_name=keywords[0] if keywords else "新知识点",
            confidence=0.5,
            reason="没有找到匹配节点",
            extracted_concepts=keywords[:5]
        )

    def get_stats(self) -> dict:
        """获取统计信息"""
        return self._stats
