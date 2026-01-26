"""
智能资料分类服务 v3 - Embedding 增强版

架构：
1. Step 0: LLM 提取关键词和描述
2. Step 1: Embedding 向量化 + 向量检索预筛选
3. Step 2: LLM 对候选结果精细判断
4. Step 3: LLM 最终决策

优势：
- 减少 70%+ 的 LLM 调用
- 支持大规模知识树（1000+ 节点）
- 更快的响应速度
"""

import json
import time
import asyncio
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from app.schemas.material_classifier import (
    KeywordExtractionOutput,
    TreeSummary,
    TreeMatch,
    TreeSelectionOutput,
    NodeSummary,
    NodeMatchOutput,
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


# ============== 配置 ==============

class ClassifierConfigV3:
    """分类器配置 v3"""
    # Embedding 检索配置
    TREE_TOP_K = 3              # 向量检索返回的候选树数量
    NODE_TOP_K = 10             # 向量检索返回的候选节点数量
    TREE_MIN_SIMILARITY = 0.3   # 树的最小相似度阈值
    NODE_MIN_SIMILARITY = 0.4   # 节点的最小相似度阈值

    # LLM 配置
    TREE_MATCH_THRESHOLD = 0.5  # 树匹配置信度阈值
    NODE_MATCH_THRESHOLD = 0.3  # 节点匹配置信度阈值


# ============== Prompts ==============

KEYWORD_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的文本分析助手。你的任务是从学习资料中提取关键信息。

请严格按照以下 JSON 格式输出，不要输出其他内容：

{{
    "keywords": ["关键词1", "关键词2", ...],  // 8-15个关键词，优先提取专业术语、概念名称
    "key_sentences": ["关键句1", "关键句2", ...],  // 3-5个最能代表内容的句子
    "domain": "学科领域",  // 如：数学、编程、历史、物理、机器学习、数据库等，要尽可能具体
    "content_type": "内容类型",  // concept/case/formula/procedure/mixed
    "summary": "100字以内的摘要",
    "detailed_description": "200-300字的详细描述"
}}

注意：
1. keywords 要提取最能代表内容的专业术语、核心概念、技术名词
2. domain 要尽可能具体（如"机器学习-聚类算法"而不是"计算机"）
3. detailed_description 要详细说明资料的核心主题、主要知识点、适用场景"""),
    ("user", "请分析以下资料并提取关键信息：\n\n{content}")
])

# 简化的树确认 Prompt（因为已经通过 Embedding 预筛选了）
TREE_CONFIRM_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识管理助手。根据资料信息和候选知识树，判断资料应该归属于哪些树。

候选知识树（已通过向量相似度预筛选）：
{trees_description}

请输出 JSON 格式：
{{
    "matched_trees": [
        {{
            "tree_id": "树的ID",
            "tree_name": "树的名称",
            "confidence": 0.0-1.0,
            "reason": "匹配理由"
        }}
    ],
    "suggest_new_tree": false,
    "suggested_tree_name": null
}}

规则：
1. confidence > 0.5 的树算匹配
2. 如果所有候选树都不匹配（confidence < 0.3），建议创建新树
3. 只输出 JSON"""),
    ("user", """资料信息：
关键词：{keywords}
领域：{domain}
摘要：{summary}

请判断应该归属于哪些知识树。""")
])

# 简化的节点匹配 Prompt（因为已经通过 Embedding 预筛选了）
NODE_MATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识管理助手。在知识树「{tree_name}」中，根据资料信息选择最匹配的节点。

候选节点（已通过向量相似度预筛选，按相似度排序）：
{nodes_description}

请输出 JSON 格式：
{{
    "best_match_node_id": "节点ID 或 null",
    "best_match_node_name": "节点名称 或 null",
    "confidence": 0.0-1.0,
    "reason": "判断理由"
}}

规则：
1. 选择语义最匹配的节点
2. 如果没有合适匹配（confidence < 0.3），设 best_match_node_id 为 null
3. 只输出 JSON"""),
    ("user", """资料信息：
关键词：{keywords}
摘要：{summary}

请选择最匹配的节点。""")
])

FINAL_DECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识管理助手。现在已经找到了最可能的目标节点，请做最终决策。

目标节点信息：
- ID: {target_node_id}
- 名称: {target_node_name}
- 描述: {target_node_description}
- 关键概念: {target_node_concepts}
- 已有资料: {existing_materials}

同级节点：
{sibling_nodes_description}

请决定如何处理这份资料。输出 JSON 格式：
{{
    "action": "link_to_node|create_child_node|create_sibling_node|update_existing",
    "target_node_id": "节点ID",
    "suggested_node_name": "新节点名称（create_* 时填）",
    "suggested_parent_id": "父节点ID（create_child_node 时填）",
    "confidence": 0.0-1.0,
    "reason": "决策理由",
    "extracted_concepts": ["新概念1", "新概念2"]
}}

决策规则：
1. link_to_node: 资料与目标节点完全匹配
2. create_child_node: 资料是目标节点的子知识点
3. create_sibling_node: 资料是同级别的新知识点
4. update_existing: 资料是对已有资料的补充

只输出 JSON。"""),
    ("user", """资料信息：
关键词：{keywords}
摘要：{summary}
内容类型：{content_type}

请做出最终决策。""")
])


# ============== 主分类器 ==============

class MaterialClassifierV3:
    """
    智能资料分类器 v3 - Embedding 增强版

    使用流程：
    1. 初始化时传入 LLM 和 Embedding 服务
    2. 调用 index_trees_and_nodes 建立向量索引
    3. 调用 classify 进行分类
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
        vector_store_path: Optional[str] = None,
    ):
        # LLM 客户端
        self.llm = ChatOpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=llm_model,
            temperature=0.3,
            max_tokens=2048,
        )

        # Embedding 服务（默认与 LLM 使用相同的 API Key）
        self.embedding_service = EmbeddingService(
            api_key=embedding_api_key or llm_api_key,
            base_url=embedding_base_url,
            model=embedding_model,
            dimension=embedding_dimension,
        )

        # 向量存储
        self.vector_store = SimpleVectorStore(storage_path=vector_store_path)

        # 统计信息
        self.stats_list: list[dict] = []

    def index_trees_and_nodes(
        self,
        trees: list[MockTree],
        nodes_by_tree: dict[str, list[MockNode]],
    ):
        """
        为树和节点建立向量索引

        Args:
            trees: 知识树列表
            nodes_by_tree: 每棵树的节点列表
        """
        logger.info(f"开始建立向量索引: {len(trees)} 棵树")

        # 收集所有需要向量化的文本
        items_to_embed = []

        # 树
        for tree in trees:
            tree_nodes = nodes_by_tree.get(tree.id, [])
            root_nodes = [n for n in tree_nodes if n.parent_id is None]
            text = build_tree_text(tree, root_nodes)
            items_to_embed.append({
                "id": f"tree:{tree.id}",
                "text": text,
                "metadata": {
                    "type": "tree",
                    "tree_id": tree.id,
                    "name": tree.name,
                }
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

        # 批量生成 Embedding
        logger.info(f"生成 {len(items_to_embed)} 个向量...")
        texts = [item["text"] for item in items_to_embed]
        embeddings = self.embedding_service.embed_texts(texts)

        # 存储
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
        """
        执行分类（Embedding 增强版）

        Args:
            content: 资料内容
            trees: 知识树列表
            nodes_by_tree: 每棵树的节点列表
            materials_by_node: 每个节点的资料名称列表

        Returns:
            分类结果
        """
        start_time = time.time()
        self.stats_list = []
        total_tokens = 0
        embedding_calls = 0

        result = MaterialClassificationResult()
        content_length = len(content)

        # ========== Step 0: 关键词提取（LLM）==========
        logger.info(f"Step 0: 提取关键词（内容长度: {content_length}）")
        keyword_result, stats = await self._extract_keywords(content, content_length)
        self.stats_list.append(stats)
        total_tokens += stats.get("tokens_used", 0)

        result.keywords = keyword_result.keywords
        result.summary = keyword_result.summary
        result.detailed_description = keyword_result.detailed_description
        result.domain = keyword_result.domain
        result.content_type = keyword_result.content_type

        # ========== Step 1: Embedding 向量检索 ==========
        logger.info("Step 1: Embedding 向量检索")

        # 生成资料的 Embedding
        material_text = f"{result.domain} | {' '.join(result.keywords)} | {result.summary}"
        material_embedding = self.embedding_service.embed_text(material_text)
        embedding_calls += 1

        # 检索候选树
        tree_candidates = self.vector_store.search_by_type(
            query_embedding=material_embedding,
            type_value="tree",
            top_k=ClassifierConfigV3.TREE_TOP_K,
            min_similarity=ClassifierConfigV3.TREE_MIN_SIMILARITY,
        )

        logger.info(f"  向量检索到 {len(tree_candidates)} 棵候选树")
        for tc in tree_candidates:
            logger.info(f"    - {tc['metadata']['name']} (相似度: {tc['similarity']:.3f})")

        self.stats_list.append({
            "step": "embedding_tree_search",
            "candidates": len(tree_candidates),
            "embedding_calls": 1,
        })

        # ========== Step 2: LLM 确认树匹配 ==========
        if tree_candidates:
            logger.info("Step 2: LLM 确认树匹配")

            # 构建候选树描述
            trees_desc = []
            for tc in tree_candidates:
                tree_id = tc["metadata"]["tree_id"]
                tree = next((t for t in trees if t.id == tree_id), None)
                if tree:
                    desc = f"- ID: {tree.id}\n  名称: {tree.name}\n  范围: {', '.join(tree.scope) or '未定义'}\n  向量相似度: {tc['similarity']:.2f}"
                    trees_desc.append(desc)

            tree_selection, stats = await self._confirm_trees(
                trees_description="\n".join(trees_desc),
                keywords=result.keywords,
                domain=result.domain,
                summary=result.summary,
            )
            self.stats_list.append(stats)
            total_tokens += stats.get("tokens_used", 0)

            result.matched_trees = tree_selection.matched_trees
            result.suggest_new_tree = tree_selection.suggest_new_tree
            result.suggested_tree_name = tree_selection.suggested_tree_name
        else:
            # 没有候选树，建议新建
            logger.info("  没有候选树，建议创建新树")
            result.suggest_new_tree = True
            result.suggested_tree_name = result.domain or result.keywords[0] if result.keywords else "新知识领域"

        # 过滤低置信度的树
        matched_trees = [t for t in result.matched_trees if t.confidence >= ClassifierConfigV3.TREE_MATCH_THRESHOLD]

        if not matched_trees and not result.suggest_new_tree:
            result.suggest_new_tree = True
            result.suggested_tree_name = result.domain

        # ========== Step 3: 对每棵匹配的树进行节点定位 ==========
        for tree_match in matched_trees:
            tree_id = tree_match.tree_id
            tree_name = tree_match.tree_name
            tree_nodes = nodes_by_tree.get(tree_id, [])

            logger.info(f"Step 3: 在树「{tree_name}」中定位节点")

            # Embedding 检索候选节点
            node_candidates = self.vector_store.search(
                query_embedding=material_embedding,
                top_k=ClassifierConfigV3.NODE_TOP_K,
                filter_fn=lambda id, meta: meta.get("tree_id") == tree_id and meta.get("type") == "node",
                min_similarity=ClassifierConfigV3.NODE_MIN_SIMILARITY,
            )

            logger.info(f"  向量检索到 {len(node_candidates)} 个候选节点")

            if node_candidates:
                # LLM 确认节点匹配
                nodes_desc = []
                for nc in node_candidates:
                    node_id = nc["metadata"]["node_id"]
                    node = next((n for n in tree_nodes if n.id == node_id), None)
                    if node:
                        desc = f"- ID: {node.id}\n  名称: {node.name}\n  描述: {node.description or '无'}\n  关键概念: {', '.join(node.key_concepts) or '无'}\n  向量相似度: {nc['similarity']:.2f}"
                        nodes_desc.append(desc)

                match_result, stats = await self._match_nodes(
                    tree_name=tree_name,
                    nodes_description="\n".join(nodes_desc),
                    keywords=result.keywords,
                    summary=result.summary,
                )
                self.stats_list.append(stats)
                total_tokens += stats.get("tokens_used", 0)

                best_node_id = match_result.best_match_node_id
            else:
                best_node_id = None

            # ========== Step 4: 最终决策 ==========
            if best_node_id:
                target_node_data = next((n for n in tree_nodes if n.id == best_node_id), None)
                if target_node_data:
                    decision, stats = await self._make_final_decision(
                        target_node=target_node_data,
                        tree_nodes=tree_nodes,
                        keywords=result.keywords,
                        summary=result.summary,
                        content_type=result.content_type,
                        materials_by_node=materials_by_node,
                    )
                    self.stats_list.append(stats)
                    total_tokens += stats.get("tokens_used", 0)
                    result.tree_node_decisions[tree_id] = decision
                else:
                    result.tree_node_decisions[tree_id] = self._default_decision(result.keywords)
            else:
                logger.info(f"  在树「{tree_name}」中没有找到匹配节点，建议创建根节点")
                result.tree_node_decisions[tree_id] = self._default_decision(result.keywords)

        # 统计信息
        result.total_api_calls = len(self.stats_list)
        result.total_tokens_used = total_tokens
        result.processing_time_ms = int((time.time() - start_time) * 1000)

        # 额外记录 embedding 调用次数
        logger.info(f"完成分类: {result.total_api_calls} 次 API 调用, {total_tokens} tokens, {embedding_calls} 次 embedding")

        return result

    async def _extract_keywords(self, content: str, content_length: int) -> tuple[KeywordExtractionOutput, dict]:
        """提取关键词"""
        truncated_content = content[:8000] if len(content) > 8000 else content

        prompt = KEYWORD_EXTRACTION_PROMPT.format_messages(content=truncated_content)

        start_time = time.time()
        response = await self.llm.ainvoke(prompt)
        elapsed_ms = int((time.time() - start_time) * 1000)

        try:
            content_str = response.content
            if "```json" in content_str:
                content_str = content_str.split("```json")[1].split("```")[0]
            elif "```" in content_str:
                content_str = content_str.split("```")[1].split("```")[0]
            parsed = json.loads(content_str.strip())
            result = KeywordExtractionOutput(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"关键词提取解析失败: {e}")
            result = KeywordExtractionOutput(
                keywords=content[:500].split()[:10],
                key_sentences=[],
                domain="未知",
                content_type="mixed",
                summary=content[:100],
                detailed_description=content[:300]
            )

        stats = {
            "step": "keyword_extraction",
            "elapsed_ms": elapsed_ms,
            "tokens_used": response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        }

        return result, stats

    async def _confirm_trees(
        self,
        trees_description: str,
        keywords: list[str],
        domain: str,
        summary: str,
    ) -> tuple[TreeSelectionOutput, dict]:
        """确认树匹配"""
        prompt = TREE_CONFIRM_PROMPT.format_messages(
            trees_description=trees_description,
            keywords=", ".join(keywords[:15]),
            domain=domain,
            summary=summary,
        )

        start_time = time.time()
        response = await self.llm.ainvoke(prompt)
        elapsed_ms = int((time.time() - start_time) * 1000)

        try:
            content_str = response.content
            if "```json" in content_str:
                content_str = content_str.split("```json")[1].split("```")[0]
            elif "```" in content_str:
                content_str = content_str.split("```")[1].split("```")[0]
            parsed = json.loads(content_str.strip())
            result = TreeSelectionOutput(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"树确认解析失败: {e}")
            result = TreeSelectionOutput(
                matched_trees=[],
                suggest_new_tree=True,
                suggested_tree_name=domain
            )

        stats = {
            "step": "tree_confirmation",
            "elapsed_ms": elapsed_ms,
            "tokens_used": response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        }

        return result, stats

    async def _match_nodes(
        self,
        tree_name: str,
        nodes_description: str,
        keywords: list[str],
        summary: str,
    ) -> tuple[NodeMatchOutput, dict]:
        """匹配节点"""
        prompt = NODE_MATCH_PROMPT.format_messages(
            tree_name=tree_name,
            nodes_description=nodes_description,
            keywords=", ".join(keywords[:15]),
            summary=summary,
        )

        start_time = time.time()
        response = await self.llm.ainvoke(prompt)
        elapsed_ms = int((time.time() - start_time) * 1000)

        try:
            content_str = response.content
            if "```json" in content_str:
                content_str = content_str.split("```json")[1].split("```")[0]
            elif "```" in content_str:
                content_str = content_str.split("```")[1].split("```")[0]
            parsed = json.loads(content_str.strip())
            result = NodeMatchOutput(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"节点匹配解析失败: {e}")
            result = NodeMatchOutput(
                best_match_node_id=None,
                best_match_node_name=None,
                confidence=0.0,
                needs_deeper_search=False,
                reason="解析失败"
            )

        stats = {
            "step": "node_matching",
            "elapsed_ms": elapsed_ms,
            "tokens_used": response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        }

        return result, stats

    async def _make_final_decision(
        self,
        target_node: MockNode,
        tree_nodes: list[MockNode],
        keywords: list[str],
        summary: str,
        content_type: str,
        materials_by_node: dict[str, list[str]],
    ) -> tuple[FinalDecisionOutput, dict]:
        """最终决策"""
        # 构建节点层级关系
        nodes_by_parent: dict[Optional[str], list[MockNode]] = {}
        for node in tree_nodes:
            parent_id = node.parent_id
            if parent_id not in nodes_by_parent:
                nodes_by_parent[parent_id] = []
            nodes_by_parent[parent_id].append(node)

        # 获取同级节点
        sibling_nodes = nodes_by_parent.get(target_node.parent_id, [])
        sibling_desc = []
        for node in sibling_nodes[:10]:
            if node.id != target_node.id:
                sibling_desc.append(f"- {node.name}: {node.description or '无描述'}")
        sibling_nodes_description = "\n".join(sibling_desc) if sibling_desc else "无同级节点"

        existing_mats = materials_by_node.get(target_node.id, [])

        prompt = FINAL_DECISION_PROMPT.format_messages(
            target_node_id=target_node.id,
            target_node_name=target_node.name,
            target_node_description=target_node.description or "无",
            target_node_concepts=", ".join(target_node.key_concepts) if target_node.key_concepts else "无",
            existing_materials=", ".join(existing_mats) if existing_mats else "暂无资料",
            sibling_nodes_description=sibling_nodes_description,
            keywords=", ".join(keywords[:15]),
            summary=summary,
            content_type=content_type,
        )

        start_time = time.time()
        response = await self.llm.ainvoke(prompt)
        elapsed_ms = int((time.time() - start_time) * 1000)

        try:
            content_str = response.content
            if "```json" in content_str:
                content_str = content_str.split("```json")[1].split("```")[0]
            elif "```" in content_str:
                content_str = content_str.split("```")[1].split("```")[0]
            parsed = json.loads(content_str.strip())
            result = FinalDecisionOutput(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"最终决策解析失败: {e}")
            result = FinalDecisionOutput(
                action="link_to_node",
                target_node_id=target_node.id,
                confidence=0.5,
                reason="解析失败，默认关联到目标节点",
                extracted_concepts=[]
            )

        stats = {
            "step": "final_decision",
            "elapsed_ms": elapsed_ms,
            "tokens_used": response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        }

        return result, stats

    def _default_decision(self, keywords: list[str]) -> FinalDecisionOutput:
        """返回默认决策"""
        return FinalDecisionOutput(
            action="create_child_node",
            suggested_node_name=keywords[0] if keywords else "新知识点",
            suggested_parent_id=None,
            confidence=0.5,
            reason="没有找到合适的匹配节点，建议创建新的根节点",
            extracted_concepts=keywords[:5]
        )

    def get_stats(self) -> list[dict]:
        """获取统计信息"""
        return self.stats_list
