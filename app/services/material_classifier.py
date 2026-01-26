"""
智能资料分类服务 - 使用 LangChain + Pydantic 实现分层匹配

优化版本 v2:
- 强制关键词提取（包括短文本）
- 加入详细描述段落
- 并行处理多棵树
- 缓存树结构摘要
"""

import json
import time
import asyncio
import logging
import hashlib
from typing import Optional
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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

logger = logging.getLogger(__name__)


# ============== 缓存管理 ==============

class TreeSummaryCache:
    """树结构摘要缓存"""
    _cache: dict[str, tuple[TreeSummary, float]] = {}
    _ttl_seconds: float = 300  # 5分钟缓存

    @classmethod
    def get(cls, tree_id: str, nodes_hash: str) -> Optional[TreeSummary]:
        """获取缓存的树摘要"""
        cache_key = f"{tree_id}:{nodes_hash}"
        if cache_key in cls._cache:
            summary, timestamp = cls._cache[cache_key]
            if time.time() - timestamp < cls._ttl_seconds:
                return summary
            else:
                del cls._cache[cache_key]
        return None

    @classmethod
    def set(cls, tree_id: str, nodes_hash: str, summary: TreeSummary):
        """设置缓存"""
        cache_key = f"{tree_id}:{nodes_hash}"
        cls._cache[cache_key] = (summary, time.time())

    @classmethod
    def clear(cls):
        """清除所有缓存"""
        cls._cache.clear()


def compute_nodes_hash(nodes: list) -> str:
    """计算节点列表的哈希值，用于缓存失效判断"""
    content = json.dumps([n.id for n in nodes], sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:8]


# ============== 配置 ==============

class ClassifierConfig:
    """分类器配置"""
    # 关键词提取阈值
    KEYWORD_EXTRACT_MIN_LENGTH = 1000
    KEYWORD_COUNT_SHORT = 8   # 短文 1000-3000 字
    KEYWORD_COUNT_MEDIUM = 14  # 中文 3000-6000 字
    KEYWORD_COUNT_LONG = 20   # 长文 6000+ 字

    # 分层匹配阈值
    DIRECT_MATCH_MAX_NODES = 100  # 少于此数直接全量匹配
    MAX_SEARCH_DEPTH = 5          # 最大搜索深度
    BATCH_SIZE = 20               # 每层最多传多少节点

    # 置信度阈值
    TREE_MATCH_THRESHOLD = 0.5    # 树匹配阈值
    NODE_MATCH_THRESHOLD = 0.3    # 节点匹配阈值
    DEEPER_SEARCH_THRESHOLD = 0.6 # 继续深搜的阈值


# ============== LLM 客户端 ==============

def create_llm_client(
    api_key: str,
    base_url: str = "https://api.deepseek.com/v1",
    model: str = "deepseek-chat",
    temperature: float = 0.3,
) -> ChatOpenAI:
    """创建 LangChain 的 ChatOpenAI 客户端（兼容 DeepSeek）"""
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=2048,
    )


# ============== Step 0: 关键词提取 ==============

KEYWORD_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的文本分析助手。你的任务是从学习资料中提取关键信息。

请严格按照以下 JSON 格式输出，不要输出其他内容：

{{
    "keywords": ["关键词1", "关键词2", ...],  // {keyword_count}个关键词，优先提取专业术语、概念名称
    "key_sentences": ["关键句1", "关键句2", ...],  // 3-5个最能代表内容的句子
    "domain": "学科领域",  // 如：数学、编程、历史、物理、机器学习、数据库等，要尽可能具体
    "content_type": "内容类型",  // concept/case/formula/procedure/mixed
    "summary": "100字以内的摘要",
    "detailed_description": "200-300字的详细描述"
}}

注意：
1. keywords 要提取最能代表内容的专业术语、核心概念、技术名词
2. domain 要尽可能具体（如"机器学习-聚类算法"而不是"计算机"）
3. content_type 根据内容主要形式判断
4. detailed_description 要详细说明：
   - 资料的核心主题是什么
   - 包含哪些主要知识点
   - 适合什么场景/人群学习
   - 与相关知识的联系"""),
    ("user", "请分析以下资料并提取关键信息：\n\n{content}")
])


async def extract_keywords(
    llm: ChatOpenAI,
    content: str,
    content_length: int,
) -> tuple[KeywordExtractionOutput, dict]:
    """
    Step 0: 提取关键词

    Returns:
        tuple: (提取结果, 统计信息)
    """
    # 确定关键词数量
    if content_length < 3000:
        keyword_count = ClassifierConfig.KEYWORD_COUNT_SHORT
    elif content_length < 6000:
        keyword_count = ClassifierConfig.KEYWORD_COUNT_MEDIUM
    else:
        keyword_count = ClassifierConfig.KEYWORD_COUNT_LONG

    # 截取内容（避免太长）
    truncated_content = content[:8000] if len(content) > 8000 else content

    prompt = KEYWORD_EXTRACTION_PROMPT.format_messages(
        keyword_count=keyword_count,
        content=truncated_content
    )

    start_time = time.time()
    response = await llm.ainvoke(prompt)
    elapsed_ms = int((time.time() - start_time) * 1000)

    # 解析响应
    try:
        content_str = response.content
        # 清理 markdown 代码块
        if "```json" in content_str:
            content_str = content_str.split("```json")[1].split("```")[0]
        elif "```" in content_str:
            content_str = content_str.split("```")[1].split("```")[0]

        parsed = json.loads(content_str.strip())
        result = KeywordExtractionOutput(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"关键词提取解析失败: {e}")
        # 返回默认值
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


# ============== Step 1: 树选择 ==============

TREE_SELECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识管理助手。你的任务是判断一份资料应该归属于哪些知识树。

用户有以下知识树：
{trees_description}

请根据资料的关键词、摘要和详细描述，判断它应该放到哪些树中。

输出 JSON 格式：
{{
    "matched_trees": [
        {{
            "tree_id": "树的ID",
            "tree_name": "树的名称",
            "confidence": 0.0-1.0,  // 匹配置信度
            "reason": "匹配理由（要具体说明与树的哪些概念相关）"
        }}
    ],
    "suggest_new_tree": false,  // 是否建议创建新树
    "suggested_tree_name": null  // 如果建议创建新树，给出名称
}}

规则：
1. confidence > 0.5 的树都算匹配
2. 如果资料涉及多个领域，可以返回多棵树
3. 如果没有任何树匹配（所有 confidence < 0.3），建议创建新树
4. 只输出 JSON，不要其他文字"""),
    ("user", """资料信息：
关键词：{keywords}
领域：{domain}
摘要：{summary}
详细描述：{detailed_description}

请判断应该归属于哪些知识树。""")
])


async def select_trees(
    llm: ChatOpenAI,
    keywords: list[str],
    summary: str,
    detailed_description: str,
    domain: str,
    available_trees: list[TreeSummary],
) -> tuple[TreeSelectionOutput, dict]:
    """
    Step 1: 选择目标树
    """
    # 构建树描述
    trees_desc = []
    for tree in available_trees:
        desc = f"- ID: {tree.tree_id}\n  名称: {tree.tree_name}\n  范围: {', '.join(tree.scope) or '未定义'}\n  关键概念: {', '.join(tree.root_concepts) or '未定义'}\n  节点数: {tree.total_nodes}"
        trees_desc.append(desc)

    trees_description = "\n".join(trees_desc) if trees_desc else "暂无知识树"

    prompt = TREE_SELECTION_PROMPT.format_messages(
        trees_description=trees_description,
        keywords=", ".join(keywords[:15]),
        domain=domain,
        summary=summary,
        detailed_description=detailed_description[:500]  # 限制长度
    )

    start_time = time.time()
    response = await llm.ainvoke(prompt)
    elapsed_ms = int((time.time() - start_time) * 1000)

    # 解析响应
    try:
        content_str = response.content
        if "```json" in content_str:
            content_str = content_str.split("```json")[1].split("```")[0]
        elif "```" in content_str:
            content_str = content_str.split("```")[1].split("```")[0]

        parsed = json.loads(content_str.strip())
        result = TreeSelectionOutput(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"树选择解析失败: {e}")
        result = TreeSelectionOutput(
            matched_trees=[],
            suggest_new_tree=True,
            suggested_tree_name=domain or keywords[0] if keywords else "新知识领域"
        )

    stats = {
        "step": "tree_selection",
        "elapsed_ms": elapsed_ms,
        "tokens_used": response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
    }

    return result, stats


# ============== Step 2: 节点匹配 ==============

NODE_MATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识管理助手。你的任务是在知识树「{tree_name}」中找到最适合放置资料的节点。

当前候选节点（深度 {current_depth}）：
{nodes_description}

请根据资料的关键词和摘要，找到最匹配的节点。

输出 JSON 格式：
{{
    "best_match_node_id": "节点ID 或 null",
    "best_match_node_name": "节点名称 或 null",
    "confidence": 0.0-1.0,
    "needs_deeper_search": true/false,  // 是否需要继续往下搜索该节点的子节点
    "reason": "判断理由"
}}

规则：
1. 找语义最匹配的节点
2. 如果最佳匹配节点有子节点，且资料可能属于更细的分类，设置 needs_deeper_search=true
3. 如果没有合适的匹配（confidence < 0.3），best_match_node_id 设为 null
4. 只输出 JSON"""),
    ("user", """资料信息：
关键词：{keywords}
摘要：{summary}

请找到最匹配的节点。""")
])


async def match_nodes(
    llm: ChatOpenAI,
    keywords: list[str],
    summary: str,
    tree_name: str,
    candidate_nodes: list[NodeSummary],
    current_depth: int,
) -> tuple[NodeMatchOutput, dict]:
    """
    Step 2: 在候选节点中找最匹配的
    """
    # 构建节点描述
    nodes_desc = []
    for node in candidate_nodes[:ClassifierConfig.BATCH_SIZE]:
        desc = f"- ID: {node.node_id}\n  名称: {node.node_name}\n  描述: {node.description or '无'}\n  关键概念: {', '.join(node.key_concepts) or '无'}\n  子节点数: {node.children_count}"
        nodes_desc.append(desc)

    nodes_description = "\n".join(nodes_desc) if nodes_desc else "暂无节点"

    prompt = NODE_MATCH_PROMPT.format_messages(
        tree_name=tree_name,
        current_depth=current_depth,
        nodes_description=nodes_description,
        keywords=", ".join(keywords[:15]),
        summary=summary
    )

    start_time = time.time()
    response = await llm.ainvoke(prompt)
    elapsed_ms = int((time.time() - start_time) * 1000)

    # 解析响应
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
        "step": f"node_match_depth_{current_depth}",
        "elapsed_ms": elapsed_ms,
        "tokens_used": response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
    }

    return result, stats


# ============== Step 3: 最终决策 ==============

FINAL_DECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个知识管理助手。现在已经找到了最可能的目标节点，请做最终决策。

目标节点信息：
- ID: {target_node_id}
- 名称: {target_node_name}
- 描述: {target_node_description}
- 关键概念: {target_node_concepts}
- 已有资料: {existing_materials}

同级节点（用于判断是否需要新建同级节点）：
{sibling_nodes_description}

请决定如何处理这份资料。

输出 JSON 格式：
{{
    "action": "link_to_node|create_child_node|create_sibling_node|update_existing",
    "target_node_id": "节点ID（link_to_node/update_existing 时填）",
    "suggested_node_name": "新节点名称（create_* 时填）",
    "suggested_parent_id": "父节点ID（create_child_node 时填）",
    "confidence": 0.0-1.0,
    "reason": "决策理由",
    "extracted_concepts": ["从资料提取的新概念1", "概念2"]
}}

决策规则：
1. link_to_node: 资料内容与目标节点完全匹配，直接关联
2. create_child_node: 资料是目标节点的子知识点，在其下创建新节点
3. create_sibling_node: 资料是同级别的新知识点，创建同级节点
4. update_existing: 资料是对已有资料的补充/更新

只输出 JSON。"""),
    ("user", """资料信息：
关键词：{keywords}
摘要：{summary}
内容类型：{content_type}

请做出最终决策。""")
])


async def make_final_decision(
    llm: ChatOpenAI,
    keywords: list[str],
    summary: str,
    content_type: str,
    target_node: NodeSummary,
    sibling_nodes: list[NodeSummary],
    existing_materials: list[str],
) -> tuple[FinalDecisionOutput, dict]:
    """
    Step 3: 做出最终决策
    """
    # 构建同级节点描述
    sibling_desc = []
    for node in sibling_nodes[:10]:
        sibling_desc.append(f"- {node.node_name}: {node.description or '无描述'}")
    sibling_nodes_description = "\n".join(sibling_desc) if sibling_desc else "无同级节点"

    prompt = FINAL_DECISION_PROMPT.format_messages(
        target_node_id=target_node.node_id,
        target_node_name=target_node.node_name,
        target_node_description=target_node.description or "无",
        target_node_concepts=", ".join(target_node.key_concepts) or "无",
        existing_materials=", ".join(existing_materials) if existing_materials else "暂无资料",
        sibling_nodes_description=sibling_nodes_description,
        keywords=", ".join(keywords[:15]),
        summary=summary,
        content_type=content_type
    )

    start_time = time.time()
    response = await llm.ainvoke(prompt)
    elapsed_ms = int((time.time() - start_time) * 1000)

    # 解析响应
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
            target_node_id=target_node.node_id,
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


# ============== 主流程：智能分类 ==============

class MaterialClassifier:
    """智能资料分类器"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
    ):
        self.llm = create_llm_client(api_key, base_url, model)
        self.stats_list: list[dict] = []

    async def classify(
        self,
        content: str,
        trees: list[MockTree],
        nodes_by_tree: dict[str, list[MockNode]],  # tree_id -> nodes
        materials_by_node: dict[str, list[str]],   # node_id -> material names
    ) -> MaterialClassificationResult:
        """
        执行完整的分类流程

        Args:
            content: 资料内容
            trees: 用户的知识树列表
            nodes_by_tree: 每棵树的节点列表
            materials_by_node: 每个节点的资料名称列表

        Returns:
            MaterialClassificationResult: 分类结果
        """
        start_time = time.time()
        self.stats_list = []
        total_tokens = 0

        result = MaterialClassificationResult()
        content_length = len(content)

        # ========== Step 0: 关键词提取（强制执行，无论内容长度）==========
        logger.info(f"Step 0: 提取关键词和详细描述（内容长度: {content_length}）")
        keyword_result, stats = await extract_keywords(self.llm, content, content_length)
        self.stats_list.append(stats)
        total_tokens += stats.get("tokens_used", 0)

        result.keywords = keyword_result.keywords
        result.summary = keyword_result.summary
        result.detailed_description = keyword_result.detailed_description
        result.domain = keyword_result.domain
        result.content_type = keyword_result.content_type

        # ========== Step 1: 树选择 ==========
        logger.info(f"Step 1: 选择目标树（共 {len(trees)} 棵树）")

        tree_summaries = []
        for tree in trees:
            tree_nodes = nodes_by_tree.get(tree.id, [])
            root_nodes = [n for n in tree_nodes if n.parent_id is None]
            root_concepts = []
            for rn in root_nodes[:5]:
                root_concepts.extend(rn.key_concepts[:3])

            tree_summaries.append(TreeSummary(
                tree_id=tree.id,
                tree_name=tree.name,
                scope=tree.scope,
                root_concepts=root_concepts[:10],
                total_nodes=len(tree_nodes),
                sample_node_names=[n.name for n in tree_nodes[:5]]
            ))

        tree_selection, stats = await select_trees(
            self.llm,
            result.keywords,
            result.summary,
            result.detailed_description,
            result.domain,
            tree_summaries
        )
        self.stats_list.append(stats)
        total_tokens += stats.get("tokens_used", 0)

        result.matched_trees = tree_selection.matched_trees
        result.suggest_new_tree = tree_selection.suggest_new_tree
        result.suggested_tree_name = tree_selection.suggested_tree_name

        # 过滤低置信度的树
        matched_trees = [t for t in tree_selection.matched_trees if t.confidence >= ClassifierConfig.TREE_MATCH_THRESHOLD]

        if not matched_trees:
            logger.info("没有匹配的树，建议创建新树")
            result.suggest_new_tree = True
            result.suggested_tree_name = result.domain or result.keywords[0] if result.keywords else "新知识领域"

        # ========== Step 2 & 3: 对每棵匹配的树进行节点定位（并行处理）==========
        if matched_trees:
            logger.info(f"开始并行处理 {len(matched_trees)} 棵匹配的树")

            # 创建并行任务
            tasks = [
                self._process_single_tree(
                    tree_match=tree_match,
                    tree_nodes=nodes_by_tree.get(tree_match.tree_id, []),
                    keywords=result.keywords,
                    summary=result.summary,
                    content_type=result.content_type,
                    materials_by_node=materials_by_node,
                )
                for tree_match in matched_trees
            ]

            # 并行执行
            tree_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 收集结果
            for tree_match, tree_result in zip(matched_trees, tree_results):
                if isinstance(tree_result, Exception):
                    logger.error(f"处理树「{tree_match.tree_name}」时出错: {tree_result}")
                    continue

                tree_id, decision, stats_list, tokens_used = tree_result
                result.tree_node_decisions[tree_id] = decision
                self.stats_list.extend(stats_list)
                total_tokens += tokens_used

        # 统计信息
        result.total_api_calls = len(self.stats_list)
        result.total_tokens_used = total_tokens
        result.processing_time_ms = int((time.time() - start_time) * 1000)

        return result

    async def _hierarchical_match(
        self,
        tree_name: str,
        nodes_by_parent: dict[Optional[str], list[MockNode]],
        keywords: list[str],
        summary: str,
    ) -> Optional[str]:
        """分层匹配找到最佳节点"""
        current_parent_id: Optional[str] = None
        best_node_id: Optional[str] = None

        for depth in range(ClassifierConfig.MAX_SEARCH_DEPTH):
            children = nodes_by_parent.get(current_parent_id, [])
            if not children:
                break

            logger.info(f"    深度 {depth}: 匹配 {len(children)} 个候选节点")

            candidate_summaries = [
                NodeSummary(
                    node_id=n.id,
                    node_name=n.name,
                    description=n.description,
                    key_concepts=n.key_concepts,
                    children_count=len(nodes_by_parent.get(n.id, [])),
                    depth=depth
                )
                for n in children[:ClassifierConfig.BATCH_SIZE]
            ]

            match_result, stats = await match_nodes(
                self.llm,
                keywords,
                summary,
                tree_name,
                candidate_summaries,
                depth
            )
            self.stats_list.append(stats)

            if match_result.best_match_node_id:
                best_node_id = match_result.best_match_node_id

                if match_result.needs_deeper_search and match_result.confidence >= ClassifierConfig.DEEPER_SEARCH_THRESHOLD:
                    # 继续往下搜索
                    current_parent_id = best_node_id
                    logger.info(f"    → 继续搜索节点「{match_result.best_match_node_name}」的子节点")
                else:
                    # 找到了
                    logger.info(f"    → 找到最佳匹配: {match_result.best_match_node_name} (置信度: {match_result.confidence})")
                    break
            else:
                # 没找到匹配的
                logger.info(f"    → 当前层没有找到匹配")
                break

        return best_node_id

    async def _process_single_tree(
        self,
        tree_match: TreeMatch,
        tree_nodes: list[MockNode],
        keywords: list[str],
        summary: str,
        content_type: str,
        materials_by_node: dict[str, list[str]],
    ) -> tuple[str, FinalDecisionOutput, list[dict], int]:
        """
        处理单棵树的节点匹配和最终决策

        Returns:
            tuple: (tree_id, decision, stats_list, tokens_used)
        """
        tree_id = tree_match.tree_id
        tree_name = tree_match.tree_name
        stats_list = []
        total_tokens = 0

        logger.info(f"Step 2: 在树「{tree_name}」中定位（共 {len(tree_nodes)} 个节点）")

        # 构建节点层级关系
        nodes_by_parent: dict[Optional[str], list[MockNode]] = {}
        for node in tree_nodes:
            parent_id = node.parent_id
            if parent_id not in nodes_by_parent:
                nodes_by_parent[parent_id] = []
            nodes_by_parent[parent_id].append(node)

        # 判断是否需要分层匹配
        if len(tree_nodes) < ClassifierConfig.DIRECT_MATCH_MAX_NODES:
            # 直接全量匹配
            logger.info(f"  节点数 < {ClassifierConfig.DIRECT_MATCH_MAX_NODES}，直接全量匹配")
            candidate_summaries = [
                NodeSummary(
                    node_id=n.id,
                    node_name=n.name,
                    description=n.description,
                    key_concepts=n.key_concepts,
                    children_count=len(nodes_by_parent.get(n.id, [])),
                    depth=0
                )
                for n in tree_nodes
            ]

            match_result, stats = await match_nodes(
                self.llm,
                keywords,
                summary,
                tree_name,
                candidate_summaries,
                0
            )
            stats_list.append(stats)
            total_tokens += stats.get("tokens_used", 0)

            best_node_id = match_result.best_match_node_id
        else:
            # 分层匹配
            logger.info(f"  节点数 >= {ClassifierConfig.DIRECT_MATCH_MAX_NODES}，启动分层匹配")
            # 注意：分层匹配暂时不能并行，因为需要依赖 self.stats_list
            # 这里创建临时的 stats 列表
            temp_stats_list = []
            best_node_id = await self._hierarchical_match_with_stats(
                tree_name,
                nodes_by_parent,
                keywords,
                summary,
                temp_stats_list,
            )
            stats_list.extend(temp_stats_list)
            total_tokens += sum(s.get("tokens_used", 0) for s in temp_stats_list)

        # ========== Step 3: 最终决策 ==========
        if best_node_id:
            target_node_data = next((n for n in tree_nodes if n.id == best_node_id), None)
            if target_node_data:
                target_node = NodeSummary(
                    node_id=target_node_data.id,
                    node_name=target_node_data.name,
                    description=target_node_data.description,
                    key_concepts=target_node_data.key_concepts,
                    children_count=len(nodes_by_parent.get(target_node_data.id, [])),
                    depth=0
                )

                # 获取同级节点
                sibling_nodes_data = nodes_by_parent.get(target_node_data.parent_id, [])
                sibling_summaries = [
                    NodeSummary(
                        node_id=n.id,
                        node_name=n.name,
                        description=n.description,
                        key_concepts=n.key_concepts,
                        children_count=0,
                        depth=0
                    )
                    for n in sibling_nodes_data if n.id != best_node_id
                ]

                existing_mats = materials_by_node.get(best_node_id, [])

                logger.info(f"Step 3: 最终决策（目标节点: {target_node.node_name}）")

                decision, stats = await make_final_decision(
                    self.llm,
                    keywords,
                    summary,
                    content_type,
                    target_node,
                    sibling_summaries,
                    existing_mats
                )
                stats_list.append(stats)
                total_tokens += stats.get("tokens_used", 0)

                return tree_id, decision, stats_list, total_tokens

        # 没找到匹配节点，建议创建根节点
        logger.info(f"  在树「{tree_name}」中没有找到匹配节点，建议创建根节点")
        default_decision = FinalDecisionOutput(
            action="create_child_node",
            suggested_node_name=keywords[0] if keywords else "新知识点",
            suggested_parent_id=None,  # 作为根节点
            confidence=0.5,
            reason="没有找到合适的匹配节点，建议创建新的根节点",
            extracted_concepts=keywords[:5]
        )
        return tree_id, default_decision, stats_list, total_tokens

    async def _hierarchical_match_with_stats(
        self,
        tree_name: str,
        nodes_by_parent: dict[Optional[str], list[MockNode]],
        keywords: list[str],
        summary: str,
        stats_list: list[dict],
    ) -> Optional[str]:
        """分层匹配找到最佳节点（带统计信息收集）"""
        current_parent_id: Optional[str] = None
        best_node_id: Optional[str] = None

        for depth in range(ClassifierConfig.MAX_SEARCH_DEPTH):
            children = nodes_by_parent.get(current_parent_id, [])
            if not children:
                break

            logger.info(f"    深度 {depth}: 匹配 {len(children)} 个候选节点")

            candidate_summaries = [
                NodeSummary(
                    node_id=n.id,
                    node_name=n.name,
                    description=n.description,
                    key_concepts=n.key_concepts,
                    children_count=len(nodes_by_parent.get(n.id, [])),
                    depth=depth
                )
                for n in children[:ClassifierConfig.BATCH_SIZE]
            ]

            match_result, stats = await match_nodes(
                self.llm,
                keywords,
                summary,
                tree_name,
                candidate_summaries,
                depth
            )
            stats_list.append(stats)

            if match_result.best_match_node_id:
                best_node_id = match_result.best_match_node_id

                if match_result.needs_deeper_search and match_result.confidence >= ClassifierConfig.DEEPER_SEARCH_THRESHOLD:
                    # 继续往下搜索
                    current_parent_id = best_node_id
                    logger.info(f"    → 继续搜索节点「{match_result.best_match_node_name}」的子节点")
                else:
                    # 找到了
                    logger.info(f"    → 找到最佳匹配: {match_result.best_match_node_name} (置信度: {match_result.confidence})")
                    break
            else:
                # 没找到匹配的
                logger.info(f"    → 当前层没有找到匹配")
                break

        return best_node_id

    def get_stats(self) -> list[dict]:
        """获取所有步骤的统计信息"""
        return self.stats_list
