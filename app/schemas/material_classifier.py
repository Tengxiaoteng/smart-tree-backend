"""
智能资料分类 - Pydantic Schema 定义
使用 LangChain + Pydantic 实现分层匹配
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ============== Step 0: 关键词提取 ==============

class KeywordExtractionInput(BaseModel):
    """关键词提取的输入"""
    content: str = Field(..., description="资料内容")
    content_length: int = Field(..., description="内容长度")


class KeywordExtractionOutput(BaseModel):
    """关键词提取的输出"""
    keywords: list[str] = Field(
        ...,
        description="提取的关键词，8-20个",
        min_length=5,
        max_length=25
    )
    key_sentences: list[str] = Field(
        default_factory=list,
        description="关键句子，3-5个",
        max_length=5
    )
    domain: str = Field(
        ...,
        description="推测的学科领域，如：数学、编程、历史、物理等"
    )
    content_type: str = Field(
        ...,
        description="内容类型：concept(概念)、case(案例)、formula(公式)、procedure(步骤)、mixed(混合)"
    )
    summary: str = Field(
        ...,
        description="100字以内的内容摘要"
    )
    detailed_description: str = Field(
        default="",
        description="200-300字的详细描述，说明资料的核心内容、主要知识点和适用场景"
    )


# ============== Step 1: 树选择 ==============

class TreeSummary(BaseModel):
    """知识树的摘要信息（传给AI判断用）"""
    tree_id: str = Field(..., description="树的ID")
    tree_name: str = Field(..., description="树的名称")
    scope: list[str] = Field(default_factory=list, description="树的范围/关键词")
    root_concepts: list[str] = Field(default_factory=list, description="根节点的关键概念")
    total_nodes: int = Field(default=0, description="节点总数")
    sample_node_names: list[str] = Field(default_factory=list, description="部分节点名称示例")


class TreeSelectionInput(BaseModel):
    """树选择的输入"""
    keywords: list[str] = Field(..., description="资料的关键词")
    summary: str = Field(..., description="资料摘要")
    domain: str = Field(..., description="资料领域")
    available_trees: list[TreeSummary] = Field(..., description="用户所有的知识树")


class TreeMatch(BaseModel):
    """单棵树的匹配结果"""
    tree_id: str = Field(..., description="树的ID")
    tree_name: str = Field(..., description="树的名称")
    confidence: float = Field(..., ge=0.0, le=1.0, description="匹配置信度 0-1")
    reason: str = Field(..., description="匹配理由")


class TreeSelectionOutput(BaseModel):
    """树选择的输出"""
    matched_trees: list[TreeMatch] = Field(
        default_factory=list,
        description="匹配的树列表，按置信度排序"
    )
    suggest_new_tree: bool = Field(
        default=False,
        description="是否建议创建新树"
    )
    suggested_tree_name: Optional[str] = Field(
        default=None,
        description="建议的新树名称（当 suggest_new_tree=True 时）"
    )


# ============== Step 2: 节点匹配（分层） ==============

class NodeSummary(BaseModel):
    """节点的摘要信息"""
    node_id: str = Field(..., description="节点ID")
    node_name: str = Field(..., description="节点名称")
    description: Optional[str] = Field(default=None, description="节点描述")
    key_concepts: list[str] = Field(default_factory=list, description="关键概念")
    children_count: int = Field(default=0, description="子节点数量")
    depth: int = Field(default=0, description="节点深度")


class NodeMatchInput(BaseModel):
    """节点匹配的输入"""
    keywords: list[str] = Field(..., description="资料的关键词")
    summary: str = Field(..., description="资料摘要")
    tree_name: str = Field(..., description="当前树的名称")
    candidate_nodes: list[NodeSummary] = Field(..., description="候选节点列表")
    current_depth: int = Field(default=0, description="当前搜索深度")


class NodeMatchOutput(BaseModel):
    """节点匹配的输出"""
    best_match_node_id: Optional[str] = Field(
        default=None,
        description="最佳匹配的节点ID"
    )
    best_match_node_name: Optional[str] = Field(
        default=None,
        description="最佳匹配的节点名称"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="匹配置信度"
    )
    needs_deeper_search: bool = Field(
        default=False,
        description="是否需要继续往下搜索子节点"
    )
    reason: str = Field(..., description="判断理由")


# ============== Step 3: 最终决策 ==============

class FinalDecisionInput(BaseModel):
    """最终决策的输入"""
    keywords: list[str] = Field(..., description="资料的关键词")
    summary: str = Field(..., description="资料摘要")
    content_type: str = Field(..., description="内容类型")
    target_node: NodeSummary = Field(..., description="目标节点信息")
    sibling_nodes: list[NodeSummary] = Field(
        default_factory=list,
        description="同级节点（用于判断是否需要新建）"
    )
    existing_materials: list[str] = Field(
        default_factory=list,
        description="目标节点已有的资料名称"
    )


class FinalDecisionOutput(BaseModel):
    """最终决策的输出"""
    action: Literal["link_to_node", "create_child_node", "create_sibling_node", "update_existing"] = Field(
        ...,
        description="操作类型"
    )
    target_node_id: Optional[str] = Field(
        default=None,
        description="目标节点ID（link_to_node/update_existing 时）"
    )
    suggested_node_name: Optional[str] = Field(
        default=None,
        description="建议的新节点名称（create_* 时）"
    )
    suggested_parent_id: Optional[str] = Field(
        default=None,
        description="建议的父节点ID（create_child_node 时）"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="决策置信度")
    reason: str = Field(..., description="决策理由")
    extracted_concepts: list[str] = Field(
        default_factory=list,
        description="从资料中提取的新概念"
    )


# ============== 完整分类结果 ==============

class MaterialClassificationResult(BaseModel):
    """完整的资料分类结果"""
    # 关键词提取结果
    keywords: list[str] = Field(default_factory=list)
    summary: str = Field(default="")
    detailed_description: str = Field(default="")
    domain: str = Field(default="")
    content_type: str = Field(default="")

    # 树匹配结果
    matched_trees: list[TreeMatch] = Field(default_factory=list)
    suggest_new_tree: bool = Field(default=False)
    suggested_tree_name: Optional[str] = Field(default=None)

    # 每棵树的节点定位结果
    tree_node_decisions: dict[str, FinalDecisionOutput] = Field(
        default_factory=dict,
        description="每棵树的最终决策，key 为 tree_id"
    )

    # 统计信息
    total_api_calls: int = Field(default=0, description="总 API 调用次数")
    total_tokens_used: int = Field(default=0, description="总 token 消耗")
    processing_time_ms: int = Field(default=0, description="处理时间（毫秒）")


# ============== 测试用的模拟数据 ==============

class MockTree(BaseModel):
    """模拟的知识树（用于测试）"""
    id: str
    name: str
    scope: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class MockNode(BaseModel):
    """模拟的知识节点（用于测试）"""
    id: str
    name: str
    parent_id: Optional[str] = None
    description: Optional[str] = None
    key_concepts: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)  # 资料名称列表
