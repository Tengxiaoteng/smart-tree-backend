"""
Pydantic 结构化输出 Schema 定义
用于多 Agent 协同的标准化输入输出
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TaskDifficulty(str, Enum):
    """任务难度等级"""
    SIMPLE = "simple"       # 简单：直接回答，无需复杂推理
    MEDIUM = "medium"       # 中等：需要一定推理或结构化输出
    COMPLEX = "complex"     # 复杂：需要多步推理、专业知识


class TaskType(str, Enum):
    """任务类型"""
    TEXT_ANALYSIS = "text_analysis"       # 文本分析
    IMAGE_ANALYSIS = "image_analysis"     # 图片分析
    TREE_GENERATION = "tree_generation"   # 知识树生成
    QA = "qa"                             # 问答
    SUMMARY = "summary"                   # 摘要


class IntentResult(BaseModel):
    """意图识别结果（Agent 1 输出）"""
    task_type: TaskType = Field(..., description="识别出的任务类型")
    difficulty: TaskDifficulty = Field(..., description="任务难度等级")
    recommended_model: str = Field(..., description="推荐使用的模型")
    reasoning: str = Field(..., description="判断理由")

    class Config:
        use_enum_values = True


class KnowledgeNode(BaseModel):
    """知识节点"""
    id: str = Field(..., description="节点唯一标识")
    name: str = Field(..., description="节点名称")
    description: Optional[str] = Field(None, description="节点描述")
    parent_id: Optional[str] = Field(None, description="父节点ID")
    order: int = Field(0, description="排序顺序")


class KnowledgeTree(BaseModel):
    """知识树结构（Agent 3 输出）"""
    topic: str = Field(..., description="主题名称")
    summary: str = Field(..., description="主题摘要")
    nodes: list[KnowledgeNode] = Field(..., description="知识节点列表")
    concepts: list[str] = Field(default_factory=list, description="核心概念列表")


class ImageAnalysisResult(BaseModel):
    """图片分析结果（Agent 2 输出）"""
    content: str = Field(..., description="识别的内容（Markdown 格式）")
    summary: str = Field(..., description="一句话摘要")
    concepts: list[str] = Field(default_factory=list, description="提取的关键概念")
    has_math: bool = Field(False, description="是否包含数学公式")
    has_code: bool = Field(False, description="是否包含代码")


class MultiAgentResult(BaseModel):
    """多 Agent 协同处理的最终结果"""
    intent: IntentResult = Field(..., description="意图识别结果")
    processing_time_ms: float = Field(..., description="处理总耗时（毫秒）")
    model_used: str = Field(..., description="实际使用的模型")
    result: dict = Field(..., description="处理结果")
