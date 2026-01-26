"""
积分扣费配置 - 混合制方案
简单功能：固定积分
复杂功能：按 Token 计费（有上下限）
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_UP
from enum import Enum
from typing import Literal

from app.core.config import settings


class FeatureType(str, Enum):
    """AI 功能类型"""
    # 固定积分功能
    CHAT_SHORT = "chat_short"           # AI 对话（短）
    CHAT_MEDIUM = "chat_medium"         # AI 对话（中）
    CHAT_LONG = "chat_long"             # AI 对话（长）
    NODE_EXPAND = "node_expand"         # 节点展开
    MATERIAL_ORGANIZE = "material_organize"      # 资料整理
    MATERIAL_STRUCTURE = "material_structure"    # 资料结构化
    PORTRAIT_REFRESH = "portrait_refresh"        # 学习画像刷新

    # 按 Token 计费功能
    TREE_GENERATE = "tree_generate"     # 生成知识树
    QUESTION_GENERATE = "question_generate"  # 生成问题
    BATCH_TASK = "batch_task"           # 批量任务


@dataclass(frozen=True)
class FixedPricing:
    """固定积分定价"""
    points: int
    description: str


@dataclass(frozen=True)
class TokenPricing:
    """按 Token 计费定价"""
    min_points: int           # 最低积分
    max_points: int           # 最高积分
    base_points: int          # 基础积分
    points_per_1k_tokens: int # 每 1000 token 积分
    description: str


# 固定积分定价表
FIXED_PRICING: dict[FeatureType, FixedPricing] = {
    FeatureType.CHAT_SHORT: FixedPricing(
        points=2,
        description="AI 对话（输入<500字符）"
    ),
    FeatureType.CHAT_MEDIUM: FixedPricing(
        points=4,
        description="AI 对话（输入500-2000字符）"
    ),
    FeatureType.CHAT_LONG: FixedPricing(
        points=8,
        description="AI 对话（输入>2000字符）"
    ),
    FeatureType.NODE_EXPAND: FixedPricing(
        points=5,
        description="节点展开"
    ),
    FeatureType.MATERIAL_ORGANIZE: FixedPricing(
        points=5,
        description="资料整理"
    ),
    FeatureType.MATERIAL_STRUCTURE: FixedPricing(
        points=5,
        description="资料结构化"
    ),
    FeatureType.PORTRAIT_REFRESH: FixedPricing(
        points=5,
        description="学习画像刷新"
    ),
}


# 固定积分的复杂功能（简化计费）
FIXED_COMPLEX_PRICING: dict[FeatureType, FixedPricing] = {
    FeatureType.TREE_GENERATE: FixedPricing(
        points=20,
        description="生成知识树"
    ),
    FeatureType.QUESTION_GENERATE: FixedPricing(
        points=5,
        description="生成问题"
    ),
    FeatureType.NODE_EXPAND: FixedPricing(
        points=5,
        description="节点展开"
    ),
}

# 按 Token 计费定价表（保留用于需要精确计费的场景）
TOKEN_PRICING: dict[FeatureType, TokenPricing] = {
    FeatureType.BATCH_TASK: TokenPricing(
        min_points=10,
        max_points=200,
        base_points=5,
        points_per_1k_tokens=2,
        description="批量任务"
    ),
}


def get_chat_feature_type(input_length: int) -> FeatureType:
    """根据输入长度确定对话类型"""
    if input_length < 500:
        return FeatureType.CHAT_SHORT
    elif input_length < 2000:
        return FeatureType.CHAT_MEDIUM
    else:
        return FeatureType.CHAT_LONG


def calculate_fixed_points(feature: FeatureType) -> int:
    """计算固定积分功能的扣费"""
    # 先检查简单功能
    pricing = FIXED_PRICING.get(feature)
    if pricing:
        return pricing.points
    # 再检查复杂功能
    complex_pricing = FIXED_COMPLEX_PRICING.get(feature)
    if complex_pricing:
        return complex_pricing.points
    return 1  # 默认 1 积分


def calculate_token_points(
    feature: FeatureType,
    total_tokens: int,
    extra_count: int = 0,  # 额外计数（如问题数量）
    extra_points_per_item: int = 0,  # 每个额外项的积分
) -> int:
    """
    计算按 Token 计费功能的扣费

    Args:
        feature: 功能类型
        total_tokens: 总 token 数
        extra_count: 额外计数（如生成的问题数量）
        extra_points_per_item: 每个额外项的积分（如每题 2 分）

    Returns:
        应扣积分
    """
    pricing = TOKEN_PRICING.get(feature)
    if not pricing:
        # 回退到通用 token 计费
        return max(1, int(total_tokens / 500))

    # 基础积分 + Token 消耗积分
    token_points = int(Decimal(total_tokens) * Decimal(pricing.points_per_1k_tokens) / Decimal(1000))
    total_points = pricing.base_points + token_points

    # 加上额外项积分（如每道题的积分）
    if extra_count > 0 and extra_points_per_item > 0:
        total_points += extra_count * extra_points_per_item

    # 限制在最低和最高范围内
    return max(pricing.min_points, min(pricing.max_points, total_points))


def estimate_reserve_points(feature: FeatureType, estimated_tokens: int = 0) -> int:
    """
    预估预扣积分（用于 reserve）

    固定积分功能：直接返回固定值
    Token 计费功能：返回最高值的 80%（避免预扣过多）
    """
    if feature in FIXED_PRICING:
        return FIXED_PRICING[feature].points

    if feature in TOKEN_PRICING:
        pricing = TOKEN_PRICING[feature]
        # 预扣使用估算值，但不超过最高值的 80%
        if estimated_tokens > 0:
            estimated = calculate_token_points(feature, estimated_tokens)
            return min(estimated, int(pricing.max_points * 0.8))
        # 没有估算值时，使用中位数
        return (pricing.min_points + pricing.max_points) // 2

    return 5  # 默认预扣


def get_feature_description(feature: FeatureType) -> str:
    """获取功能描述"""
    if feature in FIXED_PRICING:
        return FIXED_PRICING[feature].description
    if feature in TOKEN_PRICING:
        return TOKEN_PRICING[feature].description
    return str(feature.value)
