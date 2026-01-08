#!/usr/bin/env python3
"""
多 Agent 架构测试脚本

测试场景：
1. 简单问答 - 验证意图识别选择 qwen-turbo
2. 中等复杂度任务 - 验证选择 qwen-plus
3. 复杂任务 - 验证选择 qwen-max
4. 知识树生成 - 验证动态模型选择
5. 对比原有单模型方案的耗时

使用方法：
    cd smart-tree-backend
    python -m tests.multi_agent.test_agents
"""
import asyncio
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv(".env.sqlite")
load_dotenv(".env")

from tests.multi_agent.agents import (
    MultiAgentOrchestrator,
    LLMClient,
    MODEL_CONFIG,
    TaskDifficulty,
    DIFFICULTY_MODEL_MAP,
)


# 测试用例
TEST_CASES = [
    {
        "name": "简单问答",
        "input": "什么是 Python？",
        "expected_difficulty": "simple",
        "description": "简单概念问答，应使用 qwen-turbo",
    },
    {
        "name": "中等复杂度",
        "input": "请解释机器学习中的梯度下降算法，并说明它是如何优化损失函数的",
        "expected_difficulty": "medium",
        "description": "需要一定专业知识和结构化解释",
    },
    {
        "name": "复杂任务",
        "input": "请详细分析 Transformer 架构的自注意力机制，包括 Query/Key/Value 的计算过程、多头注意力的作用，以及位置编码的必要性。同时比较它与 RNN/LSTM 在处理序列数据时的优缺点。",
        "expected_difficulty": "complex",
        "description": "深度技术分析，需要 qwen-max",
    },
    {
        "name": "知识树生成",
        "input": "请帮我生成一个关于「Python 数据分析」的知识树，包含 NumPy、Pandas、Matplotlib 等核心库",
        "expected_difficulty": "medium",
        "description": "知识树生成任务",
    },
    {
        "name": "学习资料整理",
        "input": "帮我整理以下学习内容：线性代数是机器学习的数学基础，主要包括向量、矩阵运算、特征值分解、SVD分解等内容。向量可以表示数据点，矩阵可以表示线性变换，特征值分解用于数据降维。",
        "expected_difficulty": "medium",
        "description": "内容整理成结构化知识",
    },
]


async def test_single_model_baseline(client: LLMClient, test_input: str, model: str) -> tuple[str, float]:
    """单模型基线测试"""
    messages = [
        {"role": "system", "content": "你是一个智能学习助手，请帮助用户解答问题。"},
        {"role": "user", "content": test_input},
    ]

    start = time.time()
    result = await client.chat(model, messages)
    elapsed = (time.time() - start) * 1000

    return result, elapsed


async def run_comparison_test(api_key: str, base_url: str):
    """运行对比测试"""
    print("=" * 60)
    print("多 Agent 架构 vs 单模型 对比测试")
    print("=" * 60)
    print()

    orchestrator = MultiAgentOrchestrator(api_key, base_url)
    client = LLMClient(api_key, base_url)

    results = []

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(TEST_CASES)}: {test['name']}")
        print(f"描述: {test['description']}")
        print(f"输入: {test['input'][:80]}...")
        print("-" * 60)

        # 1. 多 Agent 测试
        print("\n[多 Agent 模式]")
        multi_start = time.time()
        multi_result = await orchestrator.process(test["input"])
        multi_time = (time.time() - multi_start) * 1000

        print(f"  意图识别: {multi_result.intent.task_type}, 难度: {multi_result.intent.difficulty}")
        print(f"  推荐模型: {multi_result.intent.recommended_model}")
        print(f"  实际使用: {multi_result.model_used}")
        print(f"  判断理由: {multi_result.intent.reasoning}")
        print(f"  总耗时: {multi_time:.0f}ms")

        # 2. 单模型基线测试（使用 qwen-plus 作为默认）
        print("\n[单模型基线 - qwen-plus]")
        baseline_result, baseline_time = await test_single_model_baseline(
            client, test["input"], MODEL_CONFIG["standard"]
        )
        print(f"  耗时: {baseline_time:.0f}ms")

        # 3. 验证难度判断是否符合预期
        difficulty_match = multi_result.intent.difficulty == test["expected_difficulty"]
        print(f"\n[验证] 难度判断{'✓ 符合' if difficulty_match else '✗ 不符合'}预期 (期望: {test['expected_difficulty']}, 实际: {multi_result.intent.difficulty})")

        # 计算效率提升
        if multi_result.intent.difficulty == TaskDifficulty.SIMPLE:
            # 简单任务使用更快的模型，应该有时间优势
            efficiency = "使用轻量模型，节省成本"
        elif multi_result.intent.difficulty == TaskDifficulty.COMPLEX:
            # 复杂任务使用更强的模型，质量优先
            efficiency = "使用高级模型，确保质量"
        else:
            efficiency = "使用标准模型，平衡性能"

        print(f"[策略] {efficiency}")

        results.append({
            "name": test["name"],
            "multi_time": multi_time,
            "baseline_time": baseline_time,
            "difficulty": multi_result.intent.difficulty,
            "expected": test["expected_difficulty"],
            "match": difficulty_match,
            "model_used": multi_result.model_used,
        })

    # 总结报告
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    correct = sum(1 for r in results if r["match"])
    print(f"\n难度判断准确率: {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")

    print("\n耗时对比:")
    print(f"{'测试用例':<20} {'多Agent(ms)':<15} {'单模型(ms)':<15} {'使用模型':<15}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<20} {r['multi_time']:<15.0f} {r['baseline_time']:<15.0f} {r['model_used']:<15}")

    # 成本分析
    print("\n成本分析:")
    print("- 简单任务使用 qwen-turbo: 成本约为 qwen-plus 的 1/5")
    print("- 复杂任务使用 qwen-max: 质量更高，避免多次重试")
    print("- 意图识别开销: 约 100-300ms，使用最便宜的模型")


async def test_tree_generation(api_key: str, base_url: str):
    """专门测试知识树生成"""
    print("\n" + "=" * 60)
    print("知识树生成专项测试")
    print("=" * 60)

    orchestrator = MultiAgentOrchestrator(api_key, base_url)

    test_content = """
    机器学习是人工智能的一个分支，它使计算机能够从数据中学习。
    主要分为三类：
    1. 监督学习：有标签数据，如分类和回归
    2. 无监督学习：无标签数据，如聚类和降维
    3. 强化学习：通过奖励信号学习策略

    常用算法包括：线性回归、决策树、支持向量机、神经网络等。
    """

    result = await orchestrator.process(f"请将以下内容整理成知识树：{test_content}")

    print(f"\n意图识别: {result.intent.task_type}")
    print(f"难度判断: {result.intent.difficulty}")
    print(f"使用模型: {result.model_used}")
    print(f"处理耗时: {result.processing_time_ms:.0f}ms")

    if "topic" in result.result:
        print(f"\n生成的知识树:")
        print(f"主题: {result.result.get('topic')}")
        print(f"摘要: {result.result.get('summary')}")
        print(f"核心概念: {result.result.get('concepts')}")
        print(f"节点数量: {len(result.result.get('nodes', []))}")

        for node in result.result.get("nodes", [])[:5]:
            indent = "  " if node.get("parent_id") else ""
            print(f"{indent}- {node.get('name')}: {node.get('description', '')[:50]}")


async def main():
    """主入口"""
    # 从环境变量获取 API 配置
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        print("错误: 未找到 DASHSCOPE_API_KEY 环境变量")
        print("请确保 .env.sqlite 或 .env 中配置了 DASHSCOPE_API_KEY")
        sys.exit(1)

    print(f"使用 API: {base_url}")
    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")

    # 运行对比测试
    await run_comparison_test(api_key, base_url)

    # 运行知识树专项测试
    await test_tree_generation(api_key, base_url)

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
