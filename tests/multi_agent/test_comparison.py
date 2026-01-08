#!/usr/bin/env python3
"""
多 Agent 架构对比测试 - v1 vs v2 vs LangChain

对比:
1. v1: 原始版本（顺序执行）
2. v2: 并行处理版本
3. LangChain: 使用 LangChain 框架

运行: python -m tests.multi_agent.test_comparison
"""
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(".env.sqlite")
load_dotenv(".env")

from tests.multi_agent.agents import MultiAgentOrchestrator
from tests.multi_agent.agents_v2 import ParallelOrchestrator, LangChainOrchestrator, LANGCHAIN_AVAILABLE
from tests.multi_agent.schemas import TaskDifficulty


# 测试用例 - 专注知识树生成
TREE_TEST_CASES = [
    {
        "name": "机器学习基础",
        "input": """请帮我整理机器学习的知识体系，包括：
        1. 监督学习：分类、回归
        2. 无监督学习：聚类、降维
        3. 强化学习：马尔可夫决策、Q-learning
        4. 深度学习：神经网络、CNN、RNN、Transformer
        5. 常用算法：线性回归、决策树、SVM、随机森林、XGBoost
        6. 评估指标：准确率、精确率、召回率、F1、AUC""",
    },
    {
        "name": "线性代数核心",
        "input": """整理线性代数在机器学习中的应用：
        - 向量：点积、范数、正交
        - 矩阵：转置、逆矩阵、行列式
        - 特征分解：特征值、特征向量
        - SVD分解：奇异值、降维应用
        - 矩阵分解：LU分解、QR分解
        - 应用：PCA、线性回归、推荐系统""",
    },
    {
        "name": "Python 数据分析",
        "input": """Python 数据分析完整知识树：
        NumPy: 数组操作、广播机制、线性代数
        Pandas: DataFrame、数据清洗、分组聚合、时间序列
        Matplotlib: 图表类型、样式定制、子图布局
        Seaborn: 统计图表、热力图、分布图
        Scikit-learn: 预处理、模型训练、交叉验证
        数据处理流程：导入、清洗、转换、分析、可视化""",
    },
]


async def compare_tree_generation(api_key: str, base_url: str):
    """对比知识树生成效果"""
    print("=" * 70)
    print("知识树生成对比测试 - 三层结构 + 丰富内容")
    print("=" * 70)

    # 初始化三个版本
    v1_orch = MultiAgentOrchestrator(api_key, base_url)
    v2_orch = ParallelOrchestrator(api_key, base_url)

    langchain_orch = None
    if LANGCHAIN_AVAILABLE:
        try:
            langchain_orch = LangChainOrchestrator(api_key, base_url)
            print("[✓] LangChain 可用")
        except Exception as e:
            print(f"[✗] LangChain 初始化失败: {e}")
    else:
        print("[✗] LangChain 未安装")

    results = []

    for i, test in enumerate(TREE_TEST_CASES, 1):
        print(f"\n{'='*70}")
        print(f"测试 {i}/{len(TREE_TEST_CASES)}: {test['name']}")
        print("=" * 70)

        test_result = {"name": test["name"]}

        # V1 测试
        print("\n[V1 - 原始版本]")
        v1_start = time.time()
        v1_result = await v1_orch.process(test["input"])
        v1_time = (time.time() - v1_start) * 1000
        v1_nodes = len(v1_result.result.get("nodes", []))
        print(f"  耗时: {v1_time:.0f}ms, 节点数: {v1_nodes}")
        test_result["v1_time"] = v1_time
        test_result["v1_nodes"] = v1_nodes

        # V2 测试
        print("\n[V2 - 并行版本]")
        v2_start = time.time()
        v2_result = await v2_orch.process(test["input"])
        v2_time = (time.time() - v2_start) * 1000
        v2_nodes = len(v2_result.result.get("nodes", []))
        print(f"  耗时: {v2_time:.0f}ms, 节点数: {v2_nodes}")
        test_result["v2_time"] = v2_time
        test_result["v2_nodes"] = v2_nodes

        # LangChain 测试
        if langchain_orch:
            print("\n[LangChain 版本]")
            lc_start = time.time()
            lc_result = await langchain_orch.process(test["input"])
            lc_time = (time.time() - lc_start) * 1000
            lc_nodes = len(lc_result.result.get("nodes", []))
            print(f"  耗时: {lc_time:.0f}ms, 节点数: {lc_nodes}")
            test_result["lc_time"] = lc_time
            test_result["lc_nodes"] = lc_nodes
        else:
            test_result["lc_time"] = None
            test_result["lc_nodes"] = None

        results.append(test_result)

        # 打印 V2 生成的知识树结构（作为示例）
        print(f"\n[V2 知识树结构 - {test['name']}]")
        tree = v2_result.result
        print(f"主题: {tree.get('topic')}")
        print(f"概述: {tree.get('summary', '')[:100]}...")
        print(f"核心概念: {tree.get('concepts', [])}")
        print(f"节点层级分布:")

        nodes = tree.get("nodes", [])
        level_counts = {}
        for node in nodes:
            level = len(str(node.get("id", "1")).split("."))
            level_counts[level] = level_counts.get(level, 0) + 1

        for level in sorted(level_counts.keys()):
            print(f"  第{level}层: {level_counts[level]} 个节点")

        # 打印前几个节点作为示例
        print("\n节点示例:")
        for node in nodes[:6]:
            indent = "  " * (len(str(node.get("id", "1")).split(".")) - 1)
            desc = (node.get("description") or "")[:50]
            print(f"{indent}[{node.get('id')}] {node.get('name')}: {desc}...")

    # 总结
    print("\n" + "=" * 70)
    print("性能对比总结")
    print("=" * 70)

    print(f"\n{'测试用例':<20} {'V1(ms)':<12} {'V2(ms)':<12} {'LC(ms)':<12} {'V1节点':<10} {'V2节点':<10} {'LC节点':<10}")
    print("-" * 86)

    for r in results:
        lc_time = f"{r['lc_time']:.0f}" if r["lc_time"] else "N/A"
        lc_nodes = r["lc_nodes"] if r["lc_nodes"] else "N/A"
        print(f"{r['name']:<20} {r['v1_time']:<12.0f} {r['v2_time']:<12.0f} {lc_time:<12} {r['v1_nodes']:<10} {r['v2_nodes']:<10} {lc_nodes:<10}")

    # 计算平均值
    avg_v1 = sum(r["v1_time"] for r in results) / len(results)
    avg_v2 = sum(r["v2_time"] for r in results) / len(results)
    avg_v1_nodes = sum(r["v1_nodes"] for r in results) / len(results)
    avg_v2_nodes = sum(r["v2_nodes"] for r in results) / len(results)

    print("-" * 86)
    print(f"{'平均':<20} {avg_v1:<12.0f} {avg_v2:<12.0f}")

    speedup = (avg_v1 - avg_v2) / avg_v1 * 100 if avg_v1 > 0 else 0
    node_increase = (avg_v2_nodes - avg_v1_nodes) / avg_v1_nodes * 100 if avg_v1_nodes > 0 else 0

    print(f"\nV2 相比 V1:")
    print(f"  速度提升: {speedup:.1f}%")
    print(f"  节点增加: {node_increase:.1f}%")


async def main():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        print("错误: 未找到 DASHSCOPE_API_KEY")
        sys.exit(1)

    print(f"API: {base_url}")
    print(f"Key: {api_key[:8]}...{api_key[-4:]}")

    await compare_tree_generation(api_key, base_url)

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
