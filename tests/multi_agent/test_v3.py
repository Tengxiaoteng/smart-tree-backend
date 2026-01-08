#!/usr/bin/env python3
"""
V3 两阶段并行架构测试

对比：
- V2: 单次生成所有内容
- V3: 规划 + 并发填充
- V3-Batch: 规划 + 批量并发

运行: python -m tests.multi_agent.test_v3
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

from tests.multi_agent.agents_v2 import ParallelOrchestrator
from tests.multi_agent.agents_v3 import TwoStageOrchestrator, TwoStageBatchOrchestrator


TEST_CASES = [
    {
        "name": "机器学习基础",
        "input": """请帮我整理机器学习的知识体系，包括：
        1. 监督学习：分类、回归
        2. 无监督学习：聚类、降维
        3. 强化学习：马尔可夫决策、Q-learning
        4. 深度学习：神经网络、CNN、RNN
        5. 常用算法：线性回归、决策树、SVM""",
    },
    {
        "name": "Python 数据分析",
        "input": """Python 数据分析知识树：
        NumPy: 数组操作、广播机制
        Pandas: DataFrame、数据清洗
        Matplotlib: 图表绑制
        Scikit-learn: 模型训练""",
    },
]


async def run_test():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        print("错误: 未找到 DASHSCOPE_API_KEY")
        sys.exit(1)

    print("=" * 70)
    print("V3 两阶段并行架构测试")
    print("=" * 70)

    v2_orch = ParallelOrchestrator(api_key, base_url)
    v3_orch = TwoStageOrchestrator(api_key, base_url)
    v3_batch_orch = TwoStageBatchOrchestrator(api_key, base_url)

    results = []

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n{'='*70}")
        print(f"测试 {i}/{len(TEST_CASES)}: {test['name']}")
        print("=" * 70)

        test_result = {"name": test["name"]}

        # V2 测试
        print("\n" + "-" * 40)
        print("[V2 - 单次生成]")
        print("-" * 40)
        v2_start = time.time()
        v2_result = await v2_orch.process(test["input"])
        v2_time = (time.time() - v2_start) * 1000
        v2_nodes = len(v2_result.result.get("nodes", []))
        print(f"\n结果: {v2_nodes} 节点, {v2_time:.0f}ms")
        test_result["v2_time"] = v2_time
        test_result["v2_nodes"] = v2_nodes

        # V3 测试（并发版）
        print("\n" + "-" * 40)
        print("[V3 - 规划 + 并发填充]")
        print("-" * 40)
        v3_start = time.time()
        v3_result = await v3_orch.process(test["input"])
        v3_time = (time.time() - v3_start) * 1000
        v3_nodes = len(v3_result.result.get("nodes", []))
        print(f"\n结果: {v3_nodes} 节点, {v3_time:.0f}ms")
        test_result["v3_time"] = v3_time
        test_result["v3_nodes"] = v3_nodes

        # V3-Batch 测试
        print("\n" + "-" * 40)
        print("[V3-Batch - 规划 + 批量并发]")
        print("-" * 40)
        v3b_start = time.time()
        v3b_result = await v3_batch_orch.process(test["input"])
        v3b_time = (time.time() - v3b_start) * 1000
        v3b_nodes = len(v3b_result.result.get("nodes", []))
        print(f"\n结果: {v3b_nodes} 节点, {v3b_time:.0f}ms")
        test_result["v3b_time"] = v3b_time
        test_result["v3b_nodes"] = v3b_nodes

        results.append(test_result)

        # 打印 V3 生成的结构
        print(f"\n[V3 知识树结构]")
        tree = v3_result.result
        print(f"主题: {tree.get('topic')}")

        nodes = tree.get("nodes", [])
        for node in nodes[:8]:
            level = len(str(node.get("id", "1")).split("."))
            indent = "  " * (level - 1)
            desc = (node.get("description") or "")[:40]
            print(f"{indent}[{node.get('id')}] {node.get('name')}: {desc}...")

    # 总结
    print("\n" + "=" * 70)
    print("性能对比总结")
    print("=" * 70)

    print(f"\n{'测试用例':<20} {'V2(ms)':<12} {'V3(ms)':<12} {'V3-Batch(ms)':<14} {'V2节点':<8} {'V3节点':<8}")
    print("-" * 74)

    for r in results:
        print(f"{r['name']:<20} {r['v2_time']:<12.0f} {r['v3_time']:<12.0f} {r['v3b_time']:<14.0f} {r['v2_nodes']:<8} {r['v3_nodes']:<8}")

    # 计算平均
    avg_v2 = sum(r["v2_time"] for r in results) / len(results)
    avg_v3 = sum(r["v3_time"] for r in results) / len(results)
    avg_v3b = sum(r["v3b_time"] for r in results) / len(results)

    print("-" * 74)
    print(f"{'平均':<20} {avg_v2:<12.0f} {avg_v3:<12.0f} {avg_v3b:<14.0f}")

    v3_speedup = (avg_v2 - avg_v3) / avg_v2 * 100 if avg_v2 > 0 else 0
    v3b_speedup = (avg_v2 - avg_v3b) / avg_v2 * 100 if avg_v2 > 0 else 0

    print(f"\n速度提升:")
    print(f"  V3 vs V2: {v3_speedup:+.1f}%")
    print(f"  V3-Batch vs V2: {v3b_speedup:+.1f}%")

    print("\n架构对比:")
    print("  V2: 单次大请求生成所有内容")
    print("  V3: 规划(快速模型) + 并发N个小请求填充")
    print("  V3-Batch: 规划(快速模型) + 按层级批量并发填充")


if __name__ == "__main__":
    asyncio.run(run_test())
