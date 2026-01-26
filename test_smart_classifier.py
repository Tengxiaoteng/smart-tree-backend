"""
测试智能分类器 - 验证 v2/v3 自动切换
"""
import asyncio
import os
import sys
sys.path.insert(0, '/app')

from app.services.smart_classifier import SmartClassifier

async def test():
    print("=" * 60)
    print("智能分类器测试")
    print("=" * 60)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 未设置 DASHSCOPE_API_KEY")
        return
    
    # 创建测试知识树（小树）
    from app.schemas.material_classifier import MockTree, MockNode
    
    small_tree = MockTree(
        id="tree_small",
        name="小型测试树",
        root_node=MockNode(
            id="root",
            name="根节点",
            children=[
                MockNode(id="n1", name="Python编程", children=[]),
                MockNode(id="n2", name="数据分析", children=[]),
                MockNode(id="n3", name="机器学习", children=[])
            ]
        )
    )
    
    # 构建 nodes_by_tree
    nodes_by_tree = {
        "tree_small": [
            MockNode(id="n1", name="Python编程", children=[]),
            MockNode(id="n2", name="数据分析", children=[]),
            MockNode(id="n3", name="机器学习", children=[])
        ]
    }
    
    # 构建 materials_by_node
    materials_by_node = {
        "n1": ["Python基础", "Python进阶"],
        "n2": ["数据分析入门"],
        "n3": ["机器学习概述"]
    }
    
    # 测试资料内容
    test_content = "这是一份Python基础教程，包含变量、函数、类等内容"
    
    print("\n1. 测试小树（应使用 v2）")
    print("-" * 40)
    
    classifier = SmartClassifier(api_key=api_key)
    result = await classifier.classify(
        content=test_content,
        trees=[small_tree],
        nodes_by_tree=nodes_by_tree,
        materials_by_node=materials_by_node
    )
    
    print(f"关键词: {result.keywords}")
    print(f"摘要: {result.summary}")
    print(f"匹配的树: {[t.tree_id for t in result.matched_trees]}")
    print(f"节点决策: {list(result.tree_node_decisions.keys())}")
    print(f"API 调用次数: {result.total_api_calls}")
    print(f"Token 消耗: {result.total_tokens_used}")
    print(f"处理时间: {result.processing_time_ms}ms")
    
    # 打印详细的节点决策
    for tree_id, decision in result.tree_node_decisions.items():
        print(f"\n树 {tree_id} 的决策:")
        print(f"  目标节点: {decision.target_node_id}")
        print(f"  节点名称: {decision.target_node_name}")
        print(f"  置信度: {decision.confidence}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    
asyncio.run(test())
