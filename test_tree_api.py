#!/usr/bin/env python3
"""
测试知识树生成 API

运行: python test_tree_api.py
"""
import asyncio
import os
import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv(".env.sqlite")
load_dotenv(".env")

from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.services.tree_generation import generate_knowledge_tree


async def test():
    print("=" * 60)
    print("知识树生成 API 测试")
    print("=" * 60)

    db = SessionLocal()

    # 获取一个测试用户 ID（使用第一个用户）
    from app.models import User
    user = db.query(User).first()
    if not user:
        print("错误: 数据库中没有用户，请先创建用户")
        return

    print(f"测试用户: {user.id}")

    content = """机器学习是人工智能的分支，包括：
    1. 监督学习：分类和回归
    2. 无监督学习：聚类和降维
    3. 深度学习：神经网络"""

    # 测试官方 API 模式
    print("\n[测试1] 官方 API 模式 (V3 两阶段并行)")
    print("-" * 40)
    result = await generate_knowledge_tree(
        db=db,
        user_id=user.id,
        content=content,
        use_system=True,
    )

    print(f"成功: {result.success}")
    print(f"模式: {result.mode}")
    print(f"策略: {result.strategy}")
    print(f"耗时: {result.processing_time_ms:.0f}ms")
    print(f"模型: {result.model_used}")

    if result.tree:
        print(f"主题: {result.tree.topic}")
        print(f"节点数: {len(result.tree.nodes)}")
        for node in result.tree.nodes[:5]:
            level = len(str(node.id).split("."))
            indent = "  " * (level - 1)
            print(f"{indent}[{node.id}] {node.name}: {(node.description or '')[:40]}...")

    db.close()
    print("\n测试完成!")


if __name__ == "__main__":
    asyncio.run(test())
