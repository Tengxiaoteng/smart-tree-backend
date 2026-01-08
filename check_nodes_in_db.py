#!/usr/bin/env python3
"""
检查数据库中的节点数据
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.models import KnowledgeNode, User
from app.core.database import get_db

def check_nodes():
    print("=" * 80)
    print("检查数据库中的知识节点")
    print("=" * 80)
    
    # 创建数据库连接
    from app.core.database import SessionLocal
    db = SessionLocal()
    
    try:
        # 1. 获取所有用户
        print("\n[1] 所有用户:")
        users = db.query(User).all()
        for user in users:
            print(f"  - {user.username} (ID: {user.id})")
        
        if not users:
            print("  ❌ 没有用户")
            return
        
        # 2. 获取所有节点
        print("\n[2] 所有节点:")
        nodes = db.query(KnowledgeNode).order_by(KnowledgeNode.createdAt.desc()).all()
        
        if not nodes:
            print("  ❌ 没有节点")
            return
        
        print(f"  ✓ 找到 {len(nodes)} 个节点\n")
        
        # 3. 按用户分组显示
        for user in users:
            user_nodes = [n for n in nodes if n.userId == user.id]
            if not user_nodes:
                continue
                
            print(f"\n用户: {user.username}")
            print("-" * 80)
            
            # 分类显示
            root_nodes = [n for n in user_nodes if not n.parentId]
            child_nodes = [n for n in user_nodes if n.parentId]
            
            print(f"\n  根节点 ({len(root_nodes)} 个):")
            for node in root_nodes:
                print(f"    ✓ {node.name}")
                print(f"      ID: {node.id}")
                print(f"      topicId: {node.topicId}")
                print(f"      创建时间: {node.createdAt}")
                
                # 查找这个根节点的子节点
                children = [n for n in child_nodes if n.parentId == node.id]
                if children:
                    print(f"      子节点 ({len(children)} 个):")
                    for child in children:
                        print(f"        → {child.name} (ID: {child.id})")
                print()
            
            print(f"\n  子节点 ({len(child_nodes)} 个):")
            for node in child_nodes:
                parent = next((n for n in user_nodes if n.id == node.parentId), None)
                parent_name = parent.name if parent else "❌ 父节点不存在"
                
                print(f"    ✓ {node.name}")
                print(f"      ID: {node.id}")
                print(f"      parentId: {node.parentId}")
                print(f"      父节点: {parent_name}")
                print(f"      topicId: {node.topicId}")
                print(f"      创建时间: {node.createdAt}")
                print()
        
        # 4. 检查孤儿节点（父节点不存在的子节点）
        print("\n[3] 检查数据完整性:")
        orphan_nodes = []
        for node in nodes:
            if node.parentId:
                parent_exists = any(n.id == node.parentId for n in nodes)
                if not parent_exists:
                    orphan_nodes.append(node)
        
        if orphan_nodes:
            print(f"  ⚠️  发现 {len(orphan_nodes)} 个孤儿节点（父节点不存在）:")
            for node in orphan_nodes:
                print(f"    - {node.name} (parentId: {node.parentId})")
        else:
            print("  ✓ 所有节点的父子关系都正确")
        
        # 5. 查找名为 "机器" 和 "1" 的节点
        print("\n[4] 查找特定节点:")
        machine_nodes = [n for n in nodes if "机器" in n.name]
        one_nodes = [n for n in nodes if n.name == "1"]
        
        if machine_nodes:
            print(f"\n  包含 '机器' 的节点 ({len(machine_nodes)} 个):")
            for node in machine_nodes:
                print(f"    ✓ {node.name}")
                print(f"      ID: {node.id}")
                print(f"      parentId: {node.parentId}")
                print(f"      topicId: {node.topicId}")
                
                # 查找子节点
                children = [n for n in nodes if n.parentId == node.id]
                if children:
                    print(f"      子节点:")
                    for child in children:
                        print(f"        → {child.name} (ID: {child.id})")
                print()
        else:
            print("  ❌ 没有找到包含 '机器' 的节点")
        
        if one_nodes:
            print(f"\n  名为 '1' 的节点 ({len(one_nodes)} 个):")
            for node in one_nodes:
                parent = next((n for n in nodes if n.id == node.parentId), None)
                parent_name = parent.name if parent else "无父节点（根节点）"
                
                print(f"    ✓ {node.name}")
                print(f"      ID: {node.id}")
                print(f"      parentId: {node.parentId}")
                print(f"      父节点: {parent_name}")
                print(f"      topicId: {node.topicId}")
                print(f"      创建时间: {node.createdAt}")
                print()
        else:
            print("  ❌ 没有找到名为 '1' 的节点")
        
        print("\n" + "=" * 80)
        print("检查完成")
        print("=" * 80)
        
    finally:
        db.close()

if __name__ == "__main__":
    check_nodes()

