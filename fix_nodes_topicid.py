#!/usr/bin/env python3
"""
为没有 topicId 的节点创建默认主题
"""

from app.core.database import SessionLocal
from app.models import KnowledgeNode, Topic, User
from datetime import datetime
import uuid

def fix_nodes_topicid():
    print("=" * 80)
    print("修复节点的 topicId")
    print("=" * 80)
    
    db = SessionLocal()
    
    try:
        # 1. 查找所有没有 topicId 的节点
        nodes_without_topic = db.query(KnowledgeNode).filter(
            KnowledgeNode.topicId == None
        ).all()
        
        if not nodes_without_topic:
            print("\n✓ 所有节点都有 topicId，无需修复")
            return
        
        print(f"\n找到 {len(nodes_without_topic)} 个没有 topicId 的节点")
        
        # 2. 按用户分组
        users_with_orphan_nodes = {}
        for node in nodes_without_topic:
            if node.userId not in users_with_orphan_nodes:
                users_with_orphan_nodes[node.userId] = []
            users_with_orphan_nodes[node.userId].append(node)
        
        # 3. 为每个用户创建或使用默认主题
        for user_id, nodes in users_with_orphan_nodes.items():
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                print(f"\n⚠️  用户 {user_id} 不存在，跳过")
                continue
            
            print(f"\n用户: {user.username}")
            print(f"  需要修复 {len(nodes)} 个节点")
            
            # 查找或创建默认主题
            default_topic = db.query(Topic).filter(
                Topic.userId == user_id,
                Topic.name == "默认主题"
            ).first()
            
            if not default_topic:
                # 创建默认主题
                default_topic = Topic(
                    id=str(uuid.uuid4()),
                    userId=user_id,
                    name="默认主题",
                    description="自动创建的默认主题，用于存放未分类的节点",
                    createdAt=datetime.utcnow(),
                    updatedAt=datetime.utcnow()
                )
                db.add(default_topic)
                db.commit()
                print(f"  ✓ 创建了默认主题: {default_topic.id}")
            else:
                print(f"  ✓ 使用现有默认主题: {default_topic.id}")
            
            # 更新所有节点的 topicId
            updated_count = 0
            for i, node in enumerate(nodes):
                node.topicId = default_topic.id
                node.updatedAt = datetime.utcnow()

                # 如果有重名，添加序号
                if i > 0 and node.name == nodes[0].name:
                    node.name = f"{node.name}_{i+1}"
                    print(f"    - 更新节点（重命名）: {node.name}")
                else:
                    print(f"    - 更新节点: {node.name}")

                updated_count += 1

            try:
                db.commit()
                print(f"  ✓ 已更新 {updated_count} 个节点")
            except Exception as e:
                print(f"  ⚠️  批量更新失败，尝试逐个更新: {e}")
                db.rollback()

                # 逐个更新
                for i, node in enumerate(nodes):
                    try:
                        node.topicId = default_topic.id
                        node.updatedAt = datetime.utcnow()

                        # 如果有重名，添加序号
                        if i > 0:
                            node.name = f"{node.name}_{i+1}"

                        db.commit()
                        updated_count += 1
                        print(f"    ✓ 更新成功: {node.name}")
                    except Exception as e2:
                        print(f"    ✗ 更新失败: {node.name} - {e2}")
                        db.rollback()
                        continue

                print(f"  ✓ 成功更新 {updated_count}/{len(nodes)} 个节点")
        
        print("\n" + "=" * 80)
        print("✅ 修复完成！")
        print("=" * 80)
        
        # 4. 验证
        remaining = db.query(KnowledgeNode).filter(
            KnowledgeNode.topicId == None
        ).count()
        
        if remaining == 0:
            print("\n✓ 所有节点现在都有 topicId")
        else:
            print(f"\n⚠️  还有 {remaining} 个节点没有 topicId")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    fix_nodes_topicid()

