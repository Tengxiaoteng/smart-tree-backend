"""
为测试账号生成并保存知识树到数据库
"""
import asyncio
import os
import sys
import uuid
sys.path.insert(0, '/app')

from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.topic import Topic
from app.models.knowledge_node import KnowledgeNode
from app.services.tree_generation import generate_knowledge_tree

# 测试账号
USERS = {
    "大树账号": {
        "id": "50e46b58-7c2a-4f05-b38d-107151cdd3ca",
        "content": "计算机科学与软件工程完整知识体系，包括：编程语言（Python、Java、JavaScript、C++、Go、Rust）、数据结构与算法、操作系统、计算机网络、数据库系统、分布式系统、云计算、人工智能与机器学习、深度学习、自然语言处理、计算机视觉、Web开发（前端框架React/Vue/Angular、后端框架Spring/Django/Express）、移动开发（iOS/Android/Flutter）、DevOps与CI/CD、微服务架构、容器化技术Docker/Kubernetes、网络安全、密码学、软件工程方法论、敏捷开发、系统设计、性能优化"
    },
    "中等树账号": {
        "id": "18ed6edc-90c0-40d7-be7d-e6df48a6108a",
        "content": "Python编程基础：变量与数据类型、控制流程、函数、面向对象编程、文件操作、异常处理"
    }
}

def save_tree_to_db(db: Session, user_id: str, tree_result):
    """将生成的知识树保存到数据库"""
    if not tree_result.success or not tree_result.tree:
        print(f"  生成失败: {tree_result.error}")
        return None
    
    tree = tree_result.tree
    
    # 创建 Topic
    topic_id = f"topic-{uuid.uuid4().hex[:12]}"
    topic = Topic(
        id=topic_id,
        userId=user_id,
        name=tree.topic,
        description=tree.summary,
        scope="private",
        keywords=tree.concepts
    )
    db.add(topic)
    db.flush()
    
    # 创建节点
    nodes_created = []
    for node_data in tree.nodes:
        node = KnowledgeNode(
            id=f"node-{uuid.uuid4().hex[:12]}",
            userId=user_id,
            topicId=topic_id,
            name=node_data.name,
            description=node_data.description,
            parentId=None,  # 先设为 None，后面更新
            learningObjectives=node_data.learning_objectives,
            keyConcepts=node_data.key_concepts,
            knowledgeType=node_data.knowledge_type or "concept",
            difficulty=node_data.difficulty or "beginner",
            estimatedMinutes=node_data.estimated_minutes or 15,
            questionPatterns=node_data.question_patterns,
            commonMistakes=node_data.common_mistakes,
            source="ai_generated"
        )
        nodes_created.append((node_data.id, node_data.parent_id, node))
        db.add(node)
    
    db.flush()
    
    # 建立 ID 映射
    old_to_new_id = {old_id: node.id for old_id, _, node in nodes_created}
    
    # 更新父节点 ID
    for old_id, old_parent_id, node in nodes_created:
        if old_parent_id and old_parent_id in old_to_new_id:
            node.parentId = old_to_new_id[old_parent_id]
    
    # 设置根节点
    root_nodes = [node for _, parent_id, node in nodes_created if not parent_id]
    if root_nodes:
        topic.rootNodeId = root_nodes[0].id
    
    db.commit()
    
    return topic, len(nodes_created)

async def main():
    print("=" * 70)
    print("为测试账号生成并保存知识树")
    print("=" * 70)
    
    db = SessionLocal()
    
    try:
        for user_name, user_info in USERS.items():
            user_id = user_info["id"]
            content = user_info["content"]
            
            print(f"\n处理: {user_name}")
            print("-" * 50)
            
            # 检查是否已有知识树
            existing = db.query(Topic).filter(Topic.userId == user_id).first()
            if existing:
                print(f"  已有知识树: {existing.name}，跳过")
                continue
            
            # 生成知识树
            print(f"  正在生成知识树...")
            result = await generate_knowledge_tree(
                db=db,
                user_id=user_id,
                content=content,
                use_system=True
            )
            
            if result.success:
                print(f"  生成成功: {result.tree.topic} ({len(result.tree.nodes)} 节点)")
                print(f"  耗时: {result.processing_time_ms/1000:.1f}秒")
                
                # 保存到数据库
                print(f"  正在保存到数据库...")
                topic, node_count = save_tree_to_db(db, user_id, result)
                print(f"  保存成功: {topic.name} (ID: {topic.id})")
                print(f"  节点数: {node_count}")
            else:
                print(f"  生成失败: {result.error}")
        
        print("\n" + "=" * 70)
        print("完成！")
        
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main())
