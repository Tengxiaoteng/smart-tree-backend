"""
测试数据库持久化
用于验证节点创建后是否真正保存到数据库
"""
import sys
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, engine
from app.models import User, Topic, KnowledgeNode
from app.core.security import get_password_hash
import uuid

def test_node_persistence():
    """测试节点持久化"""
    db: Session = SessionLocal()
    
    try:
        # 1. 创建测试用户
        test_username = f"test_user_{uuid.uuid4().hex[:8]}"
        test_user = User(
            id=str(uuid.uuid4()),
            username=test_username,
            password=get_password_hash("test123"),
            nickname="测试用户"
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        print(f"✓ 创建测试用户: {test_user.username} (ID: {test_user.id})")
        
        # 2. 创建测试主题
        test_topic = Topic(
            id=str(uuid.uuid4()),
            userId=test_user.id,
            name="测试主题",
            description="用于测试的主题"
        )
        db.add(test_topic)
        db.commit()
        db.refresh(test_topic)
        print(f"✓ 创建测试主题: {test_topic.name} (ID: {test_topic.id})")
        
        # 3. 创建测试节点
        test_node = KnowledgeNode(
            id=str(uuid.uuid4()),
            userId=test_user.id,
            topicId=test_topic.id,
            name="测试节点",
            description="用于测试的节点",
            knowledgeType="concept",
            difficulty="beginner",
            estimatedMinutes=15,
            source="manual"
        )
        db.add(test_node)
        db.commit()
        db.refresh(test_node)
        print(f"✓ 创建测试节点: {test_node.name} (ID: {test_node.id})")
        
        # 4. 关闭当前会话，模拟"退出登录"
        node_id = test_node.id
        user_id = test_user.id
        db.close()
        print("\n✓ 关闭数据库会话（模拟退出登录）")
        
        # 5. 创建新会话，模拟"重新登录"
        db = SessionLocal()
        print("✓ 创建新数据库会话（模拟重新登录）")
        
        # 6. 查询节点是否还存在
        found_node = db.query(KnowledgeNode).filter(
            KnowledgeNode.id == node_id,
            KnowledgeNode.userId == user_id
        ).first()
        
        if found_node:
            print(f"\n✅ 成功！节点在新会话中找到: {found_node.name}")
            print(f"   节点ID: {found_node.id}")
            print(f"   用户ID: {found_node.userId}")
            print(f"   主题ID: {found_node.topicId}")
            print(f"   创建时间: {found_node.createdAt}")
        else:
            print(f"\n❌ 失败！节点在新会话中未找到")
            print(f"   查询条件: node_id={node_id}, user_id={user_id}")
            
        # 7. 查询该用户的所有节点
        all_nodes = db.query(KnowledgeNode).filter(
            KnowledgeNode.userId == user_id
        ).all()
        print(f"\n该用户的所有节点数量: {len(all_nodes)}")
        for node in all_nodes:
            print(f"  - {node.name} (ID: {node.id})")
            
        # 8. 清理测试数据
        print("\n清理测试数据...")
        db.query(KnowledgeNode).filter(KnowledgeNode.userId == user_id).delete()
        db.query(Topic).filter(Topic.userId == user_id).delete()
        db.query(User).filter(User.id == user_id).delete()
        db.commit()
        print("✓ 测试数据已清理")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("=" * 60)
    print("测试数据库持久化")
    print("=" * 60)
    test_node_persistence()
    print("=" * 60)

