"""
测试完整的API流程
模拟用户创建知识树、退出登录、重新登录、查询节点的完整流程
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_complete_flow():
    """测试完整流程"""
    print("=" * 60)
    print("测试完整的API流程")
    print("=" * 60)
    
    # 1. 注册新用户
    print("\n1. 注册新用户...")
    import uuid
    username = f"test_{uuid.uuid4().hex[:8]}"
    register_data = {
        "username": username,
        "password": "test123",
        "nickname": "测试用户"
    }
    
    response = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
    if response.status_code != 200:
        print(f"❌ 注册失败: {response.text}")
        return
    
    user_data = response.json()
    user_id = user_data["id"]
    token1 = user_data["token"]
    print(f"✓ 注册成功 - 用户ID: {user_id}")
    print(f"  Token: {token1[:20]}...")
    
    # 2. 创建主题
    print("\n2. 创建主题...")
    headers = {"Authorization": f"Bearer {token1}"}
    topic_data = {
        "name": "测试主题",
        "description": "用于测试的主题"
    }
    
    response = requests.post(f"{BASE_URL}/api/topics", json=topic_data, headers=headers)
    if response.status_code != 200:
        print(f"❌ 创建主题失败: {response.text}")
        return
    
    topic = response.json()
    topic_id = topic["id"]
    print(f"✓ 创建主题成功 - 主题ID: {topic_id}")
    
    # 3. 创建节点
    print("\n3. 创建节点...")
    node_data = {
        "name": "测试节点1",
        "description": "第一个测试节点",
        "topicId": topic_id,
        "knowledgeType": "concept",
        "difficulty": "beginner"
    }
    
    response = requests.post(f"{BASE_URL}/api/nodes", json=node_data, headers=headers)
    if response.status_code != 200:
        print(f"❌ 创建节点失败: {response.text}")
        return
    
    node1 = response.json()
    node1_id = node1["id"]
    print(f"✓ 创建节点成功 - 节点ID: {node1_id}")
    print(f"  节点名称: {node1['name']}")
    print(f"  用户ID: {node1['userId']}")
    print(f"  主题ID: {node1['topicId']}")
    
    # 4. 查询节点（第一次）
    print("\n4. 查询节点（登录状态）...")
    response = requests.get(f"{BASE_URL}/api/nodes?topicId={topic_id}", headers=headers)
    if response.status_code != 200:
        print(f"❌ 查询节点失败: {response.text}")
        return
    
    nodes = response.json()
    print(f"✓ 找到 {len(nodes)} 个节点")
    for node in nodes:
        print(f"  - {node['name']} (ID: {node['id']})")
    
    # 5. 模拟退出登录（清除token）
    print("\n5. 模拟退出登录...")
    print("✓ Token已清除")
    
    # 6. 重新登录
    print("\n6. 重新登录...")
    login_data = {
        "username": username,
        "password": "test123"
    }
    
    response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
    if response.status_code != 200:
        print(f"❌ 登录失败: {response.text}")
        return
    
    user_data2 = response.json()
    user_id2 = user_data2["id"]
    token2 = user_data2["token"]
    print(f"✓ 登录成功 - 用户ID: {user_id2}")
    print(f"  Token: {token2[:20]}...")
    
    # 验证用户ID是否一致
    if user_id != user_id2:
        print(f"⚠️  警告: 用户ID不一致！")
        print(f"  注册时: {user_id}")
        print(f"  登录时: {user_id2}")
    else:
        print(f"✓ 用户ID一致")
    
    # 7. 查询节点（重新登录后）
    print("\n7. 查询节点（重新登录后）...")
    headers2 = {"Authorization": f"Bearer {token2}"}
    response = requests.get(f"{BASE_URL}/api/nodes?topicId={topic_id}", headers=headers2)
    if response.status_code != 200:
        print(f"❌ 查询节点失败: {response.text}")
        return
    
    nodes2 = response.json()
    print(f"✓ 找到 {len(nodes2)} 个节点")
    for node in nodes2:
        print(f"  - {node['name']} (ID: {node['id']})")
    
    # 8. 对比结果
    print("\n8. 对比结果...")
    if len(nodes) == len(nodes2):
        print(f"✅ 成功！节点数量一致: {len(nodes)} = {len(nodes2)}")
    else:
        print(f"❌ 失败！节点数量不一致: {len(nodes)} ≠ {len(nodes2)}")
        print(f"  退出前: {len(nodes)} 个节点")
        print(f"  重新登录后: {len(nodes2)} 个节点")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        test_complete_flow()
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保后端服务正在运行")
        print("   运行命令: python main.py")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

