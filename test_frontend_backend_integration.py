#!/usr/bin/env python3
"""
测试前端与 FastAPI 后端的集成
验证完整的数据持久化流程
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_integration():
    print("=" * 60)
    print("测试前端与 FastAPI 后端集成")
    print("=" * 60)
    
    # 1. 注册新用户
    print("\n[1] 注册新用户...")
    username = f"test_user_{int(time.time())}"
    password = "test123456"
    
    register_data = {
        "username": username,
        "password": password,
        "nickname": "测试用户"
    }
    
    response = requests.post(f"{API_URL}/api/auth/register", json=register_data)
    if response.status_code != 200:
        print(f"❌ 注册失败: {response.text}")
        return False
    
    user_data = response.json()
    token = user_data["token"]
    user_id = user_data["id"]
    print(f"✅ 注册成功 - 用户ID: {user_id}, Token: {token[:20]}...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. 创建主题
    print("\n[2] 创建主题...")
    topic_data = {
        "name": "Python 编程",
        "description": "学习 Python 编程语言",
        "scope": ["基础语法", "数据结构", "面向对象"],
        "keywords": ["Python", "编程", "开发"]
    }
    
    response = requests.post(f"{API_URL}/api/topics", json=topic_data, headers=headers)
    if response.status_code != 200:
        print(f"❌ 创建主题失败: {response.text}")
        return False
    
    topic = response.json()
    topic_id = topic["id"]
    print(f"✅ 创建主题成功 - ID: {topic_id}, 名称: {topic['name']}")
    
    # 3. 创建节点
    print("\n[3] 创建知识节点...")
    node_data = {
        "topicId": topic_id,
        "name": "Python 基础语法",
        "description": "学习 Python 的基本语法",
        "learningObjectives": ["掌握变量和数据类型", "理解控制流"],
        "keyConcepts": ["变量", "数据类型", "if语句", "循环"],
        "knowledgeType": "concept",
        "difficulty": "beginner",
        "estimatedMinutes": 60,
        "source": "manual",
        "parentId": None,
        "mastery": 0,
        "questionCount": 0,
        "correctCount": 0
    }
    
    response = requests.post(f"{API_URL}/api/nodes", json=node_data, headers=headers)
    if response.status_code != 200:
        print(f"❌ 创建节点失败: {response.text}")
        return False
    
    node = response.json()
    node_id = node["id"]
    print(f"✅ 创建节点成功 - ID: {node_id}, 名称: {node['name']}")
    
    # 4. 查询节点（验证持久化）
    print("\n[4] 查询节点列表...")
    response = requests.get(f"{API_URL}/api/nodes?topicId={topic_id}", headers=headers)
    if response.status_code != 200:
        print(f"❌ 查询节点失败: {response.text}")
        return False
    
    nodes = response.json()
    print(f"✅ 查询成功 - 找到 {len(nodes)} 个节点")
    
    if len(nodes) != 1:
        print(f"❌ 节点数量不匹配: 期望 1, 实际 {len(nodes)}")
        return False
    
    if nodes[0]["id"] != node_id:
        print(f"❌ 节点ID不匹配: 期望 {node_id}, 实际 {nodes[0]['id']}")
        return False
    
    # 5. 模拟退出登录（清除 token）
    print("\n[5] 模拟退出登录...")
    print("✅ 退出登录（前端会清除 token）")
    
    # 6. 重新登录
    print("\n[6] 重新登录...")
    login_data = {
        "username": username,
        "password": password
    }
    
    response = requests.post(f"{API_URL}/api/auth/login", json=login_data)
    if response.status_code != 200:
        print(f"❌ 登录失败: {response.text}")
        return False
    
    new_user_data = response.json()
    new_token = new_user_data["token"]
    print(f"✅ 重新登录成功 - 新Token: {new_token[:20]}...")
    
    new_headers = {"Authorization": f"Bearer {new_token}"}
    
    # 7. 再次查询节点（验证数据仍然存在）
    print("\n[7] 重新登录后查询节点...")
    response = requests.get(f"{API_URL}/api/nodes?topicId={topic_id}", headers=new_headers)
    if response.status_code != 200:
        print(f"❌ 查询节点失败: {response.text}")
        return False
    
    nodes_after_login = response.json()
    print(f"✅ 查询成功 - 找到 {len(nodes_after_login)} 个节点")
    
    if len(nodes_after_login) != 1:
        print(f"❌ 节点数量不匹配: 期望 1, 实际 {len(nodes_after_login)}")
        return False
    
    if nodes_after_login[0]["id"] != node_id:
        print(f"❌ 节点ID不匹配: 期望 {node_id}, 实际 {nodes_after_login[0]['id']}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！数据持久化正常工作！")
    print("=" * 60)
    print("\n测试总结:")
    print(f"  - 用户: {username}")
    print(f"  - 主题: {topic['name']} (ID: {topic_id})")
    print(f"  - 节点: {node['name']} (ID: {node_id})")
    print(f"  - 退出登录后重新登录，节点仍然存在 ✓")
    print("\n🎉 数据持久化问题已解决！")
    
    return True

if __name__ == "__main__":
    try:
        success = test_integration()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

