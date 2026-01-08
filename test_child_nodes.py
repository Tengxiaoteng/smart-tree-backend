#!/usr/bin/env python3
"""
测试子节点的创建和持久化
"""

import requests
import json
from datetime import datetime

API_URL = "http://localhost:8000"

def test_child_nodes():
    print("=" * 60)
    print("测试子节点创建和持久化")
    print("=" * 60)
    
    # 1. 注册/登录
    print("\n[1] 登录...")
    username = f"test_child_{datetime.now().strftime('%H%M%S')}"
    password = "test123"
    
    register_response = requests.post(
        f"{API_URL}/api/auth/register",
        json={"username": username, "password": password, "nickname": "测试用户"}
    )
    
    if register_response.status_code == 200:
        data = register_response.json()
        token = data['token']  # 直接从响应中获取 token
        print(f"✓ 注册成功，用户: {username}")
    else:
        print(f"✗ 注册失败: {register_response.text}")
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. 创建主题
    print("\n[2] 创建主题...")
    topic_data = {
        "id": f"topic_{datetime.now().timestamp()}",
        "name": "测试主题",
        "description": "用于测试子节点",
        "createdAt": datetime.now().isoformat(),
        "updatedAt": datetime.now().isoformat()
    }
    
    topic_response = requests.post(
        f"{API_URL}/api/topics",
        headers=headers,
        json=topic_data
    )
    
    if topic_response.status_code == 200:
        topic = topic_response.json()
        topic_id = topic['id']
        print(f"✓ 主题创建成功: {topic['name']}")
    else:
        print(f"✗ 主题创建失败: {topic_response.text}")
        return
    
    # 3. 创建根节点
    print("\n[3] 创建根节点...")
    root_node_data = {
        "id": f"node_root_{datetime.now().timestamp()}",
        "name": "根节点",
        "description": "这是根节点",
        "topicId": topic_id,
        "parentId": None,
        "knowledgeType": "concept",
        "difficulty": "beginner",
        "estimatedMinutes": 30,
        "source": "manual",
        "mastery": 0,
        "questionCount": 0,
        "correctCount": 0,
        "children": [],
        "createdAt": datetime.now().isoformat(),
        "updatedAt": datetime.now().isoformat()
    }
    
    root_response = requests.post(
        f"{API_URL}/api/nodes",
        headers=headers,
        json=root_node_data
    )
    
    if root_response.status_code == 200:
        root_node = root_response.json()
        root_id = root_node['id']
        print(f"✓ 根节点创建成功: {root_node['name']} (ID: {root_id})")
    else:
        print(f"✗ 根节点创建失败: {root_response.text}")
        return
    
    # 4. 创建子节点
    print("\n[4] 创建子节点...")
    child_node_data = {
        "id": f"node_child_{datetime.now().timestamp()}",
        "name": "子节点",
        "description": "这是子节点",
        "topicId": topic_id,
        "parentId": root_id,  # 指定父节点
        "knowledgeType": "concept",
        "difficulty": "beginner",
        "estimatedMinutes": 20,
        "source": "manual",
        "mastery": 0,
        "questionCount": 0,
        "correctCount": 0,
        "children": [],
        "createdAt": datetime.now().isoformat(),
        "updatedAt": datetime.now().isoformat()
    }
    
    child_response = requests.post(
        f"{API_URL}/api/nodes",
        headers=headers,
        json=child_node_data
    )
    
    if child_response.status_code == 200:
        child_node = child_response.json()
        child_id = child_node['id']
        print(f"✓ 子节点创建成功: {child_node['name']} (ID: {child_id})")
        print(f"  父节点ID: {child_node.get('parentId')}")
    else:
        print(f"✗ 子节点创建失败: {child_response.text}")
        return
    
    # 5. 获取所有节点
    print("\n[5] 获取所有节点...")
    nodes_response = requests.get(
        f"{API_URL}/api/nodes",
        headers=headers
    )
    
    if nodes_response.status_code == 200:
        nodes = nodes_response.json()
        print(f"✓ 获取到 {len(nodes)} 个节点:")
        for node in nodes:
            parent_info = f"(父节点: {node.get('parentId')})" if node.get('parentId') else "(根节点)"
            print(f"  - {node['name']} {parent_info}")
    else:
        print(f"✗ 获取节点失败: {nodes_response.text}")
        return
    
    # 6. 验证父子关系
    print("\n[6] 验证父子关系...")
    root_nodes = [n for n in nodes if not n.get('parentId')]
    child_nodes = [n for n in nodes if n.get('parentId')]
    
    print(f"✓ 根节点数量: {len(root_nodes)}")
    print(f"✓ 子节点数量: {len(child_nodes)}")
    
    if len(child_nodes) > 0:
        for child in child_nodes:
            parent_exists = any(n['id'] == child['parentId'] for n in nodes)
            if parent_exists:
                print(f"✓ 子节点 '{child['name']}' 的父节点存在")
            else:
                print(f"✗ 子节点 '{child['name']}' 的父节点不存在！")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    print(f"\n测试账号: {username}")
    print(f"密码: {password}")
    print(f"主题ID: {topic_id}")
    print(f"根节点ID: {root_id}")
    print(f"子节点ID: {child_id}")
    print("\n你可以用这个账号登录前端验证数据是否正确显示。")

if __name__ == "__main__":
    test_child_nodes()

