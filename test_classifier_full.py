"""
完整测试智能分类器
使用现有知识树测试分类效果
"""
import asyncio
import os
import sys
sys.path.insert(0, '/app')

from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.topic import Topic
from app.models.knowledge_node import KnowledgeNode
from app.services.smart_classifier import SmartClassifier
from app.schemas.material_classifier import MockTree, MockNode

def build_mock_tree(db: Session, topic: Topic) -> tuple[MockTree, dict, dict]:
    """构建 MockTree 和相关数据"""
    nodes = db.query(KnowledgeNode).filter(KnowledgeNode.topicId == topic.id).all()
    
    if not nodes:
        return None, {}, {}
    
    # 找到根节点
    root_nodes = [n for n in nodes if n.parentId is None]
    if not root_nodes:
        root_nodes = [nodes[0]]
    
    def build_mock_node(node) -> MockNode:
        children = [n for n in nodes if n.parentId == node.id]
        return MockNode(
            id=node.id,
            name=node.name,
            children=[build_mock_node(c) for c in children]
        )
    
    root_node = build_mock_node(root_nodes[0])
    
    mock_tree = MockTree(
        id=topic.id,
        name=topic.name,
        root_node=root_node
    )
    
    nodes_by_tree = {
        topic.id: [MockNode(id=n.id, name=n.name, children=[]) for n in nodes]
    }
    
    materials_by_node = {n.id: [] for n in nodes}
    
    return mock_tree, nodes_by_tree, materials_by_node

async def test_classifier():
    print("=" * 70)
    print("智能分类器完整测试")
    print("=" * 70)
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 未设置 DASHSCOPE_API_KEY")
        return
    
    db = SessionLocal()
    
    try:
        topics = db.query(Topic).all()
        print(f"\n数据库中共有 {len(topics)} 棵知识树")
        
        # 构建节点ID到名称的映射
        node_id_to_name = {}
        for topic in topics:
            nodes = db.query(KnowledgeNode).filter(KnowledgeNode.topicId == topic.id).all()
            for n in nodes:
                node_id_to_name[n.id] = n.name
        
        test_materials = [
            {
                "name": "Docker镜像构建",
                "content": "Dockerfile是构建Docker镜像的脚本文件，包含FROM、RUN、COPY、CMD等指令。通过docker build命令可以根据Dockerfile构建镜像。多阶段构建可以减小最终镜像体积。"
            },
            {
                "name": "音频特征提取",
                "content": "音频特征提取是音频分类的关键步骤。常用特征包括MFCC（梅尔频率倒谱系数）、频谱质心、过零率等。时域特征反映信号的时间变化，频域特征反映频率分布。"
            },
            {
                "name": "细胞呼吸",
                "content": "细胞呼吸是生物体内有机物氧化分解释放能量的过程。有氧呼吸分为三个阶段：糖酵解、三羧酸循环和氧化磷酸化。线粒体是有氧呼吸的主要场所。"
            }
        ]
        
        classifier = SmartClassifier(api_key=api_key)
        
        all_trees = []
        all_nodes_by_tree = {}
        all_materials_by_node = {}
        
        for topic in topics:
            mock_tree, nodes_by_tree, materials_by_node = build_mock_tree(db, topic)
            if mock_tree:
                all_trees.append(mock_tree)
                all_nodes_by_tree.update(nodes_by_tree)
                all_materials_by_node.update(materials_by_node)
                print(f"  - {topic.name}: {len(nodes_by_tree[topic.id])} 节点")
        
        total_nodes = sum(len(nodes) for nodes in all_nodes_by_tree.values())
        print(f"\n总节点数: {total_nodes}")
        print(f"预期使用: {'v3 (Embedding)' if total_nodes >= 500 else 'v2 (纯LLM)'}")
        
        for material in test_materials:
            print(f"\n{'='*70}")
            print(f"测试资料: {material['name']}")
            print("="*70)
            print(f"内容: {material['content'][:80]}...")
            
            try:
                result = await classifier.classify(
                    content=material["content"],
                    trees=all_trees,
                    nodes_by_tree=all_nodes_by_tree,
                    materials_by_node=all_materials_by_node
                )
                
                print(f"\n分类结果:")
                print(f"  关键词: {result.keywords}")
                print(f"  摘要: {result.summary}")
                print(f"  匹配的树: {[(t.tree_name) for t in result.matched_trees]}")
                print(f"  API调用: {result.total_api_calls}, Token: {result.total_tokens_used}")
                print(f"  耗时: {result.processing_time_ms:.0f}ms")
                
                if result.tree_node_decisions:
                    print(f"\n  节点定位:")
                    for tree_id, decision in result.tree_node_decisions.items():
                        tree_name = next((t.name for t in topics if t.id == tree_id), tree_id[:8])
                        target_name = node_id_to_name.get(decision.target_node_id, decision.suggested_node_name or "未知")
                        print(f"    [{tree_name}]")
                        print(f"      操作: {decision.action}")
                        print(f"      目标节点: {target_name}")
                        print(f"      置信度: {decision.confidence}")
                        print(f"      理由: {decision.reason[:50]}...")
                else:
                    print(f"\n  节点定位: 无")
                    
            except Exception as e:
                import traceback
                print(f"  错误: {e}")
                traceback.print_exc()
        
        print("\n" + "="*70)
        print("测试完成！")
        
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_classifier())
