"""
节点扩展服务 V2

使用 LangChain + Pydantic 实现：
1. Planner（规划器）：分析父节点，规划需要的子节点
2. Deduplicator（去重器）：排除已有的子节点
3. Workers（并发生成器）：批量并发生成节点内容
"""
import asyncio
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ==================== Pydantic 结构化输出模型 ====================

class PlannedNode(BaseModel):
    """规划阶段的节点（只有名称和简要说明）"""
    name: str = Field(description="子节点名称，简洁明确")
    brief: str = Field(description="一句话说明该节点要覆盖的内容")
    reason: str = Field(description="为什么需要这个子节点")


class PlanResult(BaseModel):
    """规划结果"""
    analysis: str = Field(description="对父节点的分析")
    planned_nodes: list[PlannedNode] = Field(description="规划的子节点列表，3-6个")


class GeneratedNodeContent(BaseModel):
    """生成的节点完整内容"""
    name: str = Field(description="节点名称")
    description: str = Field(description="详细描述，50-100字")
    learning_objectives: list[str] = Field(
        default_factory=list,
        description="学习目标，2-3个，以'能够'开头"
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="关键概念/术语，2-4个"
    )
    knowledge_type: str = Field(
        default="concept",
        description="知识类型：concept/principle/procedure/application"
    )
    difficulty: str = Field(
        default="beginner",
        description="难度：beginner/intermediate/advanced"
    )
    estimated_minutes: int = Field(
        default=10,
        description="预计学习时间（分钟）"
    )
    question_patterns: list[str] = Field(
        default_factory=list,
        description="出题方向，1-2个"
    )
    common_mistakes: list[str] = Field(
        default_factory=list,
        description="常见错误，1-2个"
    )


class ExpansionResult(BaseModel):
    """扩展结果"""
    success: bool
    parent_node_name: str
    existing_children: list[str] = Field(default_factory=list)
    planned_count: int = 0
    deduplicated_count: int = 0
    generated_nodes: list[GeneratedNodeContent] = Field(default_factory=list)
    error: Optional[str] = None


# ==================== 核心服务类 ====================

class NodeExpansionService:
    """节点扩展服务（Orchestrator-Worker 模式）"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "deepseek-chat",
        max_concurrency: int = 5
    ):
        # 初始化 LangChain LLM（OpenAI 兼容）
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/") + "/v1" if not base_url.endswith("/v1") else base_url,
            model=model,
            temperature=0.3,
            max_tokens=2000,
        )
        self.max_concurrency = max_concurrency
        
        # 初始化解析器
        self.plan_parser = PydanticOutputParser(pydantic_object=PlanResult)
        self.content_parser = PydanticOutputParser(pydantic_object=GeneratedNodeContent)
    
    # ==================== Stage 1: Planner ====================
    
    async def plan_children(
        self,
        parent_name: str,
        parent_description: str,
        parent_difficulty: str,
        parent_knowledge_type: str,
        parent_key_concepts: list[str],
        parent_learning_objectives: list[str],
    ) -> PlanResult:
        """
        规划阶段：分析父节点，规划需要的子节点
        
        返回：计划生成的子节点列表（只有名称和简要说明）
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个教学设计专家。你的任务是分析一个知识节点，并规划它的子节点。

## 规划原则
1. 子节点应该是父节点的逻辑分解或细化
2. 子节点之间应该有清晰的边界，不重叠
3. 子节点应该按照学习顺序排列（从基础到进阶）
4. 每个子节点都应该可以独立学习和测试
5. 子节点的难度应该 ≤ 父节点难度或略低

## 输出格式
{format_instructions}

请只输出 JSON，不要有其他内容。"""),
            ("user", """请为以下知识节点规划子节点：

**节点名称**：{parent_name}
**节点描述**：{parent_description}
**知识类型**：{parent_knowledge_type}
**难度等级**：{parent_difficulty}
**关键概念**：{parent_key_concepts}
**学习目标**：{parent_learning_objectives}

请分析这个节点，并规划 3-6 个合理的子节点。""")
        ])
        
        chain = prompt | self.llm
        
        result = await chain.ainvoke({
            "parent_name": parent_name,
            "parent_description": parent_description or "暂无描述",
            "parent_knowledge_type": parent_knowledge_type or "concept",
            "parent_difficulty": parent_difficulty or "beginner",
            "parent_key_concepts": "、".join(parent_key_concepts) if parent_key_concepts else "暂无",
            "parent_learning_objectives": "；".join(parent_learning_objectives) if parent_learning_objectives else "暂无",
            "format_instructions": self.plan_parser.get_format_instructions(),
        })
        
        try:
            return self.plan_parser.parse(result.content)
        except Exception as e:
            # 尝试手动解析
            import json
            import re
            content = result.content
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                data = json.loads(match.group(0))
                return PlanResult(**data)
            raise ValueError(f"无法解析规划结果: {e}")
    
    # ==================== Stage 2: Deduplicator ====================
    
    def deduplicate(
        self,
        planned_nodes: list[PlannedNode],
        existing_children: list[str],
    ) -> list[PlannedNode]:
        """
        去重阶段：排除与已有子节点重复的规划节点
        
        使用模糊匹配来判断是否重复：
        - 完全相同的名称
        - 名称包含关系
        - 语义相似（简单的关键词匹配）
        """
        if not existing_children:
            return planned_nodes
        
        # 归一化已有子节点名称
        existing_normalized = set()
        for name in existing_children:
            normalized = name.lower().strip()
            existing_normalized.add(normalized)
            # 提取关键词（去除常见前后缀）
            for prefix in ["第一章", "第二章", "第1章", "第2章", "一、", "二、", "1.", "2."]:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):].strip()
            existing_normalized.add(normalized)
        
        # 过滤重复的规划节点
        unique_nodes = []
        for node in planned_nodes:
            node_normalized = node.name.lower().strip()
            
            # 检查是否与已有节点重复
            is_duplicate = False
            for existing in existing_normalized:
                # 完全匹配
                if node_normalized == existing:
                    is_duplicate = True
                    break
                # 包含关系（双向）
                if node_normalized in existing or existing in node_normalized:
                    is_duplicate = True
                    break
                # 关键词重叠度 > 50%
                node_words = set(node_normalized.split())
                existing_words = set(existing.split())
                if node_words and existing_words:
                    overlap = len(node_words & existing_words) / min(len(node_words), len(existing_words))
                    if overlap > 0.5:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_nodes.append(node)
        
        return unique_nodes
    
    # ==================== Stage 3: Workers (Parallel Content Generation) ====================
    
    async def generate_node_content(
        self,
        planned_node: PlannedNode,
        parent_context: str,
    ) -> GeneratedNodeContent:
        """
        为单个规划节点生成完整内容
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个知识内容专家。你的任务是为一个知识节点生成完整的学习信息。

## 内容要求
- description: 详细描述，50-100字，说明这个知识点的核心内容
- learning_objectives: 2-3个学习目标，每个以能够开头
- key_concepts: 2-4个关键概念或术语
- knowledge_type: 知识类型（concept概念/principle原理/procedure流程/application应用）
- difficulty: 难度（beginner初级/intermediate中级/advanced高级）
- estimated_minutes: 预计学习时间（分钟）
- question_patterns: 1-2个可能的出题方向
- common_mistakes: 1-2个学习者常犯的错误

## 输出格式
{format_instructions}

请只输出 JSON，不要有其他内容。"""),
            ("user", """**父节点上下文**：{parent_context}

**要生成内容的节点**：
- 名称：{node_name}
- 简要说明：{node_brief}
- 设计原因：{node_reason}

请为这个节点生成完整的学习信息。""")
        ])
        
        chain = prompt | self.llm
        
        result = await chain.ainvoke({
            "parent_context": parent_context,
            "node_name": planned_node.name,
            "node_brief": planned_node.brief,
            "node_reason": planned_node.reason,
            "format_instructions": self.content_parser.get_format_instructions(),
        })
        
        try:
            return self.content_parser.parse(result.content)
        except Exception as e:
            # 尝试手动解析
            import json
            import re
            content = result.content
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                data = json.loads(match.group(0))
                return GeneratedNodeContent(**data)
            # 返回默认值
            return GeneratedNodeContent(
                name=planned_node.name,
                description=planned_node.brief,
            )
    
    async def generate_all_contents(
        self,
        planned_nodes: list[PlannedNode],
        parent_context: str,
    ) -> list[GeneratedNodeContent]:
        """
        并发生成所有节点的内容（使用信号量控制并发数）
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def generate_with_limit(node: PlannedNode) -> GeneratedNodeContent:
            async with semaphore:
                return await self.generate_node_content(node, parent_context)
        
        # 并发执行所有任务
        tasks = [generate_with_limit(node) for node in planned_nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        generated = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 生成失败，使用规划信息作为默认值
                generated.append(GeneratedNodeContent(
                    name=planned_nodes[i].name,
                    description=planned_nodes[i].brief,
                ))
            else:
                generated.append(result)
        
        return generated
    
    # ==================== Main Orchestrator ====================
    
    async def expand_node(
        self,
        parent_name: str,
        parent_description: str = "",
        parent_difficulty: str = "beginner",
        parent_knowledge_type: str = "concept",
        parent_key_concepts: list[str] = None,
        parent_learning_objectives: list[str] = None,
        existing_children: list[str] = None,
    ) -> ExpansionResult:
        """
        主入口：扩展一个节点的子节点
        
        流程：
        1. Planner: 分析父节点，规划子节点
        2. Deduplicator: 排除已有子节点
        3. Workers: 并发生成节点内容
        """
        parent_key_concepts = parent_key_concepts or []
        parent_learning_objectives = parent_learning_objectives or []
        existing_children = existing_children or []
        
        try:
            # Stage 1: 规划
            plan_result = await self.plan_children(
                parent_name=parent_name,
                parent_description=parent_description,
                parent_difficulty=parent_difficulty,
                parent_knowledge_type=parent_knowledge_type,
                parent_key_concepts=parent_key_concepts,
                parent_learning_objectives=parent_learning_objectives,
            )
            planned_count = len(plan_result.planned_nodes)
            
            # Stage 2: 去重
            unique_nodes = self.deduplicate(
                planned_nodes=plan_result.planned_nodes,
                existing_children=existing_children,
            )
            deduplicated_count = len(unique_nodes)
            
            if not unique_nodes:
                return ExpansionResult(
                    success=True,
                    parent_node_name=parent_name,
                    existing_children=existing_children,
                    planned_count=planned_count,
                    deduplicated_count=0,
                    generated_nodes=[],
                    error="所有规划的子节点都与已有子节点重复",
                )
            
            # Stage 3: 并发生成内容
            parent_context = f"{parent_name}：{parent_description or '无描述'}"
            generated_nodes = await self.generate_all_contents(
                planned_nodes=unique_nodes,
                parent_context=parent_context,
            )
            
            return ExpansionResult(
                success=True,
                parent_node_name=parent_name,
                existing_children=existing_children,
                planned_count=planned_count,
                deduplicated_count=deduplicated_count,
                generated_nodes=generated_nodes,
            )
            
        except Exception as e:
            return ExpansionResult(
                success=False,
                parent_node_name=parent_name,
                existing_children=existing_children,
                error=str(e),
            )


# ==================== 便捷函数 ====================

async def expand_node_v2(
    api_key: str,
    base_url: str,
    model: str,
    parent_name: str,
    parent_description: str = "",
    parent_difficulty: str = "beginner",
    parent_knowledge_type: str = "concept",
    parent_key_concepts: list[str] = None,
    parent_learning_objectives: list[str] = None,
    existing_children: list[str] = None,
    max_concurrency: int = 5,
) -> ExpansionResult:
    """
    扩展节点的便捷函数
    """
    service = NodeExpansionService(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_concurrency=max_concurrency,
    )
    
    return await service.expand_node(
        parent_name=parent_name,
        parent_description=parent_description,
        parent_difficulty=parent_difficulty,
        parent_knowledge_type=parent_knowledge_type,
        parent_key_concepts=parent_key_concepts,
        parent_learning_objectives=parent_learning_objectives,
        existing_children=existing_children,
    )
