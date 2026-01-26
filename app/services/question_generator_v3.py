"""
V3 智能出题服务 - 基于RAG和Embedding的学习目标检索

核心流程：
1. 学习目标向量化（带缓存）
2. RAG检索相关资料片段
3. 构建精简上下文
4. LLM生成题目（带检索trace）

优化：
- 并行优化1：资料分块与目标向量化并发执行
- 并行优化2：多目标LLM生成并发执行
- 缓存优化：学习目标向量缓存
- 随机化：避免重复出题，增加题目多样性
- 用户历史：考虑用户答题记录，避免重复
"""

import json
import uuid
import logging
import asyncio
import time
import hashlib
import random
from typing import Optional
from datetime import datetime, timedelta
from functools import lru_cache

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.models.material import Material
from app.models.material_chunk import MaterialChunk
from app.models.knowledge_node import KnowledgeNode
from app.models.question import Question
from app.models.answer_record import AnswerRecord

logger = logging.getLogger(__name__)

# ===== 学习目标向量缓存 =====
# 全局缓存：goal_text_hash -> embedding
_goal_embedding_cache: dict[str, list[float]] = {}
_cache_stats = {"hits": 0, "misses": 0}


def _get_text_hash(text: str) -> str:
    """计算文本的哈希值"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_cached_embedding(text: str) -> Optional[list[float]]:
    """从缓存获取向量"""
    text_hash = _get_text_hash(text)
    if text_hash in _goal_embedding_cache:
        _cache_stats["hits"] += 1
        return _goal_embedding_cache[text_hash]
    _cache_stats["misses"] += 1
    return None


def set_cached_embedding(text: str, embedding: list[float]):
    """存入缓存"""
    text_hash = _get_text_hash(text)
    _goal_embedding_cache[text_hash] = embedding


def get_cache_stats() -> dict:
    """获取缓存统计"""
    total = _cache_stats["hits"] + _cache_stats["misses"]
    hit_rate = _cache_stats["hits"] / total if total > 0 else 0
    return {
        **_cache_stats,
        "total": total,
        "hit_rate": f"{hit_rate:.1%}",
        "cache_size": len(_goal_embedding_cache),
    }


class QuestionGeneratorV3:
    """V3智能出题生成器"""

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id
        self._embedding_service = None

    @property
    def embedding_service(self) -> EmbeddingService:
        """延迟初始化 Embedding 服务"""
        if self._embedding_service is None:
            api_key = settings.DASHSCOPE_API_KEY
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY 未配置")
            self._embedding_service = EmbeddingService(api_key=api_key)
        return self._embedding_service

    def _get_user_answer_history(self, node_id: str, days: int = 7) -> dict:
        """
        获取用户最近的答题历史

        Returns:
            {
                "answered_question_ids": set[str],  # 已答过的题目ID
                "used_chunk_ids": set[str],         # 已用过的资料片段ID
                "weak_goal_ids": set[str],          # 答错较多的目标ID
                "correct_rate": float,              # 正确率
            }
        """
        since_date = datetime.utcnow() - timedelta(days=days)

        # 查询最近的答题记录
        records = self.db.query(AnswerRecord).filter(
            AnswerRecord.userId == self.user_id,
            AnswerRecord.nodeId == node_id,
            AnswerRecord.createdAt >= since_date,
        ).all()

        answered_question_ids = set()
        correct_count = 0

        for record in records:
            answered_question_ids.add(record.questionId)
            if record.isCorrect:
                correct_count += 1

        # 获取已用过的资料片段（通过 ragTrace）
        used_chunk_ids = set()
        if answered_question_ids:
            questions = self.db.query(Question).filter(
                Question.id.in_(answered_question_ids)
            ).all()
            for q in questions:
                if q.ragTrace and isinstance(q.ragTrace, dict):
                    chunk_ids = q.ragTrace.get("retrievedMaterialIds", [])
                    used_chunk_ids.update(chunk_ids)

        # 计算正确率
        correct_rate = correct_count / len(records) if records else 0.5

        logger.info(f"[用户历史] 节点 {node_id}: 已答 {len(answered_question_ids)} 题, "
                   f"正确率 {correct_rate:.1%}, 已用片段 {len(used_chunk_ids)} 个")

        return {
            "answered_question_ids": answered_question_ids,
            "used_chunk_ids": used_chunk_ids,
            "correct_rate": correct_rate,
            "total_answered": len(records),
        }

    async def generate_questions_v3(
        self,
        node_id: str,
        goals: list[dict],
        material_ids: list[str],
        config: dict,
        learner_profile: Optional[dict] = None,
    ) -> list[dict]:
        """
        V3智能出题主函数

        Args:
            node_id: 知识节点ID
            goals: 学习目标列表 [{id, goal, importance, masteryScore, relatedConcepts}]
            material_ids: 资料ID列表
            config: 出题配置 {count, difficulty, questionTypes, mode}
            learner_profile: 用户画像（可选）

        Returns:
            生成的题目列表
        """
        start_time = time.time()
        logger.info(f"V3出题开始: node={node_id}, goals={len(goals)}, materials={len(material_ids)}")

        # 0. 获取用户答题历史（用于避免重复和个性化）
        user_history = self._get_user_answer_history(node_id)

        # 1. 获取节点和资料信息
        t1 = time.time()
        node = self.db.query(KnowledgeNode).filter(KnowledgeNode.id == node_id).first()
        if not node:
            raise ValueError(f"节点不存在: {node_id}")

        materials = self.db.query(Material).filter(Material.id.in_(material_ids)).all() if material_ids else []
        logger.info(f"[V3性能] 获取节点和资料: {(time.time() - t1) * 1000:.0f}ms")

        # 2. 【并行优化1】资料分块与目标向量化并发执行
        t2 = time.time()

        # 并行执行：资料分块 + 目标向量化
        chunk_task = self._ensure_material_chunks(materials)
        embedding_task = self._get_goal_embeddings_with_cache(goals)

        all_chunks, goal_embeddings = await asyncio.gather(
            chunk_task,
            embedding_task,
            return_exceptions=True
        )

        # 处理异常
        if isinstance(all_chunks, Exception):
            logger.error(f"资料分块失败: {all_chunks}")
            all_chunks = []
        if isinstance(goal_embeddings, Exception):
            logger.error(f"目标向量化失败: {goal_embeddings}")
            goal_embeddings = {}

        logger.info(f"[V3性能] 并行(资料分块+目标向量化): {(time.time() - t2) * 1000:.0f}ms, 缓存统计: {get_cache_stats()}")

        # 3. RAG检索（依赖上面的结果，加入随机化和历史避免）
        t3 = time.time()
        goal_contexts = self._build_goal_rag_contexts_from_embeddings(
            goals=goals,
            goal_embeddings=goal_embeddings,
            chunks=all_chunks,
            node=node,
            learner_profile=learner_profile,
            user_history=user_history,  # 传入用户历史
        )
        logger.info(f"[V3性能] RAG检索: {(time.time() - t3) * 1000:.0f}ms")

        # 4. 根据用户画像排序目标
        prioritized_goals = self._prioritize_goals_by_profile(goal_contexts, learner_profile)

        # 5. 为每个目标并行生成题目（并行优化2）
        t5 = time.time()
        all_questions = []
        questions_per_goal = max(1, config.get("count", 5) // len(prioritized_goals)) if prioritized_goals else 0

        if prioritized_goals:
            # 并行执行所有LLM调用
            tasks = [
                self._generate_questions_for_goal(
                    goal_context=goal_ctx,
                    node=node,
                    config=config,
                    count=questions_per_goal,
                    learner_profile=learner_profile,
                )
                for goal_ctx in prioritized_goals
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"目标 {idx} 生成题目失败: {result}")
                else:
                    all_questions.extend(result)

        logger.info(f"[V3性能] 并行LLM生成题目: {(time.time() - t5) * 1000:.0f}ms, 目标数: {len(prioritized_goals)}")

        # 6. 平衡题目难度和数量
        balanced_questions = self._balance_questions(all_questions, config)

        total_time = (time.time() - start_time) * 1000
        logger.info(f"V3出题完成: 生成 {len(balanced_questions)} 道题目, 总耗时: {total_time:.0f}ms")
        return balanced_questions

    async def _get_goal_embeddings_with_cache(self, goals: list[dict]) -> dict[str, list[float]]:
        """
        获取学习目标的向量（带缓存）

        Returns:
            {goal_id: embedding} 字典
        """
        if not goals:
            return {}

        result = {}
        uncached_goals = []
        uncached_texts = []

        # 1. 检查缓存
        for goal in goals:
            goal_id = goal.get("id", "")
            goal_text = goal.get("goal", "")

            cached = get_cached_embedding(goal_text)
            if cached is not None:
                result[goal_id] = cached
            else:
                uncached_goals.append(goal)
                uncached_texts.append(goal_text)

        logger.info(f"[缓存] 目标向量: 缓存命中 {len(goals) - len(uncached_goals)}/{len(goals)}")

        # 2. 批量向量化未缓存的
        if uncached_texts:
            try:
                embeddings = self.embedding_service.embed_texts(uncached_texts)
                for goal, embedding in zip(uncached_goals, embeddings):
                    goal_id = goal.get("id", "")
                    goal_text = goal.get("goal", "")
                    result[goal_id] = embedding
                    # 存入缓存
                    set_cached_embedding(goal_text, embedding)
            except Exception as e:
                logger.error(f"批量向量化失败: {e}")

        return result

    async def _ensure_material_chunks(self, materials: list[Material]) -> list[dict]:
        """
        确保资料已分块和向量化

        如果资料没有分块，则创建分块并存储
        """
        all_chunks = []

        for material in materials:
            # 检查是否已有分块
            existing_chunks = self.db.query(MaterialChunk).filter(
                MaterialChunk.materialId == material.id
            ).all()

            if existing_chunks:
                # 使用已有分块
                for chunk in existing_chunks:
                    all_chunks.append({
                        "chunk_id": chunk.id,
                        "material_id": material.id,
                        "material_name": material.name,
                        "text": chunk.chunkText,
                        "embedding": chunk.embedding,
                        "chunk_index": chunk.chunkIndex,
                    })
            else:
                # 创建新分块
                content = material.organizedContent or material.content or ""
                if not content:
                    continue

                indexed_chunks = self.embedding_service.index_material_chunks(
                    material_id=material.id,
                    content=content,
                    chunk_size=400,
                )

                # 存储到数据库
                for chunk_data in indexed_chunks:
                    db_chunk = MaterialChunk(
                        id=chunk_data["chunk_id"],
                        materialId=material.id,
                        chunkIndex=chunk_data["chunk_index"],
                        chunkText=chunk_data["text"],
                        embedding=chunk_data["embedding"],
                        tokenCount=chunk_data.get("token_count"),
                    )
                    self.db.add(db_chunk)

                    all_chunks.append({
                        "chunk_id": chunk_data["chunk_id"],
                        "material_id": material.id,
                        "material_name": material.name,
                        "text": chunk_data["text"],
                        "embedding": chunk_data["embedding"],
                        "chunk_index": chunk_data["chunk_index"],
                    })

                self.db.commit()
                logger.info(f"资料 {material.id} 分块完成: {len(indexed_chunks)} 个分块")

        return all_chunks

    def _build_goal_rag_contexts_from_embeddings(
        self,
        goals: list[dict],
        goal_embeddings: dict[str, list[float]],
        chunks: list[dict],
        node: KnowledgeNode,
        learner_profile: Optional[dict],
        user_history: Optional[dict] = None,
    ) -> list[dict]:
        """
        使用预先计算的向量构建RAG上下文（优化版，带随机化和历史避免）

        Args:
            goals: 学习目标列表
            goal_embeddings: {goal_id: embedding} 预计算的向量字典
            chunks: 资料分块列表
            node: 知识节点
            learner_profile: 用户画像
            user_history: 用户答题历史
        """
        if not goals:
            return []

        # 获取已用过的片段ID
        used_chunk_ids = user_history.get("used_chunk_ids", set()) if user_history else set()

        contexts = []
        for goal in goals:
            goal_id = goal.get("id", "")
            embedding = goal_embeddings.get(goal_id)

            if not embedding:
                logger.warning(f"目标 {goal_id} 没有向量，跳过")
                continue

            # 【随机化检索】检索更多候选，然后随机采样
            # 先获取 top_k * 3 的候选
            all_retrieved = self.embedding_service.retrieve_relevant_chunks(
                query_embedding=embedding,
                chunks=chunks,
                top_k=15,  # 取更多候选
                min_similarity=0.35,  # 稍微降低阈值
            )

            # 【历史避免】将候选分为"未用过"和"已用过"
            unused_chunks = []
            used_chunks = []
            for chunk in all_retrieved:
                chunk_id = chunk.get("chunk_id", "") or chunk.get("material_id", "")
                if chunk_id in used_chunk_ids:
                    used_chunks.append(chunk)
                else:
                    unused_chunks.append(chunk)

            # 【智能选择】优先使用未用过的片段
            # 从未用过的中随机选择（保持一定相似度排序）
            selected_chunks = []
            target_count = 5

            if unused_chunks:
                # 从未用过的片段中选择
                # 对高相似度的片段有更高的选择概率（加权随机）
                if len(unused_chunks) <= target_count:
                    selected_chunks = unused_chunks
                else:
                    # 加权随机采样：相似度越高，被选中概率越大
                    weights = [c.get("similarity", 0.5) ** 2 for c in unused_chunks]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]
                        # 使用 random.choices 进行加权采样（允许重复则去重）
                        sampled_indices = set()
                        attempts = 0
                        while len(sampled_indices) < target_count and attempts < 50:
                            idx = random.choices(range(len(unused_chunks)), weights=weights, k=1)[0]
                            sampled_indices.add(idx)
                            attempts += 1
                        selected_chunks = [unused_chunks[i] for i in sampled_indices]

            # 如果未用过的不够，补充已用过的（但相似度高的）
            remaining = target_count - len(selected_chunks)
            if remaining > 0 and used_chunks:
                # 从已用过的中选择相似度最高的
                selected_chunks.extend(used_chunks[:remaining])

            # 如果还是不够，放宽限制
            if len(selected_chunks) < 3 and all_retrieved:
                selected_chunks = all_retrieved[:target_count]

            # 按相似度重新排序
            selected_chunks.sort(key=lambda x: x.get("similarity", 0), reverse=True)

            logger.debug(f"[RAG随机化] 目标 {goal_id[:8]}: "
                        f"候选 {len(all_retrieved)}, 未用 {len(unused_chunks)}, "
                        f"选中 {len(selected_chunks)}")

            # 分析用户掌握情况
            mastery_score = goal.get("masteryScore", 0)
            is_weakness = False
            if learner_profile:
                weak_concepts = [wc.get("concept", "") for wc in learner_profile.get("weakConcepts", [])]
                related_concepts = goal.get("relatedConcepts", [])
                is_weakness = any(wc in related_concepts for wc in weak_concepts)

            # 推荐难度（也加入一点随机性）
            base_difficulty = "medium"
            if is_weakness:
                base_difficulty = "easy"
            elif mastery_score > 70:
                base_difficulty = "hard"

            # 10% 概率随机调整难度，增加多样性
            if random.random() < 0.1:
                difficulties = ["easy", "medium", "hard"]
                base_difficulty = random.choice(difficulties)

            contexts.append({
                "goalId": goal_id,
                "goalText": goal.get("goal", ""),
                "goalEmbedding": embedding,
                "importance": goal.get("importance", "should"),
                "retrievedMaterials": [
                    {
                        "materialId": c.get("material_id", ""),
                        "materialName": c.get("material_name", ""),
                        "chunkText": c.get("text", ""),
                        "similarity": c.get("similarity", 0),
                        "chunkIndex": c.get("chunk_index", 0),
                    }
                    for c in selected_chunks
                ],
                "userWeakness": is_weakness,
                "userMasteryScore": mastery_score,
                "recommendedDifficulty": base_difficulty,
                # 记录随机化信息
                "randomizationInfo": {
                    "totalCandidates": len(all_retrieved),
                    "unusedCount": len(unused_chunks),
                    "selectedCount": len(selected_chunks),
                },
            })

        return contexts

    def _prioritize_goals_by_profile(
        self,
        goal_contexts: list[dict],
        learner_profile: Optional[dict],
    ) -> list[dict]:
        """
        根据用户画像排序学习目标

        优先顺序：
        1. 薄弱点目标（用户答错较多的）
        2. must 重要性的目标
        3. 低掌握度的目标
        4. 其他目标
        """
        if not goal_contexts:
            return []

        def sort_key(ctx):
            # 薄弱点优先
            weakness_score = 100 if ctx.get("userWeakness") else 0

            # 重要性权重
            importance = ctx.get("importance", "should")
            importance_score = {"must": 50, "should": 30, "could": 10}.get(importance, 20)

            # 低掌握度优先
            mastery = ctx.get("userMasteryScore", 50)
            mastery_score = 100 - mastery

            return weakness_score + importance_score + mastery_score

        return sorted(goal_contexts, key=sort_key, reverse=True)

    async def _generate_questions_for_goal(
        self,
        goal_context: dict,
        node: KnowledgeNode,
        config: dict,
        count: int,
        learner_profile: Optional[dict],
    ) -> list[dict]:
        """
        为单个学习目标生成题目（带随机化）
        """
        from app.services.llm_service import call_llm

        # 构建精简上下文
        retrieved_materials = goal_context.get("retrievedMaterials", [])
        material_context = "\n\n".join([
            f"【片段{idx + 1}】（来自《{m.get('materialName', '未知')}》，相似度{m.get('similarity', 0):.2f}）\n{m.get('chunkText', '')}"
            for idx, m in enumerate(retrieved_materials[:5])
        ])

        # 节点知识
        node_context = f"【节点知识】\n{node.name}\n{node.description or ''}"
        if node.keyConcepts:
            concepts = node.keyConcepts if isinstance(node.keyConcepts, list) else json.loads(node.keyConcepts or "[]")
            if concepts:
                node_context += f"\n关键概念: {', '.join(concepts[:10])}"

        # 用户画像提示
        user_prompt = ""
        if learner_profile:
            user_prompt = f"""
【用户学习情况】
- 该目标掌握度：{goal_context.get('userMasteryScore', 0)}%
- 是否薄弱点：{'是' if goal_context.get('userWeakness') else '否'}
- 建议难度：{goal_context.get('recommendedDifficulty', 'medium')}
"""
            if goal_context.get("userWeakness"):
                weak_concepts = [wc.get("concept", "") for wc in learner_profile.get("weakConcepts", [])]
                if weak_concepts:
                    user_prompt += f"- 用户薄弱概念：{', '.join(weak_concepts[:5])}\n请在干扰项中针对这些薄弱点设计"

        # 【随机化】生成随机出题风格提示
        question_styles = [
            "请从概念理解的角度出题",
            "请从实际应用的角度出题",
            "请从对比分析的角度出题",
            "请从问题解决的角度出题",
            "请从案例分析的角度出题",
            "请设计需要综合运用知识的题目",
            "请设计考查细节理解的题目",
            "请设计需要推理判断的题目",
        ]
        random_style = random.choice(question_styles)

        # 随机决定是否要求创新题型
        innovation_hints = [
            "",
            "尝试设计一道情境题，将知识点放入实际场景中",
            "可以设计一道需要排除法解决的题目",
            "可以设计一道多选题来考查全面理解",
        ]
        innovation_hint = random.choice(innovation_hints)

        # 生成唯一的随机种子，用于日志追踪
        random_seed = random.randint(1000, 9999)

        # 构建prompt
        prompt = f"""你是一位专业的出题专家。请基于以下学习目标和相关资料，生成{count}道练习题。

【学习目标】
{goal_context.get('goalText', '')}

{node_context}

【相关资料片段】
{material_context if material_context else '无相关资料'}

{user_prompt}

【出题要求】
1. 题目必须紧扣学习目标"{goal_context.get('goalText', '')}"
2. 优先使用相似度高的资料片段
3. 难度：{goal_context.get('recommendedDifficulty', 'medium')}
4. 题型：单选题或多选题
5. 每道题包含4个选项
6. {random_style}
{f"7. {innovation_hint}" if innovation_hint else ""}

【多样性要求】(随机种子:{random_seed})
- 每道题的考查角度要不同
- 干扰项要设计得有迷惑性但不能有歧义
- 题目表述要清晰，避免与之前的题目雷同

请严格按照以下JSON格式返回（注意不要有任何markdown标记）：
[
  {{
    "type": "single",
    "content": "题目内容",
    "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
    "answer": "A",
    "explanation": "解析说明",
    "difficulty": "medium"
  }}
]
"""

        try:
            response = await call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=settings.SYSTEM_LLM_MODEL or "deepseek-chat",
                api_key=settings.SYSTEM_LLM_API_KEY or settings.DASHSCOPE_API_KEY,
                base_url=settings.SYSTEM_LLM_BASE_URL or settings.DASHSCOPE_BASE_URL,
            )

            # 解析响应
            questions = self._parse_questions_response(response, goal_context, node)
            return questions

        except Exception as e:
            logger.error(f"LLM生成题目失败: {e}")
            return []

    def _parse_questions_response(
        self,
        response: str,
        goal_context: dict,
        node: KnowledgeNode,
    ) -> list[dict]:
        """解析LLM返回的题目JSON"""
        try:
            # 清理响应文本
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            questions_data = json.loads(text)
            if not isinstance(questions_data, list):
                questions_data = [questions_data]

            questions = []
            for q_data in questions_data:
                question = {
                    "id": str(uuid.uuid4()),
                    "nodeId": node.id,
                    "type": q_data.get("type", "single"),
                    "content": q_data.get("content", ""),
                    "options": q_data.get("options", []),
                    "answer": q_data.get("answer", ""),
                    "explanation": q_data.get("explanation", ""),
                    "difficulty": q_data.get("difficulty", "medium"),
                    "source": "learning_goals",
                    "targetGoalIds": [goal_context.get("goalId", "")],
                    "targetGoalNames": [goal_context.get("goalText", "")],
                    "ragTrace": {
                        "retrievedMaterialIds": [
                            m.get("materialId", "") for m in goal_context.get("retrievedMaterials", [])
                        ],
                        "retrievedConcepts": [],
                        "avgSimilarity": sum(
                            m.get("similarity", 0) for m in goal_context.get("retrievedMaterials", [])
                        ) / max(len(goal_context.get("retrievedMaterials", [])), 1),
                    },
                    "createdAt": datetime.utcnow().isoformat(),
                }
                questions.append(question)

            return questions

        except json.JSONDecodeError as e:
            logger.error(f"解析题目JSON失败: {e}, response={response[:200]}")
            return []

    def _balance_questions(self, questions: list[dict], config: dict) -> list[dict]:
        """平衡题目难度和数量"""
        target_count = config.get("count", 5)
        difficulty_dist = config.get("difficulty", {"easy": 30, "medium": 50, "hard": 20})

        # 按难度分组
        by_difficulty = {"easy": [], "medium": [], "hard": []}
        for q in questions:
            diff = q.get("difficulty", "medium")
            if diff in by_difficulty:
                by_difficulty[diff].append(q)

        # 按比例选择
        result = []
        for diff, ratio in difficulty_dist.items():
            target = int(target_count * ratio / 100)
            available = by_difficulty.get(diff, [])
            result.extend(available[:target])

        # 如果不够，从其他难度补充
        if len(result) < target_count:
            remaining = [q for q in questions if q not in result]
            result.extend(remaining[:target_count - len(result)])

        return result[:target_count]


async def call_llm(
    messages: list[dict],
    model: str,
    api_key: str,
    base_url: str,
) -> str:
    """调用LLM API"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=4000,
    )
    return response.choices[0].message.content or ""
