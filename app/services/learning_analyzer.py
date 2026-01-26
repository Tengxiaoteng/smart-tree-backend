"""
学习分析服务 - 分析用户学习画像，用于智能出题

基于用户的答题记录分析：
- 薄弱概念识别
- 难度表现分析
- 时间模式分析
- 错误模式识别
- 个性化出题建议
"""
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ==================== 数据模型 ====================

class ConceptMastery(BaseModel):
    """概念掌握度"""
    concept: str
    total_attempts: int = Field(alias="totalAttempts")
    correct_count: int = Field(alias="correctCount")
    mastery_rate: float = Field(alias="masteryRate")  # 0-100
    is_weak: bool = Field(alias="isWeak")

    class Config:
        populate_by_name = True


class DifficultyStats(BaseModel):
    """难度统计"""
    total: int = 0
    correct: int = 0
    rate: float = 0  # 0-100


class DifficultyPerformance(BaseModel):
    """难度表现"""
    easy: DifficultyStats = Field(default_factory=DifficultyStats)
    medium: DifficultyStats = Field(default_factory=DifficultyStats)
    hard: DifficultyStats = Field(default_factory=DifficultyStats)


class TimeAnalysis(BaseModel):
    """时间分析"""
    fast_correct: int = Field(0, alias="fastCorrect")
    slow_correct: int = Field(0, alias="slowCorrect")
    fast_wrong: int = Field(0, alias="fastWrong")
    slow_wrong: int = Field(0, alias="slowWrong")

    class Config:
        populate_by_name = True


class ErrorPattern(BaseModel):
    """错误模式"""
    pattern: str
    frequency: int
    examples: list[str] = Field(default_factory=list)
    suggested_focus: str = Field(alias="suggestedFocus")

    class Config:
        populate_by_name = True


class QuestionRecommendation(BaseModel):
    """出题建议"""
    type: str  # weak_concept / error_pattern / difficulty_gap
    priority: int  # 1-10
    description: str
    target_concepts: list[str] = Field(default_factory=list, alias="targetConcepts")
    suggested_difficulty: str = Field("medium", alias="suggestedDifficulty")
    reason: str

    class Config:
        populate_by_name = True


class LearnerProfile(BaseModel):
    """学习者画像"""
    node_id: str = Field(alias="nodeId")
    total_questions: int = Field(0, alias="totalQuestions")
    correct_rate: float = Field(0, alias="correctRate")  # 0-100
    avg_time_per_question: float = Field(0, alias="avgTimePerQuestion")  # 秒

    weak_concepts: list[ConceptMastery] = Field(default_factory=list, alias="weakConcepts")
    strong_concepts: list[ConceptMastery] = Field(default_factory=list, alias="strongConcepts")

    difficulty_performance: DifficultyPerformance = Field(
        default_factory=DifficultyPerformance,
        alias="difficultyPerformance"
    )
    time_analysis: TimeAnalysis = Field(default_factory=TimeAnalysis, alias="timeAnalysis")
    error_patterns: list[ErrorPattern] = Field(default_factory=list, alias="errorPatterns")
    recommendations: list[QuestionRecommendation] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class AnswerRecordInput(BaseModel):
    """答题记录输入"""
    question_id: str = Field(alias="questionId")
    is_correct: bool = Field(alias="isCorrect")
    time_spent: float = Field(alias="timeSpent")  # 秒

    class Config:
        populate_by_name = True


class QuestionInput(BaseModel):
    """题目输入"""
    id: str
    node_id: str = Field(alias="nodeId")
    difficulty: str = "medium"
    type: str = "single"
    content: str = ""
    related_concepts: list[str] = Field(default_factory=list, alias="relatedConcepts")

    class Config:
        populate_by_name = True


# ==================== 分析函数 ====================

def analyze_learner_profile(
    node_id: str,
    questions: list[QuestionInput],
    records: list[AnswerRecordInput]
) -> LearnerProfile:
    """
    分析用户在特定节点的学习画像
    """
    # 筛选该节点的题目和记录
    node_questions = [q for q in questions if q.node_id == node_id]
    node_question_ids = {q.id for q in node_questions}
    node_records = [r for r in records if r.question_id in node_question_ids]

    if not node_records:
        return LearnerProfile(node_id=node_id)

    # 构建题目映射
    question_map = {q.id: q for q in node_questions}

    # 基础统计
    total = len(node_records)
    correct = sum(1 for r in node_records if r.is_correct)
    correct_rate = (correct / total * 100) if total > 0 else 0
    avg_time = sum(r.time_spent for r in node_records) / total if total > 0 else 0

    # 分析各维度
    concept_mastery = _analyze_concept_mastery(node_records, question_map)
    weak_concepts = [c for c in concept_mastery if c.is_weak]
    strong_concepts = [c for c in concept_mastery if not c.is_weak and c.mastery_rate >= 80]

    difficulty_perf = _analyze_difficulty_performance(node_records, question_map)
    time_analysis = _analyze_time_patterns(node_records, avg_time)
    error_patterns = _analyze_error_patterns(node_records, question_map)
    recommendations = _generate_recommendations(
        weak_concepts, error_patterns, difficulty_perf, time_analysis
    )

    return LearnerProfile(
        node_id=node_id,
        total_questions=total,
        correct_rate=correct_rate,
        avg_time_per_question=avg_time,
        weak_concepts=weak_concepts,
        strong_concepts=strong_concepts,
        difficulty_performance=difficulty_perf,
        time_analysis=time_analysis,
        error_patterns=error_patterns,
        recommendations=recommendations,
    )


def _analyze_concept_mastery(
    records: list[AnswerRecordInput],
    question_map: dict[str, QuestionInput]
) -> list[ConceptMastery]:
    """分析概念掌握度"""
    concept_stats: dict[str, dict] = {}

    for record in records:
        question = question_map.get(record.question_id)
        if not question:
            continue

        concepts = question.related_concepts or []
        for concept in concepts:
            if concept not in concept_stats:
                concept_stats[concept] = {"total": 0, "correct": 0}
            concept_stats[concept]["total"] += 1
            if record.is_correct:
                concept_stats[concept]["correct"] += 1

    result = []
    for concept, stats in concept_stats.items():
        mastery_rate = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        result.append(ConceptMastery(
            concept=concept,
            total_attempts=stats["total"],
            correct_count=stats["correct"],
            mastery_rate=mastery_rate,
            is_weak=mastery_rate < 60 and stats["total"] >= 2,
        ))

    return sorted(result, key=lambda x: x.mastery_rate)


def _analyze_difficulty_performance(
    records: list[AnswerRecordInput],
    question_map: dict[str, QuestionInput]
) -> DifficultyPerformance:
    """分析难度表现"""
    stats = {
        "easy": {"total": 0, "correct": 0},
        "medium": {"total": 0, "correct": 0},
        "hard": {"total": 0, "correct": 0},
    }

    for record in records:
        question = question_map.get(record.question_id)
        if not question:
            continue

        diff = question.difficulty or "medium"
        if diff in stats:
            stats[diff]["total"] += 1
            if record.is_correct:
                stats[diff]["correct"] += 1

    return DifficultyPerformance(
        easy=DifficultyStats(
            total=stats["easy"]["total"],
            correct=stats["easy"]["correct"],
            rate=(stats["easy"]["correct"] / stats["easy"]["total"] * 100) if stats["easy"]["total"] > 0 else 0
        ),
        medium=DifficultyStats(
            total=stats["medium"]["total"],
            correct=stats["medium"]["correct"],
            rate=(stats["medium"]["correct"] / stats["medium"]["total"] * 100) if stats["medium"]["total"] > 0 else 0
        ),
        hard=DifficultyStats(
            total=stats["hard"]["total"],
            correct=stats["hard"]["correct"],
            rate=(stats["hard"]["correct"] / stats["hard"]["total"] * 100) if stats["hard"]["total"] > 0 else 0
        ),
    )


def _analyze_time_patterns(
    records: list[AnswerRecordInput],
    avg_time: float
) -> TimeAnalysis:
    """分析时间模式"""
    threshold = avg_time if avg_time > 0 else 30  # 默认30秒

    fast_correct = 0
    slow_correct = 0
    fast_wrong = 0
    slow_wrong = 0

    for record in records:
        is_fast = record.time_spent < threshold
        if is_fast and record.is_correct:
            fast_correct += 1
        elif not is_fast and record.is_correct:
            slow_correct += 1
        elif is_fast and not record.is_correct:
            fast_wrong += 1
        else:
            slow_wrong += 1

    return TimeAnalysis(
        fast_correct=fast_correct,
        slow_correct=slow_correct,
        fast_wrong=fast_wrong,
        slow_wrong=slow_wrong,
    )


def _analyze_error_patterns(
    records: list[AnswerRecordInput],
    question_map: dict[str, QuestionInput]
) -> list[ErrorPattern]:
    """分析错误模式"""
    patterns = []
    wrong_records = [r for r in records if not r.is_correct]

    if len(wrong_records) < 2:
        return patterns

    # 分析错误的难度分布
    wrong_by_difficulty: dict[str, int] = {}
    wrong_examples: dict[str, list[str]] = {}

    for record in wrong_records:
        question = question_map.get(record.question_id)
        if not question:
            continue

        diff = question.difficulty or "medium"
        wrong_by_difficulty[diff] = wrong_by_difficulty.get(diff, 0) + 1

        if diff not in wrong_examples:
            wrong_examples[diff] = []
        if len(wrong_examples[diff]) < 3:
            wrong_examples[diff].append(question.content[:50] + "...")

    # 如果某个难度错误率特别高
    diff_labels = {"easy": "简单", "medium": "中等", "hard": "困难"}
    focus_labels = {"easy": "基础概念", "medium": "综合应用", "hard": "深度理解"}

    for diff, count in wrong_by_difficulty.items():
        if count >= 2:
            patterns.append(ErrorPattern(
                pattern=f"{diff_labels.get(diff, diff)}题错误较多",
                frequency=count,
                examples=wrong_examples.get(diff, []),
                suggested_focus=f"需要加强{focus_labels.get(diff, '相关')}的练习",
            ))

    # 分析快速答错（可能是粗心或概念混淆）
    fast_wrong = [r for r in wrong_records if r.time_spent < 15]
    if len(fast_wrong) >= 2:
        examples = []
        for r in fast_wrong[:3]:
            q = question_map.get(r.question_id)
            if q:
                examples.append(q.content[:50] + "...")

        patterns.append(ErrorPattern(
            pattern="快速答错（可能粗心或概念混淆）",
            frequency=len(fast_wrong),
            examples=examples,
            suggested_focus="建议放慢答题速度，仔细审题",
        ))

    return patterns


def _generate_recommendations(
    weak_concepts: list[ConceptMastery],
    error_patterns: list[ErrorPattern],
    difficulty_perf: DifficultyPerformance,
    time_analysis: TimeAnalysis
) -> list[QuestionRecommendation]:
    """生成出题建议"""
    recommendations = []

    # 1. 针对薄弱概念出题（最高优先级）
    for weak in weak_concepts[:3]:
        priority = 10 - int(weak.mastery_rate / 10)
        recommendations.append(QuestionRecommendation(
            type="weak_concept",
            priority=priority,
            description=f"针对薄弱概念「{weak.concept}」出题",
            target_concepts=[weak.concept],
            suggested_difficulty="easy" if weak.mastery_rate < 30 else "medium",
            reason=f"该概念正确率仅 {weak.mastery_rate:.0f}%，需要重点巩固",
        ))

    # 2. 针对错误模式出题
    for pattern in error_patterns:
        recommendations.append(QuestionRecommendation(
            type="error_pattern",
            priority=7,
            description=f"针对错误模式「{pattern.pattern}」出题",
            target_concepts=[],
            suggested_difficulty="medium",
            reason=pattern.suggested_focus,
        ))

    # 3. 难度差距补强
    if difficulty_perf.easy.total > 0 and difficulty_perf.easy.rate < 70:
        recommendations.append(QuestionRecommendation(
            type="difficulty_gap",
            priority=8,
            description="加强基础题练习",
            target_concepts=[],
            suggested_difficulty="easy",
            reason=f"简单题正确率 {difficulty_perf.easy.rate:.0f}%，基础需要巩固",
        ))

    # 4. 如果有很多慢速答错，说明理解有问题
    if time_analysis.slow_wrong > time_analysis.fast_wrong:
        recommendations.append(QuestionRecommendation(
            type="weak_concept",
            priority=9,
            description="深入理解核心概念",
            target_concepts=[],
            suggested_difficulty="easy",
            reason="答题时间长但仍答错，说明概念理解不够深入",
        ))

    # 按优先级排序
    return sorted(recommendations, key=lambda x: x.priority, reverse=True)


def generate_learner_profile_prompt(profile: LearnerProfile) -> str:
    """
    生成给 AI 的用户画像描述
    用于智能出题时让 AI 了解用户的学习情况
    """
    if profile.total_questions == 0:
        return "这是用户首次在该知识点练习，暂无历史数据。请出一些基础题目帮助用户建立信心。"

    parts = []

    # 基础统计
    parts.append("## 用户学习画像分析")
    parts.append(f"- 已做题目: {profile.total_questions} 道")
    parts.append(f"- 总体正确率: {profile.correct_rate:.0f}%")
    parts.append(f"- 平均答题时间: {profile.avg_time_per_question:.0f} 秒")

    # 薄弱点
    if profile.weak_concepts:
        parts.append("\n### ⚠️ 薄弱概念（重点出题）")
        for weak in profile.weak_concepts[:5]:
            parts.append(f"- 「{weak.concept}」正确率 {weak.mastery_rate:.0f}%，做过 {weak.total_attempts} 题")

    # 掌握良好的概念
    if profile.strong_concepts:
        parts.append("\n### ✅ 已掌握概念（可少出或提高难度）")
        for strong in profile.strong_concepts[:3]:
            parts.append(f"- 「{strong.concept}」正确率 {strong.mastery_rate:.0f}%")

    # 难度表现
    parts.append("\n### 难度表现")
    dp = profile.difficulty_performance
    if dp.easy.total > 0:
        parts.append(f"- 简单题: {dp.easy.rate:.0f}% 正确率 ({dp.easy.total}题)")
    if dp.medium.total > 0:
        parts.append(f"- 中等题: {dp.medium.rate:.0f}% 正确率 ({dp.medium.total}题)")
    if dp.hard.total > 0:
        parts.append(f"- 困难题: {dp.hard.rate:.0f}% 正确率 ({dp.hard.total}题)")

    # 时间分析洞察
    ta = profile.time_analysis
    if ta.fast_wrong > 2:
        parts.append("\n### ⚡ 行为洞察")
        parts.append(f"- 用户有 {ta.fast_wrong} 次快速答错，可能存在粗心或概念混淆")
    if ta.slow_wrong > ta.slow_correct and ta.slow_wrong > 2:
        parts.append(f"- 用户思考较久仍答错 {ta.slow_wrong} 次，说明某些概念理解不够深入")

    # 出题建议
    if profile.recommendations:
        parts.append("\n### 📋 出题建议")
        for rec in profile.recommendations[:3]:
            parts.append(f"- [优先级{rec.priority}] {rec.description}: {rec.reason}")

    return "\n".join(parts)


def adjust_difficulty_by_profile(
    profile: LearnerProfile,
    original_easy: int,
    original_medium: int,
    original_hard: int
) -> tuple[int, int, int]:
    """
    根据用户画像调整难度分布
    返回 (easy, medium, hard) 百分比
    """
    if profile.total_questions < 5:
        return original_easy, original_medium, original_hard

    dp = profile.difficulty_performance
    new_easy = original_easy
    new_medium = original_medium
    new_hard = original_hard

    # 如果简单题正确率低，增加简单题比例
    if dp.easy.total >= 3 and dp.easy.rate < 60:
        new_easy = min(60, new_easy + 20)
        new_medium = max(30, new_medium - 10)
        new_hard = max(10, new_hard - 10)
    # 如果简单题正确率很高，减少简单题
    elif dp.easy.total >= 3 and dp.easy.rate > 90:
        new_easy = max(10, new_easy - 10)
        new_medium = min(60, new_medium + 5)
        new_hard = min(30, new_hard + 5)

    # 如果困难题正确率高，可以增加困难题
    if dp.hard.total >= 2 and dp.hard.rate > 70:
        new_hard = min(40, new_hard + 10)
        new_easy = max(10, new_easy - 10)

    # 确保总和为100
    total = new_easy + new_medium + new_hard
    return (
        round(new_easy / total * 100),
        round(new_medium / total * 100),
        round(new_hard / total * 100),
    )
