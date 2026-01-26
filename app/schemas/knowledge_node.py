from pydantic import BaseModel, ConfigDict
from typing import Any, Optional, List
from datetime import datetime
from enum import Enum


class DifficultyEnum(str, Enum):
    """知识节点难度等级"""
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class ConfidenceLevelEnum(str, Enum):
    """学习信心等级"""
    low = "low"
    medium = "medium"
    high = "high"


class LearningStatusSchema(BaseModel):
    """学习状态跟踪"""
    lastStudied: Optional[datetime] = None
    nextReviewDate: Optional[datetime] = None
    reviewCount: int = 0
    confidenceLevel: ConfidenceLevelEnum = ConfidenceLevelEnum.low


class KnowledgeNodeCreate(BaseModel):
    """创建节点的请求"""
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    topicId: Optional[str] = None
    parentId: Optional[str] = None
    knowledgeType: Optional[str] = "concept"
    difficulty: Optional[DifficultyEnum] = DifficultyEnum.beginner
    estimatedMinutes: Optional[int] = 15
    learningObjectives: Optional[List[str]] = None
    keyConcepts: Optional[List[str]] = None
    prerequisites: Optional[List[str]] = None
    questionPatterns: Optional[List[str]] = None
    commonMistakes: Optional[List[str]] = None
    children: Optional[List[str]] = None
    materialIds: Optional[List[str]] = None
    questionIds: Optional[List[str]] = None
    userNotes: Optional[List[str]] = None
    aiInferredGoals: Optional[Any] = None
    source: Optional[str] = "manual"
    mastery: Optional[int] = 0
    questionCount: Optional[int] = 0
    correctCount: Optional[int] = 0
    learningStatus: Optional[LearningStatusSchema] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None


class KnowledgeNodeUpdate(BaseModel):
    """更新节点的请求"""
    name: Optional[str] = None
    description: Optional[str] = None
    topicId: Optional[str] = None
    parentId: Optional[str] = None
    knowledgeType: Optional[str] = None
    difficulty: Optional[DifficultyEnum] = None
    estimatedMinutes: Optional[int] = None
    learningObjectives: Optional[List[str]] = None
    keyConcepts: Optional[List[str]] = None
    prerequisites: Optional[List[str]] = None
    questionPatterns: Optional[List[str]] = None
    commonMistakes: Optional[List[str]] = None
    children: Optional[List[str]] = None
    materialIds: Optional[List[str]] = None
    questionIds: Optional[List[str]] = None
    userNotes: Optional[List[str]] = None
    aiInferredGoals: Optional[Any] = None
    source: Optional[str] = None
    mastery: Optional[int] = None
    questionCount: Optional[int] = None
    correctCount: Optional[int] = None
    learningStatus: Optional[LearningStatusSchema] = None


class KnowledgeNodeResponse(BaseModel):
    """节点响应"""
    id: str
    userId: str
    topicId: Optional[str]
    name: str
    description: Optional[str]
    knowledgeType: str
    difficulty: str
    estimatedMinutes: int
    learningObjectives: Optional[List[str]]
    keyConcepts: Optional[List[str]]
    prerequisites: Optional[List[str]]
    questionPatterns: Optional[List[str]]
    commonMistakes: Optional[List[str]]
    children: Optional[List[str]]
    materialIds: Optional[List[str]]
    questionIds: Optional[List[str]]
    userNotes: Optional[List[str]]
    aiInferredGoals: Optional[Any]
    source: str
    parentId: Optional[str]
    mastery: int
    questionCount: int
    correctCount: int
    learningStatus: Optional[LearningStatusSchema] = None
    createdAt: datetime
    updatedAt: datetime

    model_config = ConfigDict(from_attributes=True)
