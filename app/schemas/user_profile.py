from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime


class UserProfileUpdate(BaseModel):
    email: Optional[str] = None
    avatarUrl: Optional[str] = None
    bio: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    education: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    learningHabits: Optional[Dict[str, Any]] = None
    seedlingPortrait: Optional[Dict[str, Any]] = None


class SeedlingRefreshRequest(BaseModel):
    """用于生成/刷新树苗画像的输入快照（可由前端汇总后传入）。"""

    learningSnapshot: Optional[Dict[str, Any]] = None


class UserProfileResponse(BaseModel):
    userId: str
    email: Optional[str] = None
    avatarUrl: Optional[str] = None
    bio: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    education: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    learningHabits: Optional[Dict[str, Any]] = None
    seedlingPortrait: Optional[Dict[str, Any]] = None
    portraitUpdatedAt: Optional[datetime] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
