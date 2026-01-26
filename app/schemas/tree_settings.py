from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class DisplayConfig(BaseModel):
    showMastery: bool = True
    showChildCount: bool = False
    showDescription: bool = False


class VisualConfig(BaseModel):
    nodeSpacing: int = Field(default=60, ge=20, le=200)
    nodeSize: Literal["small", "medium", "large"] = "medium"
    lineStyle: Literal["curve", "straight", "polyline"] = "curve"


class ExpandConfig(BaseModel):
    defaultExpandLevel: int = Field(default=3, ge=-1, le=10)


class TreeSettingsBase(BaseModel):
    sortMode: Literal[
        "createdAt_asc",
        "createdAt_desc",
        "name_asc",
        "mastery_asc",
        "mastery_desc"
    ] = "createdAt_desc"
    displayConfig: Optional[DisplayConfig] = None
    visualConfig: Optional[VisualConfig] = None
    expandConfig: Optional[ExpandConfig] = None


class TreeSettingsCreate(TreeSettingsBase):
    topicId: str


class TreeSettingsUpdate(BaseModel):
    sortMode: Optional[Literal[
        "createdAt_asc",
        "createdAt_desc",
        "name_asc",
        "mastery_asc",
        "mastery_desc"
    ]] = None
    displayConfig: Optional[DisplayConfig] = None
    visualConfig: Optional[VisualConfig] = None
    expandConfig: Optional[ExpandConfig] = None


class TreeSettingsResponse(BaseModel):
    id: str
    topicId: str
    sortMode: str
    displayConfig: Optional[dict] = None
    visualConfig: Optional[dict] = None
    expandConfig: Optional[dict] = None
    createdAt: datetime
    updatedAt: datetime

    class Config:
        from_attributes = True
