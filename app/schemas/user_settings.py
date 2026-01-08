from pydantic import BaseModel
from typing import Optional


class UserSettingsUpdate(BaseModel):
    apiKey: Optional[str] = None
    modelId: Optional[str] = None
    baseUrl: Optional[str] = None
    remember: Optional[bool] = None
    useSystemKey: Optional[bool] = None
    routing: Optional[str] = None  # "auto" | "manual"


class UserSettingsResponse(BaseModel):
    apiKey: str
    modelId: str
    baseUrl: str
    remember: bool
    useSystemKey: bool = False
    routing: str = "manual"
