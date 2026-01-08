from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class LLMChatRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096

    # 可选：前端显式选择“使用官方额度（扣积分）/使用自己的配置（BYOK）”
    useSystemKey: Optional[bool] = None

    # 可选：当用户未选择“记住”时，由前端临时传入（不会写库）
    apiKey: Optional[str] = None
    modelId: Optional[str] = None
    baseUrl: Optional[str] = None

    # 可选：DashScope/Qwen 自动路由（用 Flash 做意图识别后选择 Plus/Max/Coder）
    routing: Optional[Literal["auto", "manual"]] = None

    # 可选：幂等请求 ID（用于重复提交/断线重试时避免重复扣费）
    requestId: Optional[str] = None

