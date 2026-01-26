from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class BatchChatItem(BaseModel):
    customId: Optional[str] = None
    messages: list[dict[str, Any]] = Field(default_factory=list)


class BatchChatCreateRequest(BaseModel):
    # 任务配置
    model: str
    completionWindow: str = "24h"
    endpoint: Optional[str] = None  # 默认按 model 推断：batch-test-model -> /v1/chat/ds-test，否则 /v1/chat/completions
    metadata: Optional[dict[str, Any]] = None

    # 请求参数（同一文件内需保持一致）
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None

    # 输入（两种方式二选一）
    items: list[BatchChatItem] = Field(default_factory=list)
    prompts: Optional[list[str]] = None
    systemPrompt: str = "You are a helpful assistant."

    # 认证模式（BYOK vs 系统额度）
    useSystemKey: Optional[bool] = None
    apiKey: Optional[str] = None
    baseUrl: Optional[str] = None


class BatchJobResponse(BaseModel):
    id: str
    status: str
    batchId: str
    inputFileId: str
    outputFileId: Optional[str] = None
    errorFileId: Optional[str] = None
    model: str
    endpoint: str
    completionWindow: str
    mode: str
    createdAt: datetime
    updatedAt: datetime
    reservedPoints: Optional[int] = None
    chargedPoints: Optional[int] = None
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    totalTokens: Optional[int] = None
    costRmbMilli: Optional[int] = None
    billedAt: Optional[datetime] = None
    requestCounts: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class BatchJobListResponse(BaseModel):
    items: list[BatchJobResponse]

