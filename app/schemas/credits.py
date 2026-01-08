from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class CreditsBalanceResponse(BaseModel):
    balance: int
    currency: str = "points"
    pointsPerRmb: int
    newUserBonus: int


class CreditsLedgerItem(BaseModel):
    id: str
    kind: str
    delta: int
    balanceAfter: int
    requestId: Optional[str] = None
    model: Optional[str] = None
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    totalTokens: Optional[int] = None
    costRmbMilli: Optional[int] = None
    meta: Optional[dict[str, Any]] = None
    createdAt: datetime


class CreditsLedgerResponse(BaseModel):
    items: list[CreditsLedgerItem]

