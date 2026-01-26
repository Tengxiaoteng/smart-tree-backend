from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User, UserCreditLedger
from app.schemas.credits import CreditsBalanceResponse, CreditsLedgerItem, CreditsLedgerResponse
from app.services.credits import get_balance, refund_stale_reservations

router = APIRouter()


@router.get("", response_model=CreditsBalanceResponse)
async def get_my_credits(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    refund_stale_reservations(db, current_user.id)
    balance = get_balance(db, current_user.id, verify=True)
    return CreditsBalanceResponse(
        balance=balance,
        pointsPerRmb=int(getattr(settings, "POINTS_PER_RMB", 1000)),
        newUserBonus=int(getattr(settings, "NEW_USER_BONUS_POINTS", 2000)),
    )


@router.get("/ledger", response_model=CreditsLedgerResponse)
async def get_my_credits_ledger(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows: list[UserCreditLedger] = (
        db.query(UserCreditLedger)
        .filter(UserCreditLedger.userId == current_user.id)
        .order_by(UserCreditLedger.createdAt.desc(), UserCreditLedger.id.desc())
        .limit(limit)
        .all()
    )

    return CreditsLedgerResponse(
        items=[
            CreditsLedgerItem(
                id=row.id,
                kind=row.kind,
                delta=int(row.delta),
                balanceAfter=int(row.balanceAfter),
                requestId=row.requestId,
                model=None,
                promptTokens=row.promptTokens,
                completionTokens=row.completionTokens,
                totalTokens=row.totalTokens,
                costRmbMilli=row.costRmbMilli,
                meta=None,
                createdAt=row.createdAt,
            )
            for row in rows
        ]
    )
