import hmac
import json
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import UserCreditAccount, UserCreditLedger


@contextmanager
def _tx(db: Session):
    # SQLAlchemy 2.x 默认 autobegin：路由里只要 query 过就已经在事务中。
    # 这里用 begin_nested 兼容“事务中再写账本”的场景（MySQL/SQLite 均支持 SAVEPOINT）。
    if db.in_transaction():
        with db.begin_nested():
            yield
    else:
        with db.begin():
            yield


def _credits_hmac_secret() -> str:
    configured = getattr(settings, "CREDITS_HMAC_SECRET", None)
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return settings.JWT_SECRET


def _hmac_sha256_hex(message: str) -> str:
    secret = _credits_hmac_secret().encode("utf-8")
    payload = message.encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def _stable_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _ledger_sig_payload(ledger: UserCreditLedger) -> str:
    return "|".join(
        [
            ledger.userId,
            ledger.kind,
            str(ledger.delta),
            str(ledger.balanceAfter),
            ledger.requestId or "",
            ledger.model or "",
            str(ledger.promptTokens or ""),
            str(ledger.completionTokens or ""),
            str(ledger.totalTokens or ""),
            str(ledger.costRmbMilli or ""),
            _stable_json(ledger.meta),
            ledger.prevSig or "",
            ledger.createdAt.isoformat(),
        ]
    )


def _sign_ledger_entry(ledger: UserCreditLedger) -> str:
    return _hmac_sha256_hex(_ledger_sig_payload(ledger))


def _lock_account_row(db: Session, user_id: str) -> Optional[UserCreditAccount]:
    stmt = select(UserCreditAccount).where(UserCreditAccount.userId == user_id)
    # MySQL 支持 FOR UPDATE；SQLite 会忽略但仍在事务内串行化写入。
    try:
        stmt = stmt.with_for_update()
    except Exception:
        pass
    return db.execute(stmt).scalar_one_or_none()


def _ensure_account(db: Session, user_id: str) -> UserCreditAccount:
    account = _lock_account_row(db, user_id)
    if account:
        return account

    account = UserCreditAccount(userId=user_id, balance=0, lastLedgerSig="")
    db.add(account)
    db.flush()
    return account


def _append_ledger(
    db: Session,
    *,
    account: UserCreditAccount,
    kind: str,
    delta: int,
    request_id: str | None = None,
    model: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_rmb_milli: int | None = None,
    meta: dict | None = None,
) -> UserCreditLedger:
    # 去掉微秒，确保数据库存储后签名一致（MySQL datetime 精度问题）
    now = datetime.utcnow().replace(microsecond=0)
    next_balance = account.balance + int(delta)
    if next_balance < 0:
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="积分不足")

    ledger = UserCreditLedger(
        userId=account.userId,
        kind=kind,
        delta=int(delta),
        balanceAfter=next_balance,
        requestId=request_id,
        model=model,
        promptTokens=prompt_tokens,
        completionTokens=completion_tokens,
        totalTokens=total_tokens,
        costRmbMilli=cost_rmb_milli,
        meta=meta,
        prevSig=account.lastLedgerSig or "",
        sig="",
        createdAt=now,
    )
    ledger.sig = _sign_ledger_entry(ledger)

    db.add(ledger)
    account.balance = next_balance
    account.lastLedgerSig = ledger.sig
    db.add(account)
    db.flush()
    return ledger


def get_or_create_account(db: Session, user_id: str) -> UserCreditAccount:
    with _tx(db):
        return _ensure_account(db, user_id)


def grant_signup_bonus(db: Session, user_id: str) -> UserCreditLedger:
    bonus = int(getattr(settings, "NEW_USER_BONUS_POINTS", 2000))
    with _tx(db):
        account = _ensure_account(db, user_id)
        return _append_ledger(db, account=account, kind="signup_bonus", delta=bonus, meta={"source": "register"})


def get_balance(db: Session, user_id: str, *, verify: bool = True) -> int:
    account = db.query(UserCreditAccount).filter(UserCreditAccount.userId == user_id).first()
    if not account:
        return 0
    if verify:
        verify_user_ledger(db, user_id)
        account = db.query(UserCreditAccount).filter(UserCreditAccount.userId == user_id).first()
        if not account:
            return 0
    return int(account.balance or 0)


def verify_user_ledger(db: Session, user_id: str) -> None:
    account = db.query(UserCreditAccount).filter(UserCreditAccount.userId == user_id).first()
    if not account:
        return

    entries: list[UserCreditLedger] = db.query(UserCreditLedger).filter(UserCreditLedger.userId == user_id).all()
    by_sig: dict[str, UserCreditLedger] = {row.sig: row for row in entries if row.sig}

    head_sig = account.lastLedgerSig or ""
    if not head_sig:
        if entries:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="积分账本校验失败（账户无头签名但存在流水），请联系管理员",
            )
        if int(account.balance or 0) != 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="积分账本校验失败（账户余额异常），请联系管理员",
            )
        return

    seen: set[str] = set()
    expected_balance = int(account.balance or 0)
    current_sig = head_sig

    while current_sig:
        if current_sig in seen:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="积分账本校验失败（链存在循环），请联系管理员",
            )
        seen.add(current_sig)

        entry = by_sig.get(current_sig)
        if not entry:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="积分账本校验失败（缺失流水），请联系管理员",
            )

        expected_sig = _sign_ledger_entry(entry)
        if (entry.sig or "") != expected_sig:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="积分账本校验失败（签名异常），请联系管理员",
            )

        if int(entry.balanceAfter) != expected_balance:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="积分账本校验失败（余额不一致），请联系管理员",
            )

        expected_balance = expected_balance - int(entry.delta)
        current_sig = entry.prevSig or ""

    if expected_balance != 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="积分账本校验失败（起始余额异常），请联系管理员",
        )

    if len(seen) != len(entries):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="积分账本校验失败（存在游离流水），请联系管理员",
        )


@dataclass(frozen=True)
class Reservation:
    request_id: str
    reserved_points: int


def reserve_points(db: Session, user_id: str, *, request_id: str, points: int, meta: dict | None = None) -> Reservation:
    with _tx(db):
        account = _ensure_account(db, user_id)
        existing = (
            db.query(UserCreditLedger)
            .filter(
                UserCreditLedger.userId == user_id,
                UserCreditLedger.requestId == request_id,
                UserCreditLedger.kind == "llm_reserve",
            )
            .first()
        )
        if existing:
            return Reservation(request_id=request_id, reserved_points=abs(int(existing.delta)))

        _append_ledger(
            db,
            account=account,
            kind="llm_reserve",
            delta=-abs(int(points)),
            request_id=request_id,
            meta=meta,
        )
        return Reservation(request_id=request_id, reserved_points=abs(int(points)))


def finalize_reservation(
    db: Session,
    user_id: str,
    *,
    request_id: str,
    reserved_points: int,
    actual_points: int,
    model: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_rmb_milli: int | None = None,
    meta: dict | None = None,
) -> None:
    if actual_points < 0:
        actual_points = 0

    delta = int(reserved_points) - int(actual_points)
    if delta == 0:
        # 仍然记录一次 finalize，便于链路对账与“无差额”显式落账
        delta = 0

    with _tx(db):
        existing = (
            db.query(UserCreditLedger)
            .filter(
                UserCreditLedger.userId == user_id,
                UserCreditLedger.requestId == request_id,
                UserCreditLedger.kind == "llm_finalize",
            )
            .first()
        )
        if existing:
            return

        account = _ensure_account(db, user_id)
        _append_ledger(
            db,
            account=account,
            kind="llm_finalize",
            delta=delta,
            request_id=request_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_rmb_milli=cost_rmb_milli,
            meta=meta,
        )


def refund_stale_reservations(db: Session, user_id: str, *, max_age_minutes: int = 60) -> int:
    cutoff = datetime.utcnow() - timedelta(minutes=max_age_minutes)
    reserves: list[UserCreditLedger] = (
        db.query(UserCreditLedger)
        .filter(
            UserCreditLedger.userId == user_id,
            UserCreditLedger.kind == "llm_reserve",
            UserCreditLedger.createdAt < cutoff,
        )
        .all()
    )

    if not reserves:
        return 0

    refunded = 0
    for reserve in reserves:
        request_id = reserve.requestId
        if not request_id:
            continue
        finalized = (
            db.query(UserCreditLedger)
            .filter(
                UserCreditLedger.userId == user_id,
                UserCreditLedger.requestId == request_id,
                UserCreditLedger.kind == "llm_finalize",
            )
            .first()
        )
        if finalized:
            continue
        reserved_points = abs(int(reserve.delta))
        try:
            finalize_reservation(
                db,
                user_id,
                request_id=request_id,
                reserved_points=reserved_points,
                actual_points=0,
                meta={"autoRefund": True, "reason": "stale_reservation"},
            )
            refunded += reserved_points
        except HTTPException:
            # ignore, ledger itself might be corrupted; verification will surface elsewhere
            continue

    return refunded
