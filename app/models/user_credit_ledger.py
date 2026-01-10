import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint, Index
from sqlalchemy.orm import relationship

from app.core.database import Base


class UserCreditLedger(Base):
    __tablename__ = "user_credit_ledger"

    id = Column(String(191), primary_key=True, default=lambda: str(uuid.uuid4()))
    userId = Column(String(191), ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)

    kind = Column(String(64), nullable=False)
    delta = Column(Integer, nullable=False)
    balanceAfter = Column(Integer, nullable=False)

    requestId = Column(String(191), nullable=True)
    model = Column(String(255), nullable=True)
    promptTokens = Column(Integer, nullable=True)
    completionTokens = Column(Integer, nullable=True)
    totalTokens = Column(Integer, nullable=True)
    costRmbMilli = Column(Integer, nullable=True)
    meta = Column(JSON, nullable=True)

    prevSig = Column(String(64), nullable=False, default="")
    sig = Column(String(64), nullable=False, default="")

    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="creditLedger", overlaps="ledger")
    account = relationship(
        "UserCreditAccount",
        back_populates="ledger",
        primaryjoin="UserCreditLedger.userId == UserCreditAccount.userId",
        foreign_keys=[userId],
        overlaps="creditLedger,user",
    )

    __table_args__ = (
        UniqueConstraint("userId", "requestId", "kind", name="uniq_user_credit_ledger_request_kind"),
        Index("idx_user_credit_ledger_user_created", "userId", "createdAt"),
    )

    def __repr__(self):
        return f"<UserCreditLedger userId={self.userId} kind={self.kind} delta={self.delta}>"
