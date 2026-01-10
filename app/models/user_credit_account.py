from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.core.database import Base


class UserCreditAccount(Base):
    __tablename__ = "user_credit_account"

    userId = Column(
        String(191),
        ForeignKey("user.id", ondelete="CASCADE"),
        primary_key=True,
    )
    balance = Column(Integer, nullable=False, default=0)
    lastLedgerSig = Column(String(64), nullable=False, default="")

    createdAt = Column(DateTime, default=datetime.utcnow, nullable=False)
    updatedAt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="creditAccount")
    ledger = relationship(
        "UserCreditLedger",
        back_populates="account",
        cascade="all, delete-orphan",
        primaryjoin="UserCreditAccount.userId == UserCreditLedger.userId",
        foreign_keys="UserCreditLedger.userId",
        overlaps="creditLedger",
    )

    def __repr__(self):
        return f"<UserCreditAccount userId={self.userId} balance={self.balance}>"
