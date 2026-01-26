from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User, AnswerRecord, Question, KnowledgeNode
from app.schemas.answer_record import AnswerRecordCreate, AnswerRecordResponse, AnswerRecordUpdate

router = APIRouter()


def _format_user_answer_for_db(answer: object | None) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, list):
        import json

        return json.dumps(answer, ensure_ascii=False)
    if isinstance(answer, str):
        return answer
    return str(answer)


@router.get("", response_model=list[AnswerRecordResponse])
async def get_answer_records(
    nodeId: str = Query(None),
    questionId: str = Query(None),
    topicId: str = Query(None),
    includeNoTopic: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取答题记录列表"""
    query = db.query(AnswerRecord).filter(AnswerRecord.userId == current_user.id)
    if nodeId:
        query = query.filter(AnswerRecord.nodeId == nodeId)
    if questionId:
        query = query.filter(AnswerRecord.questionId == questionId)
    if topicId:
        query = query.join(KnowledgeNode, KnowledgeNode.id == AnswerRecord.nodeId).filter(
            KnowledgeNode.userId == current_user.id
        )
        if includeNoTopic:
            query = query.filter(or_(KnowledgeNode.topicId == topicId, KnowledgeNode.topicId.is_(None)))
        else:
            query = query.filter(KnowledgeNode.topicId == topicId)
    return query.order_by(AnswerRecord.createdAt.desc()).all()


@router.post("", response_model=AnswerRecordResponse)
async def create_answer_record(
    data: AnswerRecordCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建答题记录"""
    if data.id:
        if len(data.id) > 191:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="记录 ID 过长")
        existing = db.query(AnswerRecord).filter(AnswerRecord.id == data.id).first()
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="记录 ID 已存在")

    question = db.query(Question).filter(
        Question.id == data.questionId,
        Question.userId == current_user.id,
    ).first()
    if not question:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="题目不存在")

    node = db.query(KnowledgeNode).filter(
        KnowledgeNode.id == data.nodeId,
        KnowledgeNode.userId == current_user.id,
    ).first()
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识点不存在")

    record_kwargs = {
        "userId": current_user.id,
        "questionId": data.questionId,
        "nodeId": data.nodeId,
        "userAnswer": _format_user_answer_for_db(data.userAnswer),
        "isCorrect": bool(data.isCorrect),
        "timeSpent": data.timeSpent,
    }
    if data.createdAt is not None:
        record_kwargs["createdAt"] = data.createdAt
    if data.id:
        record_kwargs["id"] = data.id

    record = AnswerRecord(**record_kwargs)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.patch("/{record_id}", response_model=AnswerRecordResponse)
async def update_answer_record(
    record_id: str,
    data: AnswerRecordUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新答题记录（主要用于追加追问对话）"""
    record = db.query(AnswerRecord).filter(
        AnswerRecord.id == record_id,
        AnswerRecord.userId == current_user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="答题记录不存在")

    if data.followUpMessages is not None:
        # 将 Pydantic 模型列表转换为可序列化的字典列表
        record.followUpMessages = [msg.model_dump() for msg in data.followUpMessages]

    db.commit()
    db.refresh(record)
    return record
