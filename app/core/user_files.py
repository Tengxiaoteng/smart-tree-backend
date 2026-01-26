from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.core.oss import (
    delete_file_by_key,
    get_file_stream,
    upload_file_return_key,
    OSSPath,
    get_oss_path,
)
from app.core.schema import ensure_user_file_schema
from app.models import UserFile


logger = logging.getLogger(__name__)

_PROXY_FILE_IN_TEXT_RE = re.compile(r"/api/files/([0-9a-fA-F-]{36})")


def replace_proxy_file_ids_in_text(text: str | None, file_id_map: dict[str, str]) -> str | None:
    if not text or not file_id_map:
        return text

    def _repl(match: re.Match[str]) -> str:
        old_id = match.group(1)
        new_id = file_id_map.get(old_id)
        if not new_id:
            return match.group(0)
        return f"/api/files/{new_id}"

    return _PROXY_FILE_IN_TEXT_RE.sub(_repl, text)


def replace_proxy_file_ids_in_json(value: Any, file_id_map: dict[str, str]) -> Any:
    if not file_id_map:
        return value
    if value is None:
        return None
    if isinstance(value, str):
        return replace_proxy_file_ids_in_text(value, file_id_map) or value
    if isinstance(value, list):
        return [replace_proxy_file_ids_in_json(v, file_id_map) for v in value]
    if isinstance(value, dict):
        return {k: replace_proxy_file_ids_in_json(v, file_id_map) for k, v in value.items()}
    return value


def _get_oss_folder_for_file_type(file_type: str | None, user_id: str) -> str:
    """根据文件类型返回对应的 OSS 目录路径"""
    file_type_lower = (file_type or "").lower()
    if file_type_lower in {"image", "img", "picture"}:
        return get_oss_path(OSSPath.MATERIAL_IMAGES, user_id)
    elif file_type_lower in {"audio", "voice", "sound"}:
        return get_oss_path(OSSPath.MATERIAL_AUDIO, user_id)
    else:
        # 默认使用 documents 目录
        return get_oss_path(OSSPath.MATERIAL_DOCUMENTS, user_id)


def ensure_user_files_copied(
    db: Session,
    source_file_ids: set[str],
    target_user_id: str,
) -> dict[str, str]:
    """
    For a set of source file_ids, create NEW UserFile records for target_user_id
    with PHYSICAL COPIES of the files (new ossPath).

    This ensures complete data isolation between users - modifications or deletions
    by the target user will NOT affect the source user's files.

    Returns a mapping from source file_id to new file_id.
    """
    if not source_file_ids:
        return {}

    ensure_user_file_schema(db)

    source_records = (
        db.query(UserFile)
        .filter(UserFile.id.in_(sorted(source_file_ids)))
        .all()
    )
    if not source_records:
        return {}

    mapping: dict[str, str] = {}
    for record in source_records:
        if not record.ossPath:
            continue

        # 读取源文件内容
        file_content = get_file_stream(record.ossPath)
        if file_content is None:
            logger.warning(
                "无法读取源文件进行复制: file_id=%s, ossPath=%s",
                record.id,
                record.ossPath,
            )
            continue

        # 确定目标目录
        target_folder = _get_oss_folder_for_file_type(record.fileType, target_user_id)

        # 上传为新文件（创建物理副本）
        try:
            new_oss_key = upload_file_return_key(
                file_bytes=file_content,
                folder=target_folder,
                filename=record.filename or "file",
            )
        except Exception as exc:
            logger.error(
                "复制文件到新 OSS 路径失败: file_id=%s, error=%s",
                record.id,
                exc,
            )
            continue

        # 创建新的 UserFile 记录
        new_id = str(uuid.uuid4())
        copied = UserFile(
            id=new_id,
            userId=target_user_id,
            ossPath=new_oss_key,
            filename=record.filename,
            fileType=record.fileType,
            fileSize=record.fileSize,
            mimeType=record.mimeType,
        )
        db.add(copied)
        mapping[record.id] = new_id

    return mapping


def delete_user_file_record_and_maybe_blob(db: Session, file_record: UserFile) -> bool:
    """
    Delete a UserFile record, and delete its underlying blob only if no other UserFile rows reference the same ossPath.
    """
    ensure_user_file_schema(db)

    other_ref = (
        db.query(UserFile.id)
        .filter(UserFile.ossPath == file_record.ossPath, UserFile.id != file_record.id)
        .first()
    )
    if other_ref is None:
        deleted = delete_file_by_key(file_record.ossPath)
        if not deleted:
            return False

    db.delete(file_record)
    return True

