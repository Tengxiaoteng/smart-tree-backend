from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.oss import get_file_stream
from app.core.schema import ensure_user_file_schema
from app.core.security import get_current_user
from app.core.user_files import delete_user_file_record_and_maybe_blob
from app.models import UserFile, User, Material, UserNote

router = APIRouter()


@router.get("/{file_id}")
async def get_file(
    file_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """文件代理接口 - 通过 file_id 获取文件（需认证）"""
    ensure_user_file_schema(db)
    file_record = db.query(UserFile).filter(
        UserFile.id == file_id,
        UserFile.userId == current_user.id,
    ).first()

    if not file_record:
        raise HTTPException(status_code=404, detail="文件不存在")

    content = get_file_stream(file_record.ossPath)
    if not content:
        raise HTTPException(status_code=404, detail="文件不存在")

    # URL 编码文件名以支持中文
    encoded_filename = quote(file_record.filename)

    return Response(
        content=content,
        media_type=file_record.mimeType or "application/octet-stream",
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{encoded_filename}"}
    )


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    force: bool = Query(False, description="强制删除（即使仍被引用）"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除文件 - 同时删除 OSS 文件和数据库记录"""
    ensure_user_file_schema(db)
    file_record = db.query(UserFile).filter(
        UserFile.id == file_id,
        UserFile.userId == current_user.id
    ).first()

    if not file_record:
        raise HTTPException(status_code=404, detail="文件不存在")

    proxy_url = f"/api/files/{file_id}"
    material_refs = db.query(Material).filter(Material.userId == current_user.id, Material.url == proxy_url).count()
    note_refs = (
        db.query(UserNote)
        .filter(UserNote.userId == current_user.id, UserNote.content.contains(proxy_url))
        .count()
    )
    if (material_refs or note_refs) and not force:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "文件仍被引用，拒绝删除",
                "materialRefs": material_refs,
                "noteRefs": note_refs,
                "hint": "如需强制删除，请传 force=true",
            },
        )

    ok = delete_user_file_record_and_maybe_blob(db, file_record)
    if not ok:
        raise HTTPException(status_code=502, detail="OSS 文件删除失败")
    db.commit()

    return {"success": True}
