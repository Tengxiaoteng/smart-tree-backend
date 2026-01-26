import base64
import json
import re
import uuid
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.oss import upload_file, upload_file_return_key, get_oss_path, OSSPath
from app.core.schema import ensure_user_file_schema
from app.models import User, Material, UserSettings, UserFile, FileType
from app.services.llm_context import resolve_llm_config, system_llm_available, get_vision_llm_config, vision_llm_available
from app.services.vision_service import analyze_image_with_vision_model, analyze_pdf_with_vision_model

router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_DOC_TYPES = {"application/pdf"}
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_DOC_SIZE = 20 * 1024 * 1024  # 20MB


class UploadResponse(BaseModel):
    url: str


class MaterialUploadResponse(BaseModel):
    """资料上传响应（包含 OSS URL 和可选的 AI 解析结果）"""
    url: str
    filename: str
    fileSize: int
    type: str  # image / pdf
    # AI 解析结果（仅 AI 模式返回）
    aiContent: Optional[str] = None
    aiSummary: Optional[str] = None
    extractedConcepts: Optional[list] = None


def _normalize_chat_url(base_url: str) -> str:
    normalized = re.sub(r"/+$", "", base_url.strip())
    return normalized if "chat/completions" in normalized else f"{normalized}/chat/completions"


async def _call_llm(api_key: str, model_id: str, base_url: str, messages: list, max_tokens: int = 2000) -> str:
    """调用 OpenAI 兼容的 LLM API"""
    chat_url = _normalize_chat_url(base_url)
    parsed = urlparse(chat_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("baseUrl 协议必须是 http/https")

    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            chat_url,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={"model": model_id, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens},
        )
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"LLM API 请求失败 ({resp.status_code})")
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def _extract_json(content: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ==================== 基础上传端点 ====================

@router.post("/avatar", response_model=UploadResponse)
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """上传用户头像"""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持 JPG/PNG/GIF/WebP 格式")
    content = await file.read()
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="图片大小不能超过 5MB")
    folder = get_oss_path(OSSPath.USER_AVATAR, current_user.id)
    url = upload_file(content, folder, file.filename or "avatar.jpg")
    return UploadResponse(url=url)


@router.post("/topic/cover", response_model=UploadResponse)
async def upload_topic_cover(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """上传主题封面图"""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持 JPG/PNG/GIF/WebP 格式")
    content = await file.read()
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="图片大小不能超过 5MB")
    folder = get_oss_path(OSSPath.TOPIC_COVERS, current_user.id)
    url = upload_file(content, folder, file.filename or "cover.jpg")
    return UploadResponse(url=url)


@router.post("/node/image", response_model=UploadResponse)
async def upload_node_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """上传节点笔记中的图片"""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持 JPG/PNG/GIF/WebP 格式")
    content = await file.read()
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="图片大小不能超过 5MB")

    # 统一用 /api/files/<id> 代理 URL（便于删除节点/笔记时同时清理 OSS 文件）
    folder = get_oss_path(OSSPath.NODE_IMAGES, current_user.id)
    oss_key = upload_file_return_key(content, folder, file.filename or "image.jpg")

    ensure_user_file_schema(db)
    file_id = str(uuid.uuid4())
    user_file = UserFile(
        id=file_id,
        userId=current_user.id,
        ossPath=oss_key,
        filename=file.filename or "image.jpg",
        fileType=FileType.IMAGE,
        fileSize=len(content),
        mimeType=file.content_type,
    )
    db.add(user_file)
    db.commit()

    proxy_url = f"/api/files/{file_id}"
    return UploadResponse(url=proxy_url)


# ==================== 资料上传端点（快速上传 + AI 分析） ====================

@router.post("/material/image", response_model=MaterialUploadResponse)
async def upload_material_image(
    file: UploadFile = File(...),
    ai_analyze: str = Form("false"),  # 改为字符串接收
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    上传图片资料
    - ai_analyze=false: 快速上传，只存文件
    - ai_analyze=true: AI 智能分析，识别图片内容
    """
    # 将字符串转换为布尔值
    should_analyze = ai_analyze.lower() == "true"
    print(f"[上传] ai_analyze={ai_analyze} -> should_analyze={should_analyze}, 文件={file.filename}")
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持 JPG/PNG/GIF/WebP 格式")
    content = await file.read()
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="图片大小不能超过 5MB")

    # 上传到 OSS，获取 key（带重试机制）
    folder = get_oss_path(OSSPath.MATERIAL_IMAGES, current_user.id)
    try:
        print(f"[上传] 开始上传到 OSS: folder={folder}, filename={file.filename}, size={len(content)}")
        oss_key = upload_file_return_key(content, folder, file.filename or "image.jpg")
        print(f"[上传] OSS 上传成功: key={oss_key}")
    except Exception as e:
        print(f"[上传] OSS 上传失败: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件上传失败: {str(e)}"
        )

    # 保存文件元数据
    try:
        ensure_user_file_schema(db)
        file_id = str(uuid.uuid4())
        user_file = UserFile(
            id=file_id,
            userId=current_user.id,
            ossPath=oss_key,
            filename=file.filename or "image.jpg",
            fileType=FileType.IMAGE,
            fileSize=len(content),
            mimeType=file.content_type,
        )
        db.add(user_file)
        db.commit()
        print(f"[上传] 数据库记录保存成功: file_id={file_id}")
    except Exception as e:
        print(f"[上传] 数据库保存失败: {type(e).__name__}: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存文件记录失败: {str(e)}"
        )

    # 返回代理 URL
    proxy_url = f"/api/files/{file_id}"

    response = MaterialUploadResponse(
        url=proxy_url,
        filename=file.filename or "image.jpg",
        fileSize=len(content),
        type="image",
    )

    # AI 分析模式（图片分析需要视觉模型，使用 DashScope qwen-vl-plus）
    if should_analyze:
        # 优先使用系统视觉模型（DashScope qwen-vl-plus）
        vision_config = get_vision_llm_config()
        if vision_config:
            api_key = vision_config.api_key
            base_url = vision_config.base_url
            model_id = vision_config.model_id
            print(f"[AI分析] 使用系统视觉模型: {model_id}")
        else:
            # 回退到用户自定义配置
            try:
                resolved = resolve_llm_config(
                    db,
                    user_id=current_user.id,
                    requested_use_system=None,
                    override_api_key=None,
                    override_base_url=None,
                    override_model_id=None,
                    override_routing=None,
                )
                api_key = resolved.api_key
                base_url = resolved.base_url
                model_id = resolved.model_id
                if not model_id:
                    raise HTTPException(status_code=400, detail="图片分析需要视觉模型，请在「API 设置」中配置支持视觉的 modelId")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"API 配置错误: {str(e)}")

        # 使用 LangChain 视觉服务分析图片
        mime_type = file.content_type or "image/jpeg"
        try:
            print(f"[AI分析] 开始分析图片，模型: {model_id}")
            vision_result = await analyze_image_with_vision_model(
                api_key=api_key,
                base_url=base_url,
                model_id=model_id,
                image_content=content,
                mime_type=mime_type,
            )
            if vision_result.success:
                response.aiContent = vision_result.content
                response.aiSummary = vision_result.summary
                response.extractedConcepts = vision_result.concepts or []
                print(f"[AI分析] 解析成功: summary={response.aiSummary}")
            else:
                print(f"[AI分析] 图片分析失败: {vision_result.error}")
        except Exception as e:
            print(f"[AI分析] 图片分析异常: {type(e).__name__}: {e}")

    return response


@router.post("/material/document", response_model=MaterialUploadResponse)
async def upload_material_document(
    file: UploadFile = File(...),
    ai_analyze: bool = Form(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    上传 PDF 文档
    - ai_analyze=false: 快速上传，只存文件
    - ai_analyze=true: AI 智能分析，提取文档内容（需要支持 PDF 解析的模型）
    """
    if file.content_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持 PDF 格式")
    content = await file.read()
    if len(content) > MAX_DOC_SIZE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件大小不能超过 20MB")

    # 上传到 OSS，获取 key
    folder = get_oss_path(OSSPath.MATERIAL_DOCUMENTS, current_user.id)
    oss_key = upload_file_return_key(content, folder, file.filename or "document.pdf")

    # 保存文件元数据
    ensure_user_file_schema(db)
    file_id = str(uuid.uuid4())
    user_file = UserFile(
        id=file_id,
        userId=current_user.id,
        ossPath=oss_key,
        filename=file.filename or "document.pdf",
        fileType=FileType.PDF,
        fileSize=len(content),
        mimeType=file.content_type,
    )
    db.add(user_file)
    db.commit()

    # 返回代理 URL
    proxy_url = f"/api/files/{file_id}"

    response = MaterialUploadResponse(
        url=proxy_url,
        filename=file.filename or "document.pdf",
        fileSize=len(content),
        type="pdf",
    )

    # AI 分析模式（PDF 需要视觉模型，使用 DashScope qwen-vl-plus）
    if ai_analyze:
        # 优先使用系统视觉模型（DashScope qwen-vl-plus）
        vision_config = get_vision_llm_config()
        if vision_config:
            api_key = vision_config.api_key
            base_url = vision_config.base_url
            model_id = vision_config.model_id
            print(f"[AI分析] 使用系统视觉模型: {model_id}")
        else:
            # 回退到用户自定义配置
            try:
                resolved = resolve_llm_config(
                    db,
                    user_id=current_user.id,
                    requested_use_system=None,
                    override_api_key=None,
                    override_base_url=None,
                    override_model_id=None,
                    override_routing=None,
                )
                api_key = resolved.api_key
                base_url = resolved.base_url
                model_id = resolved.model_id
                if not model_id:
                    raise HTTPException(status_code=400, detail="PDF 分析需要视觉模型，请在「API 设置」中配置支持视觉的 modelId")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"API 配置错误: {str(e)}")

        # 使用 LangChain 视觉服务分析 PDF
        try:
            print(f"[AI分析] 开始分析PDF，模型: {model_id}")
            vision_result = await analyze_pdf_with_vision_model(
                api_key=api_key,
                base_url=base_url,
                model_id=model_id,
                pdf_content=content,
                filename=file.filename or "document.pdf",
            )
            if vision_result.success:
                response.aiContent = vision_result.content
                response.aiSummary = vision_result.summary
                response.extractedConcepts = vision_result.concepts or []
                print(f"[AI分析] PDF解析成功: summary={response.aiSummary}")
            else:
                print(f"[AI分析] PDF分析失败: {vision_result.error}")
        except Exception as e:
            print(f"[AI分析] PDF分析异常: {type(e).__name__}: {e}")

    return response
