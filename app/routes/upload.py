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
from app.services.llm_context import resolve_llm_config, system_llm_available

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

    # 上传到 OSS，获取 key
    folder = get_oss_path(OSSPath.MATERIAL_IMAGES, current_user.id)
    oss_key = upload_file_return_key(content, folder, file.filename or "image.jpg")

    # 保存文件元数据
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

    # 返回代理 URL
    proxy_url = f"/api/files/{file_id}"

    response = MaterialUploadResponse(
        url=proxy_url,
        filename=file.filename or "image.jpg",
        fileSize=len(content),
        type="image",
    )

    # AI 分析模式
    if should_analyze:
        # 使用统一的 LLM 配置解析（支持系统模式和用户自定义）
        try:
            resolved = resolve_llm_config(
                db,
                user_id=current_user.id,
                requested_use_system=None,  # 自动判断
                override_api_key=None,
                override_base_url=None,
                override_model_id=None,
                override_routing=None,
            )
            api_key = resolved.api_key
            base_url = resolved.base_url
            # 图片分析需要视觉模型，系统模式使用 qwen-vl-plus
            model_id = resolved.model_id or ("qwen-vl-plus" if resolved.mode == "system" else None)
            if not model_id:
                raise HTTPException(status_code=400, detail="请先在「API 设置」中配置 modelId")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"API 配置错误: {str(e)}")

        # 构建图片识别请求
        b64_image = base64.b64encode(content).decode("utf-8")
        mime_type = file.content_type or "image/jpeg"
        messages = [
            {"role": "system", "content": "你是学习资料分析助手。请先尽量逐字转写图片中的文字/公式。数学公式必须使用 LaTeX，并用 $...$（行内）或 $$...$$（独立成行/居中展示）包裹；不要使用 \\(\\)/\\[\\] 作为分隔符；不要用 Markdown 代码块包裹公式。再用 Markdown 给出解释与学习要点。返回严格 JSON：{\"content\": \"Markdown，包含『识别原文/公式』与『解释与要点』\", \"summary\": \"一句话摘要\", \"concepts\": [\"关键概念1\", \"关键概念2\"]}"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}},
                {"type": "text", "text": "请分析这张图片的内容，提取学习相关的关键信息。"}
            ]}
        ]
        try:
            print(f"[AI分析] 开始分析图片，模型: {model_id}, 模式: {resolved.mode}")
            result = await _call_llm(api_key, model_id, base_url, messages)
            print(f"[AI分析] LLM 返回: {result[:200] if result else 'None'}...")
            parsed = _extract_json(result)
            if parsed:
                response.aiContent = parsed.get("content")
                response.aiSummary = parsed.get("summary")
                response.extractedConcepts = parsed.get("concepts", [])
                print(f"[AI分析] 解析成功: summary={response.aiSummary}")
            else:
                response.aiContent = result
                print(f"[AI分析] JSON解析失败，使用原始结果")
        except Exception as e:
            print(f"[AI分析] 图片分析失败: {type(e).__name__}: {e}")

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

    # AI 分析模式（PDF 需要支持文档解析的模型，如 GPT-4V 或专门的 PDF 解析服务）
    if ai_analyze:
        # 使用统一的 LLM 配置解析（支持系统模式和用户自定义）
        try:
            resolved = resolve_llm_config(
                db,
                user_id=current_user.id,
                requested_use_system=None,  # 自动判断
                override_api_key=None,
                override_base_url=None,
                override_model_id=None,
                override_routing=None,
            )
            api_key = resolved.api_key
            base_url = resolved.base_url
            # PDF 分析需要视觉模型，系统模式使用 qwen-vl-plus
            model_id = resolved.model_id or ("qwen-vl-plus" if resolved.mode == "system" else None)
            if not model_id:
                raise HTTPException(status_code=400, detail="请先在「API 设置」中配置 modelId")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"API 配置错误: {str(e)}")

        # 尝试用 base64 发送 PDF（部分模型支持）
        b64_pdf = base64.b64encode(content).decode("utf-8")
        messages = [
            {"role": "system", "content": "你是学习资料分析助手。请尽量逐字提取/整理 PDF 中的文字与数学公式。数学公式必须使用 LaTeX，并用 $...$（行内）或 $$...$$（独立成行/居中展示）包裹；不要使用 \\(\\)/\\[\\] 作为分隔符；不要用 Markdown 代码块包裹公式。请用 Markdown 输出结构化学习笔记。返回严格 JSON：{\"content\": \"Markdown，包含『原文/公式』与『解释与要点』\", \"summary\": \"一句话摘要\", \"concepts\": [\"关键概念1\", \"关键概念2\"]}"},
            {"role": "user", "content": [
                {"type": "file", "file": {"url": f"data:application/pdf;base64,{b64_pdf}"}},
                {"type": "text", "text": f"请分析这个 PDF 文档（{file.filename}）的内容，提取学习相关的关键信息。"}
            ]}
        ]
        try:
            print(f"[AI分析] 开始分析PDF，模型: {model_id}, 模式: {resolved.mode}")
            result = await _call_llm(api_key, model_id, base_url, messages)
            parsed = _extract_json(result)
            if parsed:
                response.aiContent = parsed.get("content")
                response.aiSummary = parsed.get("summary")
                response.extractedConcepts = parsed.get("concepts", [])
            else:
                response.aiContent = result
        except Exception as e:
            print(f"[AI分析] PDF分析失败: {type(e).__name__}: {e}")
            pass  # AI 分析失败不影响上传

    return response
