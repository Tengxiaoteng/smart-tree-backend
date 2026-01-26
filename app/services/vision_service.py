"""
Vision Service - 使用 LangChain 调用 DashScope qwen-vl-plus 进行图片分析
"""
import base64
import json
import re
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class VisionAnalysisResult:
    """视觉分析结果"""
    success: bool
    content: Optional[str] = None
    summary: Optional[str] = None
    concepts: Optional[list[str]] = None
    error: Optional[str] = None


def _extract_json(text: str) -> dict | None:
    """从文本中提取 JSON"""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


async def analyze_image_with_vision_model(
    api_key: str,
    base_url: str,
    model_id: str,
    image_content: bytes,
    mime_type: str,
    custom_prompt: Optional[str] = None,
) -> VisionAnalysisResult:
    """
    使用视觉模型分析图片

    Args:
        api_key: DashScope API Key
        base_url: DashScope Base URL
        model_id: 模型 ID (qwen-vl-plus)
        image_content: 图片二进制内容
        mime_type: 图片 MIME 类型
        custom_prompt: 自定义提示词

    Returns:
        VisionAnalysisResult
    """
    try:
        # 确保 base_url 正确格式化
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        print(f"[VisionService] 初始化 LangChain ChatOpenAI: model={model_id}, base_url={base_url}")

        # 使用 LangChain ChatOpenAI（DashScope 兼容 OpenAI API）
        llm = ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
            max_tokens=2000,
            timeout=120.0,
        )

        # Base64 编码图片
        b64_image = base64.b64encode(image_content).decode("utf-8")
        image_url = f"data:{mime_type};base64,{b64_image}"

        # 构建系统提示
        system_prompt = (
            "你是学习资料分析助手。请先尽量逐字转写图片中的文字/公式。"
            "数学公式必须使用 LaTeX，并用 $...$（行内）或 $$...$$（独立成行/居中展示）包裹；"
            "不要使用 \\(\\)/\\[\\] 作为分隔符；不要用 Markdown 代码块包裹公式。"
            "再用 Markdown 给出解释与学习要点。"
            "返回严格 JSON：{\"content\": \"Markdown，包含『识别原文/公式』与『解释与要点』\", "
            "\"summary\": \"一句话摘要\", \"concepts\": [\"关键概念1\", \"关键概念2\"]}"
        )

        user_prompt = custom_prompt or "请分析这张图片的内容，提取学习相关的关键信息。"

        # 构建消息（使用 OpenAI 兼容的 content 格式）
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": user_prompt},
            ])
        ]

        print(f"[VisionService] 开始调用视觉模型...")

        # 调用模型
        response = await llm.ainvoke(messages)
        result_text = response.content

        print(f"[VisionService] 模型返回: {result_text[:200] if result_text else 'None'}...")

        # 解析 JSON 结果
        parsed = _extract_json(result_text)
        if parsed:
            return VisionAnalysisResult(
                success=True,
                content=parsed.get("content"),
                summary=parsed.get("summary"),
                concepts=parsed.get("concepts", []),
            )
        else:
            # JSON 解析失败，返回原始文本
            return VisionAnalysisResult(
                success=True,
                content=result_text,
                summary=None,
                concepts=[],
            )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[VisionService] 图片分析失败: {error_msg}")
        return VisionAnalysisResult(
            success=False,
            error=error_msg,
        )


async def analyze_pdf_with_vision_model(
    api_key: str,
    base_url: str,
    model_id: str,
    pdf_content: bytes,
    filename: str,
    custom_prompt: Optional[str] = None,
) -> VisionAnalysisResult:
    """
    使用视觉模型分析 PDF

    注意：qwen-vl-plus 对 PDF 的支持有限，可能需要先转换为图片
    """
    try:
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        print(f"[VisionService] 初始化 PDF 分析: model={model_id}")

        llm = ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
            max_tokens=2000,
            timeout=120.0,
        )

        # Base64 编码 PDF
        b64_pdf = base64.b64encode(pdf_content).decode("utf-8")

        system_prompt = (
            "你是学习资料分析助手。请尽量逐字提取/整理 PDF 中的文字与数学公式。"
            "数学公式必须使用 LaTeX，并用 $...$（行内）或 $$...$$（独立成行/居中展示）包裹；"
            "不要使用 \\(\\)/\\[\\] 作为分隔符；不要用 Markdown 代码块包裹公式。"
            "请用 Markdown 输出结构化学习笔记。"
            "返回严格 JSON：{\"content\": \"Markdown，包含『原文/公式』与『解释与要点』\", "
            "\"summary\": \"一句话摘要\", \"concepts\": [\"关键概念1\", \"关键概念2\"]}"
        )

        user_prompt = custom_prompt or f"请分析这个 PDF 文档（{filename}）的内容，提取学习相关的关键信息。"

        # 尝试使用 file 类型（DashScope 特定格式）
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "file", "file": {"url": f"data:application/pdf;base64,{b64_pdf}"}},
                {"type": "text", "text": user_prompt},
            ])
        ]

        print(f"[VisionService] 开始调用视觉模型分析 PDF...")

        response = await llm.ainvoke(messages)
        result_text = response.content

        print(f"[VisionService] 模型返回: {result_text[:200] if result_text else 'None'}...")

        parsed = _extract_json(result_text)
        if parsed:
            return VisionAnalysisResult(
                success=True,
                content=parsed.get("content"),
                summary=parsed.get("summary"),
                concepts=parsed.get("concepts", []),
            )
        else:
            return VisionAnalysisResult(
                success=True,
                content=result_text,
                summary=None,
                concepts=[],
            )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[VisionService] PDF 分析失败: {error_msg}")
        return VisionAnalysisResult(
            success=False,
            error=error_msg,
        )
