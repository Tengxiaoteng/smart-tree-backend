import logging
import os
import tempfile
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
import oss2

# 确保加载 .env 文件
load_dotenv()

OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET", "")
OSS_BUCKET_NAME = os.getenv("OSS_BUCKET_NAME", "")
_raw_endpoint = os.getenv("OSS_ENDPOINT", "")
# 确保 endpoint 带有 https:// 前缀（用于 oss2 SDK）
OSS_ENDPOINT = _raw_endpoint if _raw_endpoint.startswith(("http://", "https://")) else f"https://{_raw_endpoint}" if _raw_endpoint else ""
# 不带协议的 endpoint（用于构建公开 URL）
OSS_ENDPOINT_HOST = _raw_endpoint.replace("https://", "").replace("http://", "") if _raw_endpoint else ""

_bucket = None


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


LOCAL_UPLOAD_MIRROR = _env_flag("LOCAL_UPLOAD_MIRROR", "true")
LOCAL_UPLOAD_ROOT = os.getenv("LOCAL_UPLOAD_ROOT", "uploads")
ALLOW_LOCAL_UPLOAD_WITHOUT_OSS = _env_flag("ALLOW_LOCAL_UPLOAD_WITHOUT_OSS", "false")
FORCE_LOCAL_UPLOAD = _env_flag("FORCE_LOCAL_UPLOAD", "false")
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def get_local_upload_root() -> Path:
    raw = (os.getenv("LOCAL_UPLOAD_ROOT", LOCAL_UPLOAD_ROOT) or "uploads").strip()
    base = Path(raw)
    if not base.is_absolute():
        base = _BACKEND_ROOT / base
    return base


def _safe_rel_key(key: str) -> str:
    value = str(key or "").strip().lstrip("/")
    if not value:
        raise ValueError("empty key")

    path = Path(value)
    if path.is_absolute():
        raise ValueError("absolute key not allowed")
    if any(part in {".", ".."} for part in path.parts):
        raise ValueError("invalid key")

    # 统一用 / 作为分隔，避免平台差异
    return "/".join(path.parts)


def get_local_path_for_key(oss_key: str) -> Path | None:
    normalized_key = normalize_oss_key(oss_key)
    if not normalized_key:
        return None
    try:
        safe_key = _safe_rel_key(normalized_key)
    except ValueError:
        return None
    return get_local_upload_root() / safe_key


def save_local_copy(oss_key: str, file_bytes: bytes) -> Path | None:
    """将文件写入本地 uploads 目录（路径与 OSS key 一致），用于编辑时引用/本地兜底。"""
    local_path = get_local_path_for_key(oss_key)
    if local_path is None:
        return None

    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=str(local_path.parent),
            prefix=".tmp_upload_",
        ) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_name = tmp_file.name
        os.replace(tmp_name, str(local_path))
        return local_path
    except Exception:
        logging.exception("写入本地上传文件失败: %s", local_path)
        try:
            if "tmp_name" in locals() and tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass
        return None


def read_local_copy(oss_key: str) -> bytes | None:
    local_path = get_local_path_for_key(oss_key)
    if local_path is None:
        return None
    if not local_path.is_file():
        return None
    try:
        return local_path.read_bytes()
    except Exception:
        logging.exception("读取本地上传文件失败: %s", local_path)
        return None


def delete_local_copy(oss_key: str) -> bool:
    local_path = get_local_path_for_key(oss_key)
    if local_path is None:
        return False
    try:
        if local_path.exists():
            local_path.unlink()
        return True
    except Exception:
        logging.exception("删除本地上传文件失败: %s", local_path)
        return False


def is_oss_configured() -> bool:
    if FORCE_LOCAL_UPLOAD:
        return False
    return bool(OSS_ACCESS_KEY_ID and OSS_ACCESS_KEY_SECRET and OSS_BUCKET_NAME and OSS_ENDPOINT and OSS_ENDPOINT_HOST)


def build_public_url(oss_key: str) -> str:
    normalized_key = normalize_oss_key(oss_key)
    if not normalized_key:
        return ""
    return f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT_HOST}/{normalized_key}"


class OSSPath(str, Enum):
    """OSS 目录路径枚举"""
    # 用户相关
    USER_AVATAR = "users/{user_id}/avatar"
    USER_EXPORTS = "users/{user_id}/exports"

    # 学习资料
    MATERIAL_DOCUMENTS = "materials/{user_id}/documents"
    MATERIAL_IMAGES = "materials/{user_id}/images"
    MATERIAL_AUDIO = "materials/{user_id}/audio"

    # 主题相关
    TOPIC_COVERS = "topics/{user_id}/covers"

    # 知识节点相关
    NODE_IMAGES = "nodes/{user_id}/images"

    # 临时文件
    TEMP = "temp/{user_id}"


def get_bucket() -> oss2.Bucket:
    global _bucket
    if _bucket is None:
        if not is_oss_configured():
            raise ValueError("OSS 未配置（缺少 OSS_ACCESS_KEY_ID/OSS_ACCESS_KEY_SECRET/OSS_BUCKET_NAME/OSS_ENDPOINT）")
        auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
        _bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET_NAME)
    return _bucket


def get_oss_path(path_type: OSSPath, user_id: str) -> str:
    """获取格式化后的 OSS 路径"""
    return path_type.value.format(user_id=user_id)


def upload_file(file_bytes: bytes, folder: str, filename: str) -> str:
    """上传文件到 OSS，返回公开访问 URL"""
    bucket = get_bucket()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    unique_name = f"{uuid.uuid4().hex}_{int(datetime.now().timestamp())}"
    if ext:
        unique_name = f"{unique_name}.{ext}"
    key = f"{folder}/{unique_name}"
    bucket.put_object(key, file_bytes)
    if LOCAL_UPLOAD_MIRROR:
        save_local_copy(key, file_bytes)
    return f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT_HOST}/{key}"


def normalize_oss_key(path_or_url: str | None) -> str | None:
    """将 URL/路径标准化为 OSS key（不含域名）。"""
    if not path_or_url:
        return None

    value = str(path_or_url).strip()
    if not value:
        return None

    # 防止误把应用层代理 URL 当作 OSS key
    if value.startswith("/api/files/") or value.startswith("api/files/"):
        return None

    if value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        # Some clients may URL-encode "/" as "%2F". Unquote to match the real object key.
        key = unquote((parsed.path or "")).lstrip("/")
        return key or None

    key = value.lstrip("/")
    return key or None


def delete_file(url: str) -> bool:
    """根据 URL 删除 OSS 文件"""
    bucket = get_bucket()
    prefix = f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT_HOST}/"
    if not url.startswith(prefix):
        return False
    key = url[len(prefix):]
    bucket.delete_object(key)
    return True


def upload_file_return_key(file_bytes: bytes, folder: str, filename: str, max_retries: int = 3) -> str:
    """上传文件到 OSS，返回 OSS key（不含域名）

    Args:
        file_bytes: 文件内容
        folder: 目标文件夹
        filename: 文件名
        max_retries: 最大重试次数
    """
    import time

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    unique_name = f"{uuid.uuid4().hex}_{int(datetime.now().timestamp())}"
    if ext:
        unique_name = f"{unique_name}.{ext}"
    key = f"{folder}/{unique_name}"

    if is_oss_configured():
        bucket = get_bucket()
        last_error = None

        for attempt in range(max_retries):
            try:
                bucket.put_object(key, file_bytes)
                if LOCAL_UPLOAD_MIRROR:
                    save_local_copy(key, file_bytes)
                logging.info(f"[OSS] 上传成功: {key} (尝试 {attempt + 1}/{max_retries})")
                return key
            except Exception as e:
                last_error = e
                logging.warning(f"[OSS] 上传失败 (尝试 {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    # 指数退避重试
                    wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                    time.sleep(wait_time)

        # 所有重试都失败了
        logging.error(f"[OSS] 上传最终失败: {key}, 错误: {last_error}")
        raise last_error

    if not ALLOW_LOCAL_UPLOAD_WITHOUT_OSS:
        raise ValueError("OSS 未配置（缺少 OSS_ACCESS_KEY_ID/OSS_ACCESS_KEY_SECRET/OSS_BUCKET_NAME/OSS_ENDPOINT）")

    # Local-only fallback for dev/tests: write to uploads/ and use the key for proxy reads.
    save_local_copy(key, file_bytes)
    return key


def get_file_stream(oss_key: str):
    """从 OSS 获取文件流，返回可迭代的字节流"""
    normalized_key = normalize_oss_key(oss_key)
    if not normalized_key:
        return None

    local_content = read_local_copy(normalized_key)
    if local_content is not None:
        return local_content

    if not is_oss_configured():
        return None

    bucket = get_bucket()
    try:
        result = bucket.get_object(normalized_key)
        return result.read()  # 返回完整字节内容
    except Exception:
        return None


def delete_file_by_key(oss_key: str) -> bool:
    """根据 OSS key 删除文件"""
    normalized_key = normalize_oss_key(oss_key)
    if not normalized_key:
        return False

    local_ok = True
    if LOCAL_UPLOAD_MIRROR:
        local_ok = delete_local_copy(normalized_key)

    if not is_oss_configured():
        return local_ok

    bucket = get_bucket()
    try:
        bucket.delete_object(normalized_key)
        return local_ok
    except Exception:
        return False
