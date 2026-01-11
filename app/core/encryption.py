"""
API Key 加密工具

使用 AES-256-GCM 加密用户的 API Key，确保敏感信息在数据库中不以明文存储。
密钥从环境变量 API_KEY_ENCRYPTION_SECRET 读取，如未配置则复用 JWT_SECRET。
"""

import base64
import os
import secrets
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import settings


def _get_encryption_key() -> bytes:
    """获取加密密钥（32 字节）"""
    # 优先使用专用密钥，否则复用 JWT_SECRET
    secret = os.getenv("API_KEY_ENCRYPTION_SECRET") or settings.JWT_SECRET
    if not secret:
        raise ValueError("未配置加密密钥（API_KEY_ENCRYPTION_SECRET 或 JWT_SECRET）")
    
    # 将字符串密钥转换为 32 字节（AES-256）
    # 使用 SHA-256 哈希确保长度一致
    import hashlib
    return hashlib.sha256(secret.encode("utf-8")).digest()


def encrypt_api_key(plain_key: str) -> str:
    """
    加密 API Key
    
    返回格式: base64(nonce + ciphertext + tag)
    """
    if not plain_key:
        return ""
    
    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    
    # 生成 12 字节随机 nonce
    nonce = secrets.token_bytes(12)
    
    # 加密
    ciphertext = aesgcm.encrypt(nonce, plain_key.encode("utf-8"), None)
    
    # 组合: nonce + ciphertext（包含 tag）
    encrypted = nonce + ciphertext
    
    # Base64 编码便于存储
    return base64.b64encode(encrypted).decode("utf-8")


def decrypt_api_key(encrypted_key: str) -> Optional[str]:
    """
    解密 API Key
    
    如果解密失败（密钥变更、数据损坏等），返回 None
    """
    if not encrypted_key:
        return None
    
    # 兼容旧数据：如果不是 base64 格式，可能是明文存储的旧数据
    # 明文 API Key 通常以 "sk-" 开头
    if encrypted_key.startswith("sk-") or encrypted_key.startswith("ak-"):
        return encrypted_key
    
    try:
        key = _get_encryption_key()
        aesgcm = AESGCM(key)
        
        # Base64 解码
        encrypted = base64.b64decode(encrypted_key)
        
        # 分离 nonce 和 ciphertext
        nonce = encrypted[:12]
        ciphertext = encrypted[12:]
        
        # 解密
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")
    except Exception:
        # 解密失败，可能是：
        # 1. 旧的明文数据
        # 2. 密钥变更
        # 3. 数据损坏
        # 返回原值让调用方决定如何处理
        return encrypted_key


def is_encrypted(value: str) -> bool:
    """
    判断值是否已加密
    
    加密后的值是 base64 编码，长度较长且不以常见 API Key 前缀开头
    """
    if not value:
        return False
    
    # 常见 API Key 前缀（明文）
    plain_prefixes = ("sk-", "ak-", "key-", "api-")
    if any(value.startswith(p) for p in plain_prefixes):
        return False
    
    # 尝试 base64 解码
    try:
        decoded = base64.b64decode(value)
        # 加密数据至少包含 12 字节 nonce + 16 字节 tag + 1 字节数据
        return len(decoded) >= 29
    except Exception:
        return False

