"""
工具函数模块
===========
提供通用工具函数：ID 生成、文本清理、日志配置等。
"""

import uuid
import hashlib
import re
from typing import Optional
from pathlib import Path
from loguru import logger
import json


def generate_id(prefix: str = "doc") -> str:
    """生成唯一 ID"""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def generate_content_hash(content: str) -> str:
    """生成内容哈希，用于去重"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def clean_text(text: str) -> str:
    """清理文本：合并空白与换行，去除首尾空白"""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_length: int = 200) -> str:
    """截断文本到指定长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def setup_logger(log_file: Optional[str] = None, level: str = "INFO"):
    """配置 loguru 日志器"""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=str(log_path),
            level=level,
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{function}:{line} - {message}",
        )
    return logger


def safe_serialize(obj):
    """安全序列化对象为 JSON"""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def merge_dicts(base: dict, override: dict, deep: bool = True) -> dict:
    """合并字典，override 覆盖 base"""
    result = base.copy()
    for key, value in override.items():
        if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result