"""
媒体资产注册表模块
================
RAG-Anything 风格的多媒体存储：
- 解析文档时抽取图片/表格为 MediaAsset（含 base64 数据或表格文本）
- 检索到文本块后，通过图存储的引用边找到 media_id，再从注册表取回资产数据
- 持久化到 JSON 文件（图片 base64 较大），并支持「内存上限 + 磁盘懒加载」，
  避免超大 PDF 的 base64 全部驻留内存

与 graph_store 的分工：
- graph_store 负责「引用图」：chunk 节点 --references--> media 节点（位置/标签/页码）
- media_store 负责「资产数据」：media_id -> MediaAsset（图片 data / 表格文本）
"""

import os
import tempfile
from threading import RLock
from typing import Dict, List, Optional
from loguru import logger

from agentic_rag.memory.multi_modal_parser import MediaAsset


class MediaRegistry:
    """媒体资产注册表（内存优先 + 磁盘持久化 + 内存上限懒加载）"""

    def __init__(self, persist_path: Optional[str] = None, auto_save: bool = False,
                 max_memory_bytes: Optional[int] = None):
        self._assets: Dict[str, MediaAsset] = {}
        self.persist_path = persist_path
        self.auto_save = auto_save
        self.max_memory_bytes = max_memory_bytes
        self._lock = RLock()
        if persist_path and auto_save:
            self.load()

    # ── 增删改查 ──

    def add(self, media: MediaAsset) -> bool:
        """注册一个媒体资产；同 id 已存在时更新"""
        self._assets[media.id] = media
        # 先持久化（数据完整），再卸载超出内存上限的数据
        if self.auto_save:
            self.save()
        self._enforce_memory_limit()
        return True

    def add_many(self, media_list: List[MediaAsset]) -> int:
        count = 0
        for m in media_list:
            if m.id and m.id not in self._assets:
                self._assets[m.id] = m
                count += 1
        if self.auto_save and count:
            self.save()
        self._enforce_memory_limit()
        return count

    def get(self, media_id: str) -> Optional[MediaAsset]:
        """获取资产；若 data 已被内存上限卸载且磁盘存在，则从磁盘懒加载"""
        asset = self._assets.get(media_id)
        if asset and not asset.data:
            self._load_data_from_disk(asset)
        return asset

    def get_many(self, media_ids: List[str]) -> List[MediaAsset]:
        seen = set()
        result = []
        for mid in media_ids:
            if mid in seen:
                continue
            seen.add(mid)
            asset = self.get(mid)
            if asset:
                result.append(asset)
        return result

    def list(self) -> List[MediaAsset]:
        return list(self._assets.values())

    def clear(self) -> int:
        count = len(self._assets)
        self._assets.clear()
        if self.auto_save:
            self.save()
        return count

    @property
    def count(self) -> int:
        return len(self._assets)

    # ── 内存上限：超出时把最早/最大的 base64 数据卸载（元数据保留，可从磁盘恢复） ──

    def _enforce_memory_limit(self):
        if not self.max_memory_bytes:
            return
        total = sum(len(a.data or "") for a in self._assets.values())
        if total <= self.max_memory_bytes:
            return
        logger.info(
            f"媒体数据占用 {total / 1024 / 1024:.1f}MB 超过上限 "
            f"{self.max_memory_bytes / 1024 / 1024:.0f}MB，卸载部分图片数据（可从磁盘懒加载）"
        )
        for aid in list(self._assets.keys()):
            if total <= self.max_memory_bytes:
                break
            a = self._assets[aid]
            if a.data:
                total -= len(a.data)
                a.data = None

    def _load_data_from_disk(self, asset: MediaAsset):
        """从持久化 JSON 读取指定资产的 data 并回填（按需懒加载）"""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        try:
            import json
            with open(self.persist_path, encoding="utf-8") as f:
                items = json.load(f)
            for it in items:
                if it.get("id") == asset.id and it.get("data"):
                    asset.data = it["data"]
                    return
        except Exception as e:
            logger.warning(f"从磁盘懒加载媒体数据失败: {e}")

    # ── 持久化 ──

    def to_dicts(self, include_data: bool = True) -> List[dict]:
        return [{
            "id": m.id,
            "doc_id": m.doc_id,
            "type": m.type,
            "page": m.page,
            "label": m.label,
            "caption": m.caption,
            "search_text": m.search_text,
            "mime_type": m.mime_type,
            "checksum": m.checksum,
            "extraction_method": m.extraction_method,
            "quality": m.quality,
            "data": m.data if include_data else None,
            "metadata": m.metadata,
        } for m in self._assets.values()]

    def save(self, path: Optional[str] = None) -> str:
        import json
        path = path or self.persist_path
        if not path:
            logger.warning("MediaRegistry 未设置 persist_path，跳过保存")
            return ""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 已被内存上限卸载的 data 需从磁盘旧文件回填，避免覆盖丢失
        old_data: Dict[str, str] = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    for it in json.load(f):
                        if it.get("data"):
                            old_data[it["id"]] = it["data"]
            except Exception:
                old_data = {}

        out = []
        for m in self._assets.values():
            data = m.data or old_data.get(m.id)
            out.append({
                "id": m.id,
                "doc_id": m.doc_id,
                "type": m.type,
                "page": m.page,
                "label": m.label,
                "caption": m.caption,
                "search_text": m.search_text,
                "mime_type": m.mime_type,
                "checksum": m.checksum,
                "extraction_method": m.extraction_method,
                "quality": m.quality,
                "data": data,
                "metadata": m.metadata,
            })
        directory = os.path.dirname(os.path.abspath(path))
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False, suffix=".tmp") as f:
            json.dump(out, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
            temp_path = f.name
        os.replace(temp_path, path)
        logger.info(f"媒体资产已保存: {self.count} 条 -> {path}")
        return path

    def load(self, path: Optional[str] = None) -> int:
        import json
        path = path or self.persist_path
        if not path or not os.path.exists(path):
            return 0
        try:
            with open(path, encoding="utf-8") as f:
                items = json.load(f)
            count = 0
            for it in items:
                self._assets[it["id"]] = MediaAsset(
                    id=it.get("id", ""),
                    doc_id=it.get("doc_id", ""),
                    type=it.get("type", "image"),
                    page=it.get("page", 1),
                    label=it.get("label", ""),
                    caption=it.get("caption", ""),
                    data=it.get("data"),
                    search_text=it.get("search_text", it.get("caption", "")),
                    mime_type=it.get("mime_type", ""),
                    checksum=it.get("checksum", ""),
                    extraction_method=it.get("extraction_method", "legacy_json"),
                    quality=it.get("quality", "derived"),
                    metadata=it.get("metadata", {}),
                )
                count += 1
            self._enforce_memory_limit()
            logger.info(f"媒体资产已加载: {count} 条 <- {path}")
            return count
        except Exception as e:
            logger.warning(f"媒体资产加载失败: {e}")
            return 0
