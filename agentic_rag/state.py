"""
全局状态管理模块
==============
持有全局编排器实例，避免循环导入。
"""

from typing import Optional, Any

_orchestrator: Optional[Any] = None


def set_orchestrator(orch: Any) -> None:
    """设置全局编排器实例"""
    global _orchestrator
    _orchestrator = orch


def get_orchestrator() -> Any:
    """获取全局编排器实例"""
    return _orchestrator