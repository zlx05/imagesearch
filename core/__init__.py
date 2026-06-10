"""核心协议模块。

对外暴露 AgentState 和共享常量，其他模块统一从这里或 core.state 导入，
避免各 agent 之间互相引用造成循环依赖。
"""

from core.state import (
    DEFAULT_MERMAID_GRAPH,
    NODE_ANALYZE,
    NODE_REPORT,
    NODE_RETRIEVE,
    NODE_UPLOAD,
    NODE_VALIDATE,
    SIMILARITY_THRESHOLD,
    AgentState,
    append_log,
    build_initial_state,
    make_log_line,
)

__all__ = [
    "AgentState",
    "DEFAULT_MERMAID_GRAPH",
    "NODE_ANALYZE",
    "NODE_REPORT",
    "NODE_RETRIEVE",
    "NODE_UPLOAD",
    "NODE_VALIDATE",
    "SIMILARITY_THRESHOLD",
    "append_log",
    "build_initial_state",
    "make_log_line",
]
