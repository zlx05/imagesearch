from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List

from typing_extensions import TypedDict


# =========================
# 共享常量
# =========================
#
# 所有节点名称集中放在这里，workflow.py 组装图时复用同一套字符串，
# 防止不同开发者在各模块中手写节点名导致连线错误。
NODE_UPLOAD = "upload_node"
NODE_RETRIEVE = "retrieve_node"
NODE_VALIDATE = "validate_node"
NODE_ANALYZE = "parse_node"
NODE_REPORT = "report_node"

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


SIMILARITY_THRESHOLD = min(max(_env_float("SIMILARITY_THRESHOLD", 0.90), 0.0), 1.0)
DEFAULT_MERMAID_GRAPH = "graph TD"


class AgentState(TypedDict, total=False):
    """多智能体工作流的统一通信协议。

    约定：
    - 每个 LangGraph 节点只接收 AgentState，并返回需要更新的 AgentState 字段。
    - target_image 只保存上传文件的轻量元信息和 local_path，不保存图片二进制，避免状态过大。
    - nodes_data 是各智能体共享的结构化中间结果列表。
    - final_report 和 mermaid_graph 由编排/输出阶段统一生成。
    """

    target_image: Dict[str, Any]
    nodes_data: List[Dict[str, Any]]
    retrieved_nodes: List[Dict[str, Any]]
    validated_nodes: List[Dict[str, Any]]
    rejected_nodes: List[Dict[str, Any]]
    search_engines: List[str]
    retriever_max_results: int
    retriever_max_results_per_engine: int
    similarity_threshold: float
    retrieval_summary: Dict[str, Any]
    validation_summary: Dict[str, Any]
    mermaid_graph: str
    final_report: str
    execution_logs: List[str]


def make_log_line(message: str) -> str:
    """生成统一格式的执行日志。"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    return f"[{timestamp}] {message}"


def append_log(state: AgentState, message: str) -> List[str]:
    """把日志写入控制台，并返回追加后的日志列表。

    节点函数应使用这个函数更新 execution_logs，保证 UI 和终端日志一致。
    """
    log_line = make_log_line(message)
    print(log_line)
    return [*state.get("execution_logs", []), log_line]


def build_initial_state(target_image: Dict[str, Any]) -> AgentState:
    """根据上传图片元信息构造工作流初始状态。"""
    return {
        "target_image": target_image,
        "nodes_data": [],
        "retrieved_nodes": [],
        "validated_nodes": [],
        "rejected_nodes": [],
        "search_engines": [],
        "retriever_max_results": 12,
        "retriever_max_results_per_engine": 10,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "retrieval_summary": {},
        "validation_summary": {},
        "mermaid_graph": "",
        "final_report": "",
        "execution_logs": [],
    }
