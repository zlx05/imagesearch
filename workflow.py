"""CLI 独立测试用 LangGraph 工作流。

注意：主应用 app.py 不使用此文件，而是通过 run.io_bound() 直接调用各 Agent 节点。
此文件保留用于命令行单独调试完整工作流。
"""

from __future__ import annotations

import json
from functools import lru_cache

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from agents.analyzer import parse_node
from agents.orchestrator import report_node, upload_node
from agents.retriever import retrieve_node
from agents.validator import validate_node
from core.state import (
    NODE_ANALYZE,
    NODE_REPORT,
    NODE_RETRIEVE,
    NODE_UPLOAD,
    NODE_VALIDATE,
    AgentState,
    build_initial_state,
)


@lru_cache(maxsize=1)
def build_graph():
    """组装并编译图片传播链溯源 StateGraph。

    接口约定：
    - workflow.py 只负责导入节点、注册节点、声明边和 compile。
    - 业务逻辑必须放在 agents/*.py 内，状态协议必须放在 core/state.py 内。
    """
    graph_builder = StateGraph(AgentState)

    # 使用 LangChain Runnable 包装每个节点，后续替换为真实 Agent/Chain 时接口保持一致。
    graph_builder.add_node(NODE_UPLOAD, RunnableLambda(upload_node))
    graph_builder.add_node(NODE_RETRIEVE, RunnableLambda(retrieve_node))
    graph_builder.add_node(NODE_VALIDATE, RunnableLambda(validate_node))
    graph_builder.add_node(NODE_ANALYZE, RunnableLambda(parse_node))
    graph_builder.add_node(NODE_REPORT, RunnableLambda(report_node))

    graph_builder.add_edge(START, NODE_UPLOAD)
    graph_builder.add_edge(NODE_UPLOAD, NODE_RETRIEVE)
    graph_builder.add_edge(NODE_RETRIEVE, NODE_VALIDATE)
    graph_builder.add_edge(NODE_VALIDATE, NODE_ANALYZE)
    graph_builder.add_edge(NODE_ANALYZE, NODE_REPORT)
    graph_builder.add_edge(NODE_REPORT, END)

    return graph_builder.compile()


def run_workflow(initial_state: AgentState) -> AgentState:
    """运行完整工作流，供 Streamlit 或命令行脚本调用。"""
    graph = build_graph()
    return graph.invoke(initial_state)


if __name__ == "__main__":
    mock_initial_state = build_initial_state(
        {
            "filename": "demo.jpg",
            "content_type": "image/jpeg",
            "size_bytes": 1024,
        }
    )
    result = run_workflow(mock_initial_state)
    print(json.dumps(result, ensure_ascii=False, indent=2))
