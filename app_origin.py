#!/usr/bin/env python
"""图片溯源智能体 — NiceGUI 稳定展示版

目标：
1. 运行阶段只做轻量状态刷新，避免 WebSocket 被大量日志/JSON 拖垮。
2. 页面内 SVG 拓扑优先复用 topology.html 内的 G6 payload，确保节点和边关系与大屏一致。
3. 结果展示只渲染摘要、分页表格和 Markdown 报告；完整 JSON/日志以下载方式提供。
4. 支持 --test：直接加载 output/ 下最新报告，方便答辩前预览。

启动：python app_nicegui.py
测试：python app_nicegui.py --test
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from nicegui import app, run, ui

# =============================================================================
# 环境初始化：尽量保持你原 app_nicegui.py 的行为
# =============================================================================

PLACEHOLDER_MARKERS = ("xxxx", "your-", "replace-", "sk-xxxx", "fc-xxxx")
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
UPLOAD_DIR = PROJECT_DIR / "data" / "uploads"
STATUS_DIR = OUTPUT_DIR / "current"
STATUS_FILE = STATUS_DIR / "status.json"


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return not normalized or any(marker in normalized for marker in PLACEHOLDER_MARKERS)


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_environment() -> list[str]:
    loaded_from: list[str] = []
    for env_path in (PROJECT_DIR / ".env", Path(".env")):
        values = _parse_env_file(env_path)
        if not values:
            continue
        applied = False
        for key, value in values.items():
            if key in os.environ or _looks_like_placeholder(value):
                continue
            os.environ[key] = value
            applied = True
        if applied:
            loaded_from.append(str(env_path))
    return loaded_from


ENV_SOURCES = load_environment()

MODEL_CACHE_DIR = PROJECT_DIR / "data" / "model_cache" / "huggingface"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
for cache_key, default_path in {
    "HF_HOME": MODEL_CACHE_DIR,
    "TRANSFORMERS_CACHE": MODEL_CACHE_DIR / "transformers",
}.items():
    configured_path = Path(os.getenv(cache_key, str(default_path)))
    if not configured_path.is_absolute():
        configured_path = PROJECT_DIR / configured_path
    configured_path.mkdir(parents=True, exist_ok=True)
    os.environ[cache_key] = str(configured_path)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _apply_env_defaults() -> None:
    os.environ.setdefault("VALIDATOR_ENABLE_MULTIMODAL_LLM", "true")
    os.environ.setdefault("ANALYZER_ENABLE_TIKOMNI", "true")
    os.environ.setdefault("TIKOMNI_BASE_URL", "https://api.tikomni.com")
    os.environ.setdefault("TIKOMNI_AUTH_HEADER", "Authorization")
    os.environ.setdefault("TIKOMNI_AUTH_SCHEME", "Bearer")
    os.environ.setdefault("ANALYZER_TIKOMNI_TIMEOUT_SECONDS", "20")
    os.environ.setdefault("ANALYZER_ENABLE_FIRECRAWL", "true")
    os.environ.setdefault("ANALYZER_ENABLE_LLM", "true")
    os.environ.setdefault("ANALYZER_ENABLE_LLM_RELATION_ANALYSIS", "true")
    os.environ.setdefault("ANALYZER_LLM_RELATION_TIMEOUT_SECONDS", "75")
    os.environ.setdefault("RETRIEVER_ENGINE_TIMEOUT_SECONDS", "45")
    os.environ.setdefault("ANALYZER_MAX_PAGE_FETCH_NODES", "30")
    os.environ.setdefault("ANALYZER_MAX_FIRECRAWL_NODES", "30")
    os.environ.setdefault("ANALYZER_MAX_LLM_NODES", "30")
    os.environ.setdefault("ANALYZER_MAX_LLM_RELATION_NODES", "12")
    os.environ.setdefault("ANALYZER_MAX_WORKERS", "5")
    os.environ.setdefault("ANALYZER_MAX_LLM_ENRICHMENT_NODES", "20")
    os.environ.setdefault("ANALYZER_PAGE_FETCH_TIMEOUT_SECONDS", "8")
    os.environ.setdefault("ANALYZER_FIRECRAWL_TIMEOUT_MS", "15000")
    os.environ.setdefault("ANALYZER_LLM_TIMEOUT_SECONDS", "30")


_apply_env_defaults()

# =============================================================================
# 业务模块导入
# =============================================================================

try:
    from agents.orchestrator import report_node, upload_node
    from agents.retriever import retrieve_node
    from agents.validator import validate_node
    from agents.analyzer import parse_node
    from core.state import AgentState, build_initial_state
    from core.visualization import write_g6_html
except Exception as import_error:  # 页面仍可启动，用于只展示已有报告
    report_node = upload_node = retrieve_node = validate_node = parse_node = None  # type: ignore[assignment]
    AgentState = dict  # type: ignore[misc,assignment]
    build_initial_state = None  # type: ignore[assignment]
    write_g6_html = None  # type: ignore[assignment]
    IMPORT_ERROR = import_error
else:
    IMPORT_ERROR = None

# =============================================================================
# 常量与状态文件
# =============================================================================

STEP_ORDER = ["upload", "retrieve", "validate", "analyze", "report"]
STEP_LABELS = {
    "upload": "上传图片",
    "retrieve": "以图搜图",
    "validate": "相似度校验",
    "analyze": "传播分析",
    "report": "生成报告",
}
STEP_PROGRESS = {
    "idle": 0.0,
    "upload": 0.08,
    "retrieve": 0.28,
    "validate": 0.64,
    "analyze": 0.90,
    "report": 1.0,
}


@dataclass
class AppRuntime:
    uploaded_content: bytes | None = None
    uploaded_filename: str = ""
    running: bool = False
    active_report_dir: str | None = None
    latest_log_lines: list[str] | None = None


RUNTIME = AppRuntime(latest_log_lines=[])


def _json_default(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


def write_status(
    *,
    status: str,
    step: str = "idle",
    message: str = "",
    progress: float | None = None,
    report_dir: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": status,
        "step": step,
        "message": message,
        "progress": STEP_PROGRESS.get(step, 0.0) if progress is None else progress,
        "report_dir": report_dir,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    for attempt in range(5):
        try:
            tmp = STATUS_FILE.with_suffix(f".tmp.{attempt}")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(STATUS_FILE)
            break
        except PermissionError:
            if attempt < 4:
                time.sleep(0.05 * (attempt + 1))
                continue
            try:
                STATUS_FILE.write_text(text, encoding="utf-8")
            except PermissionError:
                pass


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_status() -> dict[str, Any]:
    return read_json(STATUS_FILE, {
        "status": "idle",
        "step": "idle",
        "message": "等待上传图片",
        "progress": 0.0,
        "report_dir": None,
    })


def save_uploaded_image(content: bytes, filename: str) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", filename or "uploaded.jpg")
    saved_path = UPLOAD_DIR / f"{uuid4().hex}_{safe_name}"
    saved_path.write_bytes(content)
    return saved_path


def latest_report_dir() -> Path | None:
    if not OUTPUT_DIR.exists():
        return None
    dirs = [p for p in OUTPUT_DIR.glob("report_*/") if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def summarize_nodes(nodes: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    retrieval = summary.get("retrieval_summary", {}) if isinstance(summary, dict) else {}
    validation = summary.get("validation_summary", {}) if isinstance(summary, dict) else {}
    analysis = summary.get("analysis_summary", {}) if isinstance(summary, dict) else {}

    with_time = sum(1 for n in nodes if n.get("published_at"))
    suspected_sources = [n for n in nodes if n.get("is_suspected_source")]
    key_nodes = [n for n in nodes if n.get("is_key_node")]
    tampered_nodes = [n for n in nodes if n.get("suspected_tampering") or (n.get("tamper_analysis") or {}).get("is_tampered")]

    earliest = None
    dated_nodes = [n for n in nodes if n.get("published_at")]
    if dated_nodes:
        earliest = sorted(dated_nodes, key=lambda n: str(n.get("published_at")))[0]

    topology_data = summary.get("topology_data", {}) if isinstance(summary, dict) else {}
    topo_nodes = topology_data.get("nodes", []) if isinstance(topology_data, dict) else []
    topo_edges = topology_data.get("edges", []) if isinstance(topology_data, dict) else []

    return {
        "candidate_count": retrieval.get("result_count") or retrieval.get("candidate_count") or "-",
        "validated_count": validation.get("validated_count") or len(nodes),
        "with_time_count": analysis.get("with_time_count") or with_time,
        "node_count": len(nodes),
        "edge_count": len(topo_edges) if topo_edges else analysis.get("topology_edge_count", "-"),
        "key_count": len(key_nodes) or len(analysis.get("key_node_ids", []) or []),
        "tampered_count": len(tampered_nodes) or len(analysis.get("tampered_node_ids", []) or []),
        "source_count": len(suspected_sources),
        "earliest_time": earliest.get("published_at") if earliest else "-",
        "earliest_publisher": (earliest.get("publisher") or earliest.get("author") or earliest.get("domain")) if earliest else "-",
        "earliest_title": earliest.get("title") if earliest else "-",
        "search_engines": ", ".join((retrieval.get("per_engine_counts") or {}).keys()) or "-",
        "topology_node_count": len(topo_nodes) if topo_nodes else len(nodes),
    }


def compact_node_row(n: dict[str, Any], idx: int) -> dict[str, Any]:
    url = str(n.get("url") or n.get("canonical_url") or "")
    title = str(n.get("title") or n.get("description") or "")
    publisher = n.get("publisher") or n.get("author") or n.get("domain") or "未知"
    platform = n.get("platform") or n.get("platform_family") or n.get("source") or "未知"
    return {
        "idx": idx,
        "time": n.get("published_at") or "未知",
        "platform": platform,
        "publisher": publisher,
        "title": title[:90] + ("..." if len(title) > 90 else ""),
        "similarity": n.get("similarity"),
        "source_score": n.get("source_score"),
        "repost": n.get("repost_count"),
        "comment": n.get("comment_count"),
        "like": n.get("like_count"),
        "role": n.get("propagation_role") or "候选节点",
        "source": "是" if n.get("is_suspected_source") else "否",
        "key": "是" if n.get("is_key_node") else "否",
        "url": url,
    }


def top_timeline_rows(nodes: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    dated = [n for n in nodes if n.get("published_at")]
    dated = sorted(dated, key=lambda n: str(n.get("published_at")))
    return [compact_node_row(n, i + 1) for i, n in enumerate(dated[:limit])]


def make_download_url(report_dir_name: str, filename: str) -> str:
    return f"/report_files/{report_dir_name}/{filename}"


def copy_uploaded_test_files_to_output() -> str | None:
    """如果用户把测试数据直接放在当前目录/容器目录，生成一个 report_imported 方便 --test。"""
    candidates = [PROJECT_DIR, Path.cwd(), Path("/mnt/data")]
    for base in candidates:
        nodes = base / "nodes_data.json"
        summary = base / "summary.json"
        topo = base / "topology.html"
        logs = base / "logs.txt"
        if nodes.exists() and summary.exists():
            report_dir = OUTPUT_DIR / "report_imported_demo"
            report_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(nodes, report_dir / "nodes_data.json")
            shutil.copy2(summary, report_dir / "summary.json")
            if topo.exists():
                shutil.copy2(topo, report_dir / "topology.html")
            if logs.exists():
                shutil.copy2(logs, report_dir / "logs.txt")
            return report_dir.name
    return None


# =============================================================================
# 业务流程：只写状态文件和 output，不向页面推海量日志
# =============================================================================


def _progress_recorder(full_logs: list[str], key_logs: list[str] | None = None):
    """后台节点的细粒度 progress 回调。

    注意：这里不再把每条 progress 都推给前端日志，只写入 full_logs。
    前端只展示 key_logs 中的关键里程碑，避免答辩展示时日志刷屏/卡顿。
    """
    last_emit = {"t": 0.0, "count": 0}

    def on_progress(step: str, current: int, total: int, sub_total: int, message: str) -> None:
        now = time.time()
        full_logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # 降频写 status：只更新进度与一句状态，不更新前端关键日志。
        if now - last_emit["t"] < 2.5 and current != total:
            return
        last_emit["t"] = now
        last_emit["count"] += 1
        try:
            cur = float(current) if current is not None else 0.0
            tot = float(total) if total is not None else 0.0
        except (TypeError, ValueError):
            cur, tot = 0.0, 0.0
        if tot > 0:
            pct = min(max(cur / tot, 0.0), 1.0)
            if "validate" in step:
                progress = 0.30 + pct * 0.34
                logical_step = "validate"
            elif "retrieve" in step:
                progress = 0.10 + pct * 0.18
                logical_step = "retrieve"
            else:
                progress = None
                logical_step = "validate"
        else:
            progress = None
            logical_step = "validate"
        write_status(
            status="running",
            step=logical_step,
            message=message[:120],
            progress=progress,
            extra={"recent_logs": (key_logs or [])},
        )

    return on_progress


async def execute_workflow(content: bytes, filename: str) -> str | None:
    """真实运行工作流。

    日志分两套：
    - key_logs：前端展示用关键里程碑日志，写入 logs.txt。
    - full_logs：后台排错用完整细粒度日志，写入 full_logs.txt。
    """
    if IMPORT_ERROR is not None or build_initial_state is None:
        write_status(status="error", step="idle", message=f"业务模块导入失败：{IMPORT_ERROR}", progress=0.0)
        return None

    key_logs: list[str] = []
    full_logs: list[str] = []
    report_dir: Path | None = None

    def _ts() -> str:
        return time.strftime("%H:%M:%S")

    def key_log(msg: str) -> None:
        line = f"[{_ts()}] {msg}"
        key_logs.append(line)
        full_logs.append(line)
        RUNTIME.latest_log_lines = key_logs[:]

    def full_log(msg: str) -> None:
        full_logs.append(f"[{_ts()}] {msg}")

    try:
        # 1. 上传图片
        key_log("上传图片：开始保存上传文件")
        write_status(status="running", step="upload", message="正在保存上传图片...", progress=0.02, extra={"recent_logs": key_logs})
        saved_path = await run.io_bound(save_uploaded_image, content, filename)
        key_log(f"上传图片：保存完成，文件名={filename}，大小={len(content) / 1024:.1f}KB")

        state: AgentState = build_initial_state({
            "filename": filename,
            "content_type": "",
            "size_bytes": len(content),
            "local_path": str(saved_path),
        })

        env_engines = [
            e.strip() for e in os.getenv(
                "SEARCH_ENGINE",
                "baidu,tineye,yandex,bing,google,saucenao,ascii2d,serpapi_lens,mitmproxy",
            ).split(",") if e.strip()
        ]
        state["search_engines"] = env_engines  # type: ignore[index]
        state["retriever_max_results"] = int(os.getenv(
            "RETRIEVER_MAX_TOTAL_RESULTS", os.getenv("RETRIEVER_MAX_RESULTS", "999")
        ))  # type: ignore[index]
        full_log(f"upload_node: saved_path={saved_path}")

        # 2. 图片信息提取
        key_log("图片信息提取：开始读取目标图片元信息")
        write_status(status="running", step="upload", message="正在提取目标图片信息...", progress=STEP_PROGRESS["upload"], extra={"recent_logs": key_logs})
        r = await run.io_bound(upload_node, state)  # type: ignore[misc]
        state.update(r)
        target_image = state.get("target_image", {}) if isinstance(state, dict) else {}
        if isinstance(target_image, dict):
            width = target_image.get("width") or target_image.get("image_width") or "-"
            height = target_image.get("height") or target_image.get("image_height") or "-"
            local_path = target_image.get("local_path") or str(saved_path)
            key_log(f"图片信息提取：完成，尺寸={width}×{height}，路径={local_path}")
        else:
            key_log("图片信息提取：完成")

        # 3. 以图搜图
        key_log(f"以图搜图：开始调用搜索引擎（{', '.join(env_engines)}）")
        write_status(status="running", step="retrieve", message="正在以图搜图，收集候选链接...", progress=0.12, extra={"recent_logs": key_logs})
        r = await run.io_bound(retrieve_node, state, progress_callback=_progress_recorder(full_logs, key_logs))  # type: ignore[misc]
        state.update(r)
        rsummary = state.get("retrieval_summary", {})
        per_engine = rsummary.get("per_engine_counts", {}) if isinstance(rsummary, dict) else {}
        engine_text = ", ".join(f"{k}:{v}" for k, v in per_engine.items()) if isinstance(per_engine, dict) and per_engine else "-"
        candidate_count = rsummary.get("result_count", len(state.get("nodes_data", []))) if isinstance(rsummary, dict) else len(state.get("nodes_data", []))
        key_log(f"以图搜图：完成，候选结果={candidate_count}条，引擎返回={engine_text}")

        # 4. 相似度校验与去重
        key_log("相似度校验/去重：开始处理候选结果")
        write_status(status="running", step="validate", message="正在进行相似度校验和去重...", progress=0.32, extra={"recent_logs": key_logs})
        vstate = dict(state)
        vstate["_progress_callback"] = _progress_recorder(full_logs, key_logs)
        r = await run.io_bound(validate_node, vstate)  # type: ignore[misc]
        state.update(r)
        vsummary = state.get("validation_summary", {})
        if isinstance(vsummary, dict):
            key_log(
                "相似度校验/去重：完成，"
                f"通过={vsummary.get('validated_count', '-')}条，"
                f"拒绝={vsummary.get('rejected_count', '-')}条，"
                f"去重={vsummary.get('deduplicated_count', '-')}条"
            )
        else:
            key_log("相似度校验/去重：完成")

        # 5. 内容提取与传播分析
        key_log("内容提取/传播分析：开始提取发布时间、账号、互动量与传播关系")
        write_status(status="running", step="analyze", message="正在提取内容并分析传播关系...", progress=0.70, extra={"recent_logs": key_logs})
        r = await run.io_bound(parse_node, state)  # type: ignore[misc]
        state.update(r)
        analysis_summary = state.get("analysis_summary", {})
        nodes_data_after_analyze = state.get("nodes_data", [])
        if isinstance(analysis_summary, dict):
            key_log(
                "内容提取/传播分析：完成，"
                f"时间证据={analysis_summary.get('with_time_count', '-') }条，"
                f"关键节点={len(analysis_summary.get('key_node_ids', []) or [])}个，"
                f"总节点={len(nodes_data_after_analyze) if isinstance(nodes_data_after_analyze, list) else '-'}"
            )
        else:
            key_log("内容提取/传播分析：完成")

        # 6. 生成报告
        key_log("生成报告：开始生成文字报告与拓扑图")
        write_status(status="running", step="report", message="正在生成报告和拓扑图...", progress=0.94, extra={"recent_logs": key_logs})
        r = await run.io_bound(report_node, state)  # type: ignore[misc]
        state.update(r)
        key_log("生成报告：文字报告生成完成")

        report_dir = OUTPUT_DIR / f"report_{time.strftime('%Y%m%d_%H%M%S')}"
        report_dir.mkdir(parents=True, exist_ok=True)

        # 拓扑图：静态 HTML，大屏交互版；页面内 SVG 会复用其中 G6 payload。
        if write_g6_html is not None:
            try:
                await run.io_bound(write_g6_html, state, report_dir / "topology.html")  # type: ignore[misc]
                key_log("生成报告：拓扑图 HTML 生成完成")
            except Exception as e:
                key_log(f"生成报告：拓扑图 HTML 生成失败，原因={e}")
                full_log(f"topology.html 生成失败: {e}")

        nodes_data = state.get("nodes_data", [])
        summary_payload = {
            "report": state.get("final_report", ""),
            "analysis_summary": state.get("analysis_summary", {}),
            "validation_summary": state.get("validation_summary", {}),
            "retrieval_summary": state.get("retrieval_summary", {}),
            "topology_data": state.get("topology_data", {}),
            "key_logs": key_logs,
        }
        (report_dir / "nodes_data.json").write_text(json.dumps(nodes_data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        (report_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        (report_dir / "state_dump.json").write_text(json.dumps({k: v for k, v in state.items() if k != "_progress_callback"}, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        # logs.txt：前端展示用关键日志；full_logs.txt：排错用完整细粒度日志。
        key_log(f"生成报告：全部完成，输出目录={report_dir.name}")
        (report_dir / "logs.txt").write_text("\n".join(key_logs), encoding="utf-8")
        (report_dir / "full_logs.txt").write_text("\n".join(full_logs), encoding="utf-8")

        RUNTIME.active_report_dir = report_dir.name
        write_status(
            status="done",
            step="report",
            message="分析完成，结果已生成。",
            progress=1.0,
            report_dir=report_dir.name,
            extra={"recent_logs": key_logs},
        )
        return report_dir.name
    except Exception as e:
        import traceback as _tb
        tb_text = _tb.format_exc()
        key_log(f"流程异常：{e}")
        full_log(f"工作流异常: {e}\n{tb_text}")
        print(f"[ERROR] 工作流异常:\n{tb_text}")
        failed_dir = OUTPUT_DIR / f"partial_failed_{time.strftime('%Y%m%d_%H%M%S')}"
        try:
            failed_dir.mkdir(parents=True, exist_ok=True)
            (failed_dir / "logs.txt").write_text("\n".join(key_logs), encoding="utf-8")
            (failed_dir / "full_logs.txt").write_text("\n".join(full_logs), encoding="utf-8")
        except Exception:
            pass
        write_status(status="error", step="idle", message=f"流程失败：{e}", progress=0.0, extra={"recent_logs": key_logs})
        return None

# =============================================================================
# UI 工具
# =============================================================================



def _node_platform_label(node: dict[str, Any]) -> str:
    platform = str(node.get("platform") or "").lower()
    url = str(node.get("url") or "").lower()
    family = str(node.get("platform_family") or "").lower()
    if "weibo" in platform or "weibo" in url or "微博" in str(node.get("platform") or ""):
        return "微博"
    if "xiaohongshu" in platform or "xhs" in platform or "小红书" in str(node.get("platform") or ""):
        return "小红书"
    if family in {"forum", "论坛"} or "论坛" in str(node.get("platform_family") or ""):
        return "论坛/外部"
    if family in {"news", "新闻"} or "新闻" in str(node.get("platform_family") or ""):
        return "新闻/外部"
    return "其他外部"


def build_native_topology_svg(nodes: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    """生成不依赖 iframe/G6 的页面内 SVG 拓扑预览，保证 NiceGUI 页面中可直接展示。"""
    visible = [n for n in nodes if isinstance(n, dict) and n.get("is_topology_visible", True)] or [n for n in nodes if isinstance(n, dict)]
    visible = sorted(visible, key=lambda n: (str(n.get("published_at") or "9999"), str(n.get("id") or "")))
    if not visible:
        return "<div style='padding:28px;color:#64748b;'>暂无可展示节点。</div>"

    lanes: list[str] = []
    for n in visible:
        label = _node_platform_label(n)
        if label not in lanes:
            lanes.append(label)
    preferred = ["微博", "小红书", "论坛/外部", "新闻/外部", "其他外部"]
    lanes = [x for x in preferred if x in lanes] + [x for x in lanes if x not in preferred]

    n_count = len(visible)
    width = max(1400, 190 + n_count * 120)
    lane_h = 160
    top_pad = 96
    left_pad = 145
    height = top_pad + len(lanes) * lane_h + 110

    positions: dict[str, tuple[float, float]] = {}
    for idx, node in enumerate(visible):
        x = left_pad + idx * 120
        lane_label = _node_platform_label(node)
        lane = lanes.index(lane_label) if lane_label in lanes else 0
        y = top_pad + lane * lane_h + 70 + ((idx % 3) - 1) * 16
        positions[str(node.get("id") or idx)] = (x, y)

    def esc(v: Any) -> str:
        return html_escape(str(v if v is not None else ""))

    parts: list[str] = []
    parts.append(f"""
    <div style="width:100%;overflow:auto;border:1px solid #e5e7eb;border-radius:16px;background:#ffffff;">
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="display:block;">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
        </marker>
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.12"/>
        </filter>
      </defs>
      <rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>
      <text x="24" y="36" font-size="19" font-weight="800" fill="#0f172a">页面内传播拓扑预览</text>
    """)

    legend_x = max(760, width - 560)
    legend_items = [("#16a34a", "疑似源头"), ("#dc2626", "疑似篡改"), ("#2563eb", "关键节点"), ("#64748b", "普通节点")]
    for i, (c, t) in enumerate(legend_items):
        x = legend_x + i * 125
        parts.append(f'<circle cx="{x}" cy="34" r="7" fill="{c}"/><text x="{x+12}" y="39" font-size="12" fill="#334155">{t}</text>')

    for lane_idx, lane in enumerate(lanes):
        y0 = top_pad + lane_idx * lane_h
        bg = "#ffffff" if lane_idx % 2 == 0 else "#f1f5f9"
        parts.append(f'<rect x="0" y="{y0}" width="{width}" height="{lane_h}" fill="{bg}"/>')
        parts.append(f'<line x1="0" y1="{y0}" x2="{width}" y2="{y0}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="24" y="{y0+82}" font-size="15" font-weight="700" fill="#334155">{esc(lane)}</text>')

    edge_count = 0
    for node in visible:
        nid = str(node.get("id") or "")
        pid = str(node.get("parent_id") or "")
        if pid and pid in positions and nid in positions:
            x1, y1 = positions[pid]; x2, y2 = positions[nid]
            parts.append(f'<path d="M {x1+16:.1f} {y1:.1f} C {(x1+x2)/2:.1f} {y1:.1f}, {(x1+x2)/2:.1f} {y2:.1f}, {x2-16:.1f} {y2:.1f}" fill="none" stroke="#64748b" stroke-width="1.8" marker-end="url(#arrow)" opacity="0.72"/>')
            edge_count += 1
    if edge_count == 0:
        for a, b in zip(visible, visible[1:]):
            aid = str(a.get("id") or ""); bid = str(b.get("id") or "")
            if aid in positions and bid in positions:
                x1, y1 = positions[aid]; x2, y2 = positions[bid]
                parts.append(f'<path d="M {x1+16:.1f} {y1:.1f} C {(x1+x2)/2:.1f} {y1:.1f}, {(x1+x2)/2:.1f} {y2:.1f}, {x2-16:.1f} {y2:.1f}" fill="none" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="5 5" marker-end="url(#arrow)" opacity="0.55"/>')
                edge_count += 1

    for idx, node in enumerate(visible, start=1):
        nid = str(node.get("id") or idx)
        x, y = positions[nid]
        is_source = bool(node.get("is_suspected_source"))
        is_key = bool(node.get("is_key_node"))
        is_tampered = bool(node.get("suspected_tampering") or (node.get("tamper_analysis") or {}).get("is_tampered"))
        if is_source:
            color = "#16a34a"; fill = "#dcfce7"
        elif is_tampered:
            color = "#dc2626"; fill = "#fee2e2"
        elif is_key:
            color = "#2563eb"; fill = "#dbeafe"
        else:
            color = "#64748b"; fill = "#f8fafc"
        sim = node.get("similarity")
        try:
            sim_f = float(sim)
        except Exception:
            sim_f = 0.55
        r = max(14, min(28, 13 + sim_f * 22))
        title = esc(node.get("title") or "无标题")
        publisher = esc(node.get("publisher") or node.get("author") or "未知")
        time_s = esc(node.get("published_at") or "未知时间")
        url = esc(node.get("url") or "")
        parts.append(f"""
        <g filter="url(#shadow)">
          <title>#{idx} {title}\n发布者：{publisher}\n时间：{time_s}\n相似度：{esc(sim)}\n{url}</title>
          <circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{color}" stroke-width="3"/>
          <text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" font-size="11" font-weight="800" fill="{color}">{idx}</text>
        </g>
        <text x="{x:.1f}" y="{y+r+18:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="#0f172a">{publisher[:14]}</text>
        <text x="{x:.1f}" y="{y+r+34:.1f}" text-anchor="middle" font-size="10" fill="#64748b">{time_s[:16]}</text>
        """)

    parts.append("</svg></div>")
    return "".join(parts)

def card_metric(title: str, value: Any, hint: str = "") -> None:
    with ui.card().classes("mcard p-4 min-w-[110px] grow"):
        ui.html(f"<div style='font-size:.62rem;font-weight:650;color:#90897e;text-transform:uppercase;letter-spacing:.06em;'>{title}</div>")
        ui.html(f"<div style='font-size:1.55rem;font-weight:780;color:#17120e;letter-spacing:-.01em;line-height:1.1;'>{value}</div>")
        if hint:
            ui.html(f"<div style='font-size:.6rem;color:#90897e;opacity:.75;'>{hint}</div>")


def add_styles() -> None:
    ui.add_head_html("""
    <style>
      :root {
        --ink:#17120e;--sub:#5c5650;--dim:#90897e;--navy:#0f1d30;--navy2:#1e3a5c;
        --blue:#2e6894;--sage:#3b7056;--rust:#b04040;--amber:#c07838;
        --bg:#faf8f3;--warm:#f3efe6;--wht:#ffffff;--line:#e2ded6;
        --term:#13171d;--termfg:#bcc6d2;
      }
      body{background:var(--bg)!important;font-family:"Segoe UI",system-ui,-apple-system,sans-serif!important;color:var(--ink)!important;-webkit-font-smoothing:antialiased!important;}
      .glass-card,.card-raise{border:1px solid var(--line)!important;border-radius:10px!important;background:var(--wht)!important;box-shadow:0 2px 12px rgba(0,0,0,.045)!important;}
      .metric-card{border-radius:5px!important;background:var(--wht)!important;border:1px solid var(--line)!important;border-top:3px solid var(--navy2)!important;min-width:110px!important;}
      .step-pill,.spill{display:inline-flex!important;align-items:center!important;gap:6px!important;padding:6px 14px!important;border-radius:20px!important;border:1.5px solid var(--line)!important;background:var(--wht)!important;font-size:.78rem!important;font-weight:550!important;color:var(--dim)!important;}
      .step-running,.spill-go{border-color:var(--blue)!important;background:#e6eef7!important;color:var(--navy2)!important;font-weight:700!important;box-shadow:0 0 0 2px rgba(46,104,148,.10)!important;}
      .step-done,.spill-ok{border-color:var(--sage)!important;background:#eaf3ed!important;color:var(--sage)!important;}
      .step-error,.spill-err{border-color:var(--rust)!important;background:#faf0f0!important;color:var(--rust)!important;font-weight:650!important;}
      .hero-title{letter-spacing:-.025em!important;font-weight:800!important;font-size:2.6rem!important;line-height:1.05!important;}
      .mini-log,.logbox{background:var(--term)!important;color:var(--termfg)!important;border-radius:6px!important;padding:14px 16px!important;max-height:240px!important;overflow:auto!important;font-family:"Cascadia Code","Fira Code","JetBrains Mono","SF Mono",Consolas,monospace!important;font-size:11.5px!important;line-height:1.6!important;white-space:pre-wrap!important;border:1px solid rgba(255,255,255,.04)!important;}
      .link-button a{text-decoration:none!important;}
      .platform-table .q-table__middle{max-height:560px!important;}
      .platform-table td{max-width:230px!important;overflow:hidden!important;text-overflow:ellipsis!important;white-space:nowrap!important;font-size:.77rem!important;}
      .platform-table th{font-weight:650!important;font-size:.67rem!important;text-transform:uppercase!important;letter-spacing:.05em!important;color:var(--dim)!important;border-bottom:2px solid var(--line)!important;}
      .q-tabs{border-bottom:1.5px solid var(--line)!important;}
      .q-tab{font-size:.84rem!important;font-weight:550!important;color:var(--sub)!important;padding:10px 20px!important;}
      .q-tab--active{color:var(--navy2)!important;font-weight:700!important;border-bottom:2px solid var(--navy2)!important;}
      .q-linear-progress__track{background:#e7e3db!important;}
      .q-linear-progress__model{background:var(--navy2)!important;}
      .q-uploader{border:2px dashed #d0cbc1!important;border-radius:5px!important;background:#fdfcfb!important;}
      .q-uploader:hover{border-color:var(--navy2)!important;background:#eef3f8!important;}
      .q-expansion-item{border:1px solid var(--line)!important;border-radius:5px!important;margin-bottom:6px!important;background:var(--wht)!important;}
      @media(max-width:768px){.hide-on-mobile{display:none!important;}.hero-title{font-size:1.5rem!important;}}
    </style>
    """)


def update_step_labels(step_labels: dict[str, Any], active_step: str, status: str) -> None:
    """Update step pills: use inline .style() to bypass Quasar CSS."""
    active_index = STEP_ORDER.index(active_step) if active_step in STEP_ORDER else -1
    base = ("display:inline-flex;align-items:center;gap:6px;padding:7px 16px;"
            "border-radius:20px;border:1.5px solid #e2ded6;background:#fff;"
            "font-size:.78rem;font-weight:550;color:#90897e;")
    styles = {
        "running": base + "border-color:#2e6894;background:#e6eef7;color:#1e3a5c;font-weight:700;"
                   "box-shadow:0 0 0 2px rgba(46,104,148,.10);",
        "done": base + "border-color:#3b7056;background:#eaf3ed;color:#3b7056;",
        "error": base + "border-color:#b04040;background:#faf0f0;color:#b04040;font-weight:650;",
    }
    for i, step in enumerate(STEP_ORDER):
        lbl = step_labels[step]
        if status == "error":
            if i <= max(active_index, 0):
                lbl.set_text(f"✗ {STEP_LABELS[step]}"); lbl.style(styles["error"])
            else:
                lbl.set_text(f"○ {STEP_LABELS[step]}"); lbl.style(base)
        elif status == "done" or (active_index >= 0 and i < active_index):
            lbl.set_text(f"✓ {STEP_LABELS[step]}"); lbl.style(styles["done"])
        elif i == active_index and status == "running":
            lbl.set_text(f"◉ {STEP_LABELS[step]}"); lbl.style(styles["running"])
        else:
            lbl.set_text(f"○ {STEP_LABELS[step]}"); lbl.style(base)


def render_report(result_area: Any, report_dir_name: str) -> None:
    report_dir = OUTPUT_DIR / report_dir_name
    summary = read_json(report_dir / "summary.json", {})
    nodes = read_json(report_dir / "nodes_data.json", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(summary, dict):
        summary = {}

    summary["_report_dir"] = str(report_dir)
    overview = summarize_nodes(nodes, summary)
    result_area.clear()
    result_area.set_visibility(True)

    with result_area:
        with ui.row().classes("w-full items-center justify-between mb-2"):
            ui.markdown("### 分析结果").classes("m-0")
            with ui.row().classes("gap-2"):
                topo_path = report_dir / "topology.html"
                if topo_path.exists():
                    ui.link("全屏查看拓扑图", make_download_url(report_dir_name, "topology.html"), new_tab=True).classes("link-button")
                if (report_dir / "summary.json").exists():
                    ui.link("下载报告 JSON", make_download_url(report_dir_name, "summary.json"), new_tab=True)
                if (report_dir / "nodes_data.json").exists():
                    ui.link("下载节点数据", make_download_url(report_dir_name, "nodes_data.json"), new_tab=True)
                if (report_dir / "logs.txt").exists():
                    ui.link("下载日志", make_download_url(report_dir_name, "logs.txt"), new_tab=True)

        with ui.row().classes("w-full gap-3 flex-wrap"):
            card_metric("检索候选", overview["candidate_count"], "搜索引擎返回")
            card_metric("通过校验", overview["validated_count"], "相似/篡改候选")
            card_metric("时间证据", overview["with_time_count"], "可排时间线")
            card_metric("拓扑节点", overview["topology_node_count"], "参与展示")
            card_metric("拓扑边", overview["edge_count"], "传播/关系边")
            card_metric("关键节点", overview["key_count"], "高影响节点")

        with ui.card().classes("w-full glass-card p-4 mt-3"):
            ui.markdown(
                f"""
                **疑似最早来源**：{overview['earliest_publisher']}  
                **最早发布时间**：{overview['earliest_time']}  
                **标题**：{overview['earliest_title']}  
                **搜索引擎**：{overview['search_engines']}
                """
            )

        tabs = ui.tabs().classes("w-full mt-4")
        with tabs:
            tab_graph = ui.tab("传播拓扑")
            tab_timeline = ui.tab("时间线")
            tab_nodes = ui.tab("检索结果")
            tab_report = ui.tab("分析报告")
            tab_logs = ui.tab("日志摘要")

        with ui.tab_panels(tabs, value=tab_graph).classes("w-full"):
            with ui.tab_panel(tab_graph):
                topo_file = report_dir / "topology.html"
                topo_url = make_download_url(report_dir_name, "topology.html") if topo_file.exists() else ""
                with ui.row().classes("items-center gap-3 mb-3"):
                    if topo_file.exists():
                        ui.link("打开 G6 大屏交互版", topo_url, new_tab=True).classes("text-blue-700 underline")
                    ui.label(f"页面内预览节点：{len([n for n in nodes if isinstance(n, dict)])}").classes("text-slate-500")
                svg_html = build_native_topology_svg(nodes, summary)
                ui.html(svg_html).classes("w-full")

            with ui.tab_panel(tab_timeline):
                timeline_columns = [
                    {"name": "idx", "label": "#", "field": "idx", "align": "left"},
                    {"name": "time", "label": "时间", "field": "time", "align": "left", "sortable": True},
                    {"name": "platform", "label": "平台", "field": "platform", "align": "left"},
                    {"name": "publisher", "label": "发布者", "field": "publisher", "align": "left"},
                    {"name": "title", "label": "标题", "field": "title", "align": "left"},
                    {"name": "similarity", "label": "相似度", "field": "similarity", "align": "left", "sortable": True},
                    {"name": "source", "label": "疑似源头", "field": "source", "align": "left"},
                    {"name": "key", "label": "关键节点", "field": "key", "align": "left"},
                ]
                ui.table(columns=timeline_columns, rows=top_timeline_rows(nodes), pagination=12).classes("w-full")

            with ui.tab_panel(tab_nodes):
                node_columns = [
                    {"name": "idx", "label": "#", "field": "idx", "align": "left"},
                    {"name": "time", "label": "时间", "field": "time", "align": "left", "sortable": True},
                    {"name": "platform", "label": "平台", "field": "platform", "align": "left"},
                    {"name": "publisher", "label": "发布者", "field": "publisher", "align": "left"},
                    {"name": "title", "label": "标题", "field": "title", "align": "left"},
                    {"name": "similarity", "label": "相似度", "field": "similarity", "align": "left", "sortable": True},
                    {"name": "source_score", "label": "源头分数", "field": "source_score", "align": "left", "sortable": True},
                    {"name": "repost", "label": "转发", "field": "repost", "align": "left", "sortable": True},
                    {"name": "comment", "label": "评论", "field": "comment", "align": "left", "sortable": True},
                    {"name": "like", "label": "点赞", "field": "like", "align": "left", "sortable": True},
                    {"name": "role", "label": "角色", "field": "role", "align": "left"},
                ]
                rows = [compact_node_row(n, i + 1) for i, n in enumerate(nodes)]
                ui.table(columns=node_columns, rows=rows, pagination=15).classes("w-full")

            with ui.tab_panel(tab_report):
                report_text = summary.get("report", "未生成报告。")
                ui.markdown(report_text).classes("prose max-w-none")

            with ui.tab_panel(tab_logs):
                log_text = ""
                log_path = report_dir / "logs.txt"
                if log_path.exists():
                    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                    log_text = "\n".join(lines[-120:])
                ui.html(f"<div class='mini-log'>{html_escape(log_text or '暂无日志。')}</div>")


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


# 先保存 v3 的纯节点推断 SVG 作为兜底；v4/v5 会优先读取 G6 payload。
_build_native_topology_svg_v3 = build_native_topology_svg

# =============================================================================
# v4 override：页面内 SVG 复用 G6 payload，保证边关系与大屏一致
# =============================================================================

def _read_topology_payload_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    report_dir_value = summary.get("_report_dir") if isinstance(summary, dict) else None
    if report_dir_value:
        html_path = Path(str(report_dir_value)) / "topology.html"
        if html_path.exists():
            try:
                html = html_path.read_text(encoding="utf-8", errors="ignore")
                match = re.search(r"const\s+payload\s*=\s*(\{.*?\});\s*\n", html, re.S)
                if match:
                    payload = json.loads(match.group(1))
                    if isinstance(payload, dict):
                        return payload
            except Exception:
                pass
    topo = summary.get("topology_data") if isinstance(summary, dict) else None
    return topo if isinstance(topo, dict) else {}


def _original_summarize_nodes_v3(nodes: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    retrieval = summary.get("retrieval_summary", {}) if isinstance(summary, dict) else {}
    validation = summary.get("validation_summary", {}) if isinstance(summary, dict) else {}
    analysis = summary.get("analysis_summary", {}) if isinstance(summary, dict) else {}
    with_time = sum(1 for n in nodes if n.get("published_at"))
    suspected_sources = [n for n in nodes if n.get("is_suspected_source")]
    key_nodes = [n for n in nodes if n.get("is_key_node")]
    tampered_nodes = [n for n in nodes if n.get("suspected_tampering") or (n.get("tamper_analysis") or {}).get("is_tampered")]
    earliest = None
    dated_nodes = [n for n in nodes if n.get("published_at")]
    if dated_nodes:
        earliest = sorted(dated_nodes, key=lambda n: str(n.get("published_at")))[0]
    payload = _read_topology_payload_from_summary(summary)
    topo_nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
    topo_edges = payload.get("edges", []) if isinstance(payload, dict) else []
    return {
        "candidate_count": retrieval.get("result_count") or retrieval.get("candidate_count") or "-",
        "validated_count": validation.get("validated_count") or len(nodes),
        "with_time_count": analysis.get("with_time_count") or with_time,
        "node_count": len(nodes),
        "edge_count": len(topo_edges) if topo_edges else analysis.get("topology_edge_count", "-"),
        "key_count": len(key_nodes) or len(analysis.get("key_node_ids", []) or []),
        "tampered_count": len(tampered_nodes) or len(analysis.get("tampered_node_ids", []) or []),
        "source_count": len(suspected_sources),
        "earliest_time": earliest.get("published_at") if earliest else "-",
        "earliest_publisher": (earliest.get("publisher") or earliest.get("author") or earliest.get("domain")) if earliest else "-",
        "earliest_title": earliest.get("title") if earliest else "-",
        "search_engines": ", ".join((retrieval.get("per_engine_counts") or {}).keys()) or "-",
        "topology_node_count": len(topo_nodes) if topo_nodes else len(nodes),
    }


def summarize_nodes(nodes: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return _original_summarize_nodes_v3(nodes, summary)


def build_native_topology_svg(nodes: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    payload = _read_topology_payload_from_summary(summary)
    g6_nodes = [n for n in payload.get("nodes", []) if isinstance(n, dict)] if payload else []
    g6_edges = [e for e in payload.get("edges", []) if isinstance(e, dict)] if payload else []
    if not g6_nodes:
        return _build_native_topology_svg_v3(nodes, summary)

    def esc(v: Any) -> str:
        return html_escape(str(v if v is not None else ""))

    xs, ys = [], []
    for n in g6_nodes:
        try:
            xs.append(float(n.get("x", 0)))
            ys.append(float(n.get("y", 0)))
        except Exception:
            pass
    min_x, max_x = (min(xs), max(xs)) if xs else (0, 1200)
    min_y, max_y = (min(ys), max(ys)) if ys else (0, 800)
    pad_x, pad_y = 170, 120
    width = int(max(1200, max_x - min_x + pad_x * 2))
    height = int(max(680, max_y - min_y + pad_y * 2))
    positions: dict[str, tuple[float, float]] = {}
    for n in g6_nodes:
        nid = str(n.get("id") or "")
        if not nid:
            continue
        try:
            positions[nid] = (float(n.get("x", 0)) - min_x + pad_x, float(n.get("y", 0)) - min_y + pad_y)
        except Exception:
            positions[nid] = (pad_x, pad_y)

    parts: list[str] = []
    parts.append(f'''
    <div style="width:100%;overflow:auto;border:1px solid #e5e7eb;border-radius:16px;background:#ffffff;">
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="display:block;">
      <defs>
        <marker id="arrow-g6" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
        </marker>
        <filter id="shadow-g6" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.12"/>
        </filter>
      </defs>
      <rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc"/>
      <text x="24" y="36" font-size="19" font-weight="800" fill="#0f172a">页面内传播拓扑预览</text>
    ''')
    lanes = payload.get("lanes") if isinstance(payload.get("lanes"), list) else []
    for i, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            continue
        try:
            y = float(lane.get("y", 0)) - min_y + pad_y
            h = float(lane.get("height", 150))
        except Exception:
            continue
        label = esc(lane.get("label") or lane.get("name") or f"泳道{i+1}")
        bg = "#ffffff" if i % 2 == 0 else "#f1f5f9"
        parts.append(f'<rect x="0" y="{y:.1f}" width="{width}" height="{h:.1f}" fill="{bg}" opacity="0.78"/>')
        parts.append(f'<line x1="0" y1="{y:.1f}" x2="{width}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="24" y="{y+82:.1f}" font-size="15" font-weight="700" fill="#334155">{label}</text>')

    for edge in g6_edges:
        data = edge.get("data") if isinstance(edge.get("data"), dict) else {}
        source = str(edge.get("source") or data.get("source") or "")
        target = str(edge.get("target") or data.get("target") or "")
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        style = edge.get("style") if isinstance(edge.get("style"), dict) else {}
        stroke = esc(style.get("stroke") or "#64748b")
        try:
            line_w = float(style.get("lineWidth", 1.8))
        except Exception:
            line_w = 1.8
        dash = style.get("lineDash")
        dash_attr = ""
        if isinstance(dash, list) and dash:
            dash_attr = f' stroke-dasharray="{esc(" ".join(str(x) for x in dash))}"'
        directed = bool(data.get("directed") or style.get("endArrow"))
        arrow_attr = ' marker-end="url(#arrow-g6)"' if directed else ""
        method = esc(data.get("method") or data.get("edge_type") or edge.get("label") or "关系边")
        evidence = esc(data.get("evidence") or "")
        parts.append(
            f'<path d="M {x1:.1f} {y1:.1f} C {(x1+x2)/2:.1f} {y1:.1f}, {(x1+x2)/2:.1f} {y2:.1f}, {x2:.1f} {y2:.1f}" '
            f'fill="none" stroke="{stroke}" stroke-width="{line_w:.1f}"{dash_attr}{arrow_attr} opacity="0.82">'
            f'<title>{source} → {target}\n{method}\n{evidence}</title></path>'
        )

    for idx, node in enumerate(g6_nodes, start=1):
        nid = str(node.get("id") or idx)
        if nid not in positions:
            continue
        x, y = positions[nid]
        data = node.get("data") if isinstance(node.get("data"), dict) else {}
        style = node.get("style") if isinstance(node.get("style"), dict) else {}
        fill = esc(style.get("fill") or "#f8fafc")
        stroke = esc(style.get("stroke") or "#64748b")
        try:
            size = float(node.get("size", 36))
        except Exception:
            size = 36
        r = max(13, min(33, size / 2))
        author = data.get("author") if isinstance(data.get("author"), dict) else {}
        publisher = esc(author.get("name") or data.get("publisher") or "未知")
        time_s = esc(data.get("publish_time") or data.get("published_at") or "未知时间")
        sim = esc(data.get("similarity") or "")
        title = esc(data.get("title") or node.get("label") or nid)
        badges = data.get("badges") if isinstance(data.get("badges"), list) else []
        badge_text = " / ".join(str(x) for x in badges[:3])
        label_main = esc(publisher[:14] or nid)
        parts.append(f'''
        <g filter="url(#shadow-g6)">
          <title>#{idx} {title}\nID：{esc(nid)}\n发布者：{publisher}\n时间：{time_s}\n相似度：{sim}\n{esc(badge_text)}</title>
          <circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>
          <text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" font-size="11" font-weight="800" fill="{stroke}">{idx}</text>
        </g>
        <text x="{x:.1f}" y="{y+r+18:.1f}" text-anchor="middle" font-size="11" font-weight="700" fill="#0f172a">{label_main}</text>
        <text x="{x:.1f}" y="{y+r+34:.1f}" text-anchor="middle" font-size="10" fill="#64748b">{time_s[:16]}</text>
        ''')
    legend_y = 82
    lx = 24
    for c, t in [("#1f9d68", "疑似源头"), ("#d95f35", "疑似篡改"), ("#1d5f99", "关键节点"), ("#ff4d64", "重复簇关系")]:
        parts.append(f'<circle cx="{lx}" cy="{legend_y}" r="6" fill="{c}"/><text x="{lx+12}" y="{legend_y+4}" font-size="12" fill="#334155">{t}</text>')
        lx += 130
    parts.append("</svg></div>")
    return "".join(parts)

# build_native_topology_svg 已覆盖为 G6 payload 优先版本；兜底函数在 v4 override 前已保存。


# =============================================================================
# v5 override：按用户要求重构结果栏目
# 栏目顺序：报告 / 拓扑图 / 检索结果（分平台）/ 日志 / 原始状态
# =============================================================================


def _safe_load_report_files(report_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    summary = read_json(report_dir / "summary.json", {})
    nodes = read_json(report_dir / "nodes_data.json", [])
    state_dump = read_json(report_dir / "state_dump.json", {})
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(nodes, list):
        nodes = []
    nodes = [n for n in nodes if isinstance(n, dict)]
    if not isinstance(state_dump, dict):
        state_dump = {}
    return summary, nodes, state_dump


def truncate_raw_for_display(obj: Any, max_items: int = 40, max_str_len: int = 260, depth: int = 0, max_depth: int = 5) -> Any:
    """前端原始状态预览：保留结构，但限制体积，避免浏览器卡死。"""
    if depth >= max_depth:
        if isinstance(obj, (dict, list)):
            return f"<{type(obj).__name__} truncated at depth {max_depth}>"
        return obj
    if isinstance(obj, dict):
        items = list(obj.items())
        kept = {str(k): truncate_raw_for_display(v, max_items, max_str_len, depth + 1, max_depth) for k, v in items[:max_items]}
        if len(items) > max_items:
            kept["_truncated_keys"] = len(items) - max_items
        return kept
    if isinstance(obj, list):
        kept = [truncate_raw_for_display(x, max_items, max_str_len, depth + 1, max_depth) for x in obj[:max_items]]
        if len(obj) > max_items:
            kept.append(f"_truncated_{len(obj) - max_items}_items")
        return kept
    if isinstance(obj, str) and len(obj) > max_str_len:
        return obj[:max_str_len] + "..."
    return obj


def pretty_json_for_html(obj: Any, *, max_items: int = 40, max_str_len: int = 260, max_depth: int = 5) -> str:
    data = truncate_raw_for_display(obj, max_items=max_items, max_str_len=max_str_len, max_depth=max_depth)
    return html_escape(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default))


def platform_group_name(node: dict[str, Any]) -> str:
    platform_raw = str(node.get("platform") or "")
    platform = platform_raw.lower()
    url = str(node.get("url") or node.get("canonical_url") or "").lower()
    family = str(node.get("platform_family") or "").lower()
    source = str(node.get("source") or node.get("engine") or "").lower()
    text = " ".join([platform, url, family, source, platform_raw]).lower()
    if "weibo" in text or "微博" in platform_raw:
        return "微博"
    if "xiaohongshu" in text or "xhs" in text or "小红书" in platform_raw:
        return "小红书"
    if "douyin" in text or "抖音" in platform_raw:
        return "抖音"
    if "bilibili" in text or "b站" in platform_raw or "哔哩" in platform_raw:
        return "B站"
    if family in {"news", "media", "baidu_media"} or "news" in text or "新闻" in platform_raw:
        return "新闻/媒体"
    if family in {"forum", "bbs"} or "forum" in text or "论坛" in platform_raw:
        return "论坛/社区"
    if family in {"blog"} or "blog" in text or "博客" in platform_raw:
        return "博客/文章"
    return "其他外部"


def group_nodes_for_display(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        groups.setdefault(platform_group_name(node), []).append(node)
    preferred = ["微博", "小红书", "抖音", "B站", "新闻/媒体", "论坛/社区", "博客/文章", "其他外部"]
    ordered: dict[str, list[dict[str, Any]]] = {}
    for name in preferred:
        if name in groups:
            ordered[name] = groups.pop(name)
    for name in sorted(groups):
        ordered[name] = groups[name]
    return ordered


def render_download_buttons(report_dir_name: str, report_dir: Path) -> None:
    with ui.row().classes("gap-2 flex-wrap"):
        file_specs = [
            ("下载报告 JSON", "summary.json"),
            ("下载节点数据", "nodes_data.json"),
            ("下载原始状态", "state_dump.json"),
            ("下载关键日志", "logs.txt"),
            ("下载完整调试日志", "full_logs.txt"),
        ]
        for label, filename in file_specs:
            if (report_dir / filename).exists():
                ui.link(label, make_download_url(report_dir_name, filename), new_tab=True).classes("text-blue-700 underline text-sm")
        if (report_dir / "topology.html").exists():
            ui.link("打开 G6 大屏拓扑", make_download_url(report_dir_name, "topology.html"), new_tab=True).classes("text-blue-700 underline text-sm")


def render_report_tab(summary: dict[str, Any], nodes: list[dict[str, Any]], overview: dict[str, Any]) -> None:
    with ui.row().classes("w-full gap-3 flex-wrap"):
        card_metric("检索候选", overview["candidate_count"], "搜索引擎返回")
        card_metric("通过校验", overview["validated_count"], "相似/篡改候选")
        card_metric("时间证据", overview["with_time_count"], "可排时间线")
        card_metric("拓扑节点", overview["topology_node_count"], "参与展示")
        card_metric("拓扑边", overview["edge_count"], "传播/关系边")
        card_metric("关键节点", overview["key_count"], "高影响节点")

    with ui.card().classes("w-full glass-card p-4 mt-3"):
        ui.markdown(
            f"""
            **疑似最早来源**：{overview['earliest_publisher']}  
            **最早发布时间**：{overview['earliest_time']}  
            **标题**：{overview['earliest_title']}  
            **搜索引擎**：{overview['search_engines']}
            """
        )

    report_text = summary.get("report", "未生成报告。")
    ui.markdown(report_text).classes("prose max-w-none mt-4")


def render_topology_tab(report_dir_name: str, report_dir: Path, summary: dict[str, Any], nodes: list[dict[str, Any]], overview: dict[str, Any]) -> None:
    topo_file = report_dir / "topology.html"
    topo_url = make_download_url(report_dir_name, "topology.html") if topo_file.exists() else ""
    ui.markdown("### 传播拓扑图").classes("m-0")
    with ui.row().classes("items-center gap-3 mb-3 flex-wrap"):
        ui.label(f"拓扑节点：{overview['topology_node_count']} / 拓扑边：{overview['edge_count']}").classes("text-slate-600")
        if topo_file.exists():
            ui.link("打开 G6 大屏交互版", topo_url, new_tab=True).classes("text-blue-700 underline")
    svg_html = build_native_topology_svg(nodes, summary)
    ui.html(svg_html).classes("w-full")


def render_platform_table(group_name: str, group_nodes: list[dict[str, Any]]) -> None:
    rows = [compact_node_row(n, i + 1) for i, n in enumerate(group_nodes)]
    columns = [
        {"name": "idx", "label": "#", "field": "idx", "align": "left", "sortable": True},
        {"name": "time", "label": "时间", "field": "time", "align": "left", "sortable": True},
        {"name": "publisher", "label": "发布者", "field": "publisher", "align": "left", "sortable": True},
        {"name": "title", "label": "标题", "field": "title", "align": "left"},
        {"name": "similarity", "label": "相似度", "field": "similarity", "align": "left", "sortable": True},
        {"name": "source_score", "label": "源头分数", "field": "source_score", "align": "left", "sortable": True},
        {"name": "repost", "label": "转发", "field": "repost", "align": "left", "sortable": True},
        {"name": "comment", "label": "评论", "field": "comment", "align": "left", "sortable": True},
        {"name": "like", "label": "点赞", "field": "like", "align": "left", "sortable": True},
        {"name": "source", "label": "疑似源头", "field": "source", "align": "left", "sortable": True},
        {"name": "key", "label": "关键节点", "field": "key", "align": "left", "sortable": True},
        {"name": "role", "label": "角色", "field": "role", "align": "left"},
        {"name": "url", "label": "链接", "field": "url", "align": "left"},
    ]
    ui.markdown(f"#### {group_name}：{len(group_nodes)} 条").classes("mt-0")
    ui.table(columns=columns, rows=rows, pagination=15).classes("w-full platform-table")


def render_search_results_tab(nodes: list[dict[str, Any]]) -> None:
    groups = group_nodes_for_display(nodes)
    if not groups:
        ui.markdown("暂无检索结果。")
        return
    with ui.row().classes("w-full gap-3 flex-wrap mb-3"):
        for name, items in groups.items():
            ui.badge(f"{name} {len(items)}", color="blue" if name in {"微博", "小红书"} else "grey")
    group_tabs = ui.tabs().classes("w-full")
    with group_tabs:
        all_tab = ui.tab(f"全部 ({len(nodes)})")
        tab_map: dict[str, Any] = {}
        for name, items in groups.items():
            tab_map[name] = ui.tab(f"{name} ({len(items)})")
    with ui.tab_panels(group_tabs, value=all_tab).classes("w-full"):
        with ui.tab_panel(all_tab):
            render_platform_table("全部平台", nodes)
        for name, tab in tab_map.items():
            with ui.tab_panel(tab):
                render_platform_table(name, groups[name])


def _line_time(line: str) -> str:
    m = re.search(r"\[(\d{2}:\d{2}:\d{2})\]", line)
    return m.group(1) if m else "--:--:--"


def _format_stage_line(ts: str, stage: str, agent: str, status: str, detail: str = "") -> str:
    suffix = f"，{detail}" if detail else ""
    return f"[{ts}] {stage} / {agent}：{status}{suffix}"


def _dedupe_keep_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        text = str(line).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _has_v6_key_log_shape(lines: list[str]) -> bool:
    stages = ["上传图片", "图片信息提取", "以图搜图", "相似度校验/去重", "内容提取/传播分析", "生成报告"]
    hit = sum(1 for line in lines for stage in stages if stage in line and ("开始" in line or "完成" in line))
    noisy = sum(1 for line in lines if "处理候选" in line or "强确定去重合并" in line)
    return hit >= 4 and noisy == 0


def _synthesize_key_logs_from_raw(lines: list[str], summary: dict[str, Any] | None = None, nodes: list[dict[str, Any]] | None = None) -> list[str]:
    """从旧版完整日志中提炼展示用阶段日志。

    只输出每个智能体/阶段的开始与完成，不展示候选逐条处理、重复去重行等细节。
    """
    if summary is None:
        summary = {}
    nodes = nodes or []
    result: list[str] = []

    def first_time_contains(*terms: str) -> str | None:
        for line in lines:
            if all(term in line for term in terms):
                return _line_time(line)
        return None

    def last_time_contains(*terms: str) -> str | None:
        for line in reversed(lines):
            if all(term in line for term in terms):
                return _line_time(line)
        return None

    # 上传与目标图信息
    t_upload = first_time_contains("upload_node") or first_time_contains("已接收目标图片") or ( _line_time(lines[0]) if lines else "--:--:--")
    result.append(_format_stage_line(t_upload, "上传图片", "upload_node", "开始"))
    result.append(_format_stage_line(t_upload, "上传图片", "upload_node", "完成"))
    result.append(_format_stage_line(t_upload, "图片信息提取", "upload_node", "开始"))
    result.append(_format_stage_line(t_upload, "图片信息提取", "upload_node", "完成"))

    # 检索
    t_retr_start = first_time_contains("retrieve_node", "starting") or first_time_contains("retrieve_node") or t_upload
    result.append(_format_stage_line(t_retr_start, "以图搜图", "retrieve_node", "开始"))
    collected_line = next((line for line in lines if "retrieve_node" in line and "collected" in line), "")
    t_retr_done = _line_time(collected_line) if collected_line else (last_time_contains("retrieve_node") or t_retr_start)
    retrieval = summary.get("retrieval_summary", {}) if isinstance(summary, dict) else {}
    per_engine = retrieval.get("per_engine_counts", {}) if isinstance(retrieval, dict) else {}
    count = retrieval.get("result_count") if isinstance(retrieval, dict) else None
    if not count and collected_line:
        m = re.search(r"collected\s+(\d+)", collected_line)
        count = m.group(1) if m else None
    engine_text = ""
    if isinstance(per_engine, dict) and per_engine:
        engine_text = "，引擎=" + ", ".join(f"{k}:{v}" for k, v in per_engine.items())
    detail = f"候选={count}条{engine_text}" if count else engine_text.lstrip("，")
    result.append(_format_stage_line(t_retr_done, "以图搜图", "retrieve_node", "完成", detail))

    # 校验/去重
    t_val_start = first_time_contains("validate_node", "处理候选") or first_time_contains("validate_node") or t_retr_done
    result.append(_format_stage_line(t_val_start, "相似度校验/去重", "validate_node", "开始"))
    val_line = next((line for line in reversed(lines) if "validate_node" in line and ("输出" in line or "校验完成" in line or "强确定去重" in line)), "")
    t_val_done = _line_time(val_line) if val_line else (last_time_contains("validate_node") or t_val_start)
    validation = summary.get("validation_summary", {}) if isinstance(summary, dict) else {}
    val_detail_parts: list[str] = []
    if isinstance(validation, dict):
        if validation.get("validated_count") is not None:
            val_detail_parts.append(f"通过={validation.get('validated_count')}条")
        if validation.get("rejected_count") is not None:
            val_detail_parts.append(f"拒绝={validation.get('rejected_count')}条")
        if validation.get("deduplicated_count") is not None:
            val_detail_parts.append(f"去重={validation.get('deduplicated_count')}条")
    if not val_detail_parts and val_line:
        m = re.search(r"输出\s*(\d+)\s*个节点", val_line)
        if m:
            val_detail_parts.append(f"输出={m.group(1)}个节点")
    result.append(_format_stage_line(t_val_done, "相似度校验/去重", "validate_node", "完成", "，".join(val_detail_parts)))

    # 分析
    t_parse_start = first_time_contains("parse_node") or t_val_done
    result.append(_format_stage_line(t_parse_start, "内容提取/传播分析", "parse_node", "开始"))
    t_parse_done = last_time_contains("parse_node") or t_parse_start
    analysis = summary.get("analysis_summary", {}) if isinstance(summary, dict) else {}
    with_time = analysis.get("with_time_count") if isinstance(analysis, dict) else None
    if with_time is None:
        with_time = sum(1 for n in nodes if isinstance(n, dict) and n.get("published_at"))
    key_count = len(analysis.get("key_node_ids", []) or []) if isinstance(analysis, dict) else 0
    result.append(_format_stage_line(t_parse_done, "内容提取/传播分析", "parse_node", "完成", f"时间证据={with_time}条，关键节点={key_count}个"))

    # 报告
    t_report_start = first_time_contains("report_node") or t_parse_done
    result.append(_format_stage_line(t_report_start, "报告生成", "report_node", "开始"))
    t_report_done = last_time_contains("report_node") or last_time_contains("完整报告已保存") or ( _line_time(lines[-1]) if lines else t_report_start)
    result.append(_format_stage_line(t_report_done, "报告生成", "report_node", "完成"))

    return _dedupe_keep_order(result)


def load_key_log_lines(report_dir: Path, summary: dict[str, Any] | None = None, nodes: list[dict[str, Any]] | None = None) -> list[str]:
    """读取前端展示用关键日志。

    真实运行优先使用 logs.txt 中的阶段日志；旧版完整日志会被归纳成阶段开始/完成。
    """
    if isinstance(summary, dict):
        key_logs = summary.get("key_logs")
        if isinstance(key_logs, list) and key_logs and _has_v6_key_log_shape([str(x) for x in key_logs]):
            return _dedupe_keep_order([str(x) for x in key_logs])

    log_path = report_dir / "logs.txt"
    full_log_path = report_dir / "full_logs.txt"
    source_path = log_path if log_path.exists() else full_log_path
    if not source_path.exists():
        return []
    lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return []
    if _has_v6_key_log_shape(lines):
        return _dedupe_keep_order(lines)
    return _synthesize_key_logs_from_raw(lines, summary, nodes)


def render_logs_tab(report_dir_name: str, report_dir: Path, summary: dict[str, Any] | None = None) -> None:
    key_lines = load_key_log_lines(report_dir, summary)
    full_log_exists = (report_dir / "full_logs.txt").exists()
    with ui.row().classes("items-center justify-between w-full mb-2"):
        ui.markdown(f"### 关键运行日志（{len(key_lines)} 条）").classes("m-0")
        with ui.row().classes("gap-3"):
            if (report_dir / "logs.txt").exists():
                ui.link("下载关键日志", make_download_url(report_dir_name, "logs.txt"), new_tab=True).classes("text-blue-700 underline")
            if full_log_exists:
                ui.link("下载完整调试日志", make_download_url(report_dir_name, "full_logs.txt"), new_tab=True).classes("text-blue-700 underline")
    log_text = "\n".join(key_lines) if key_lines else "暂无关键日志。"
    ui.html(f"<div class='mini-log' style='max-height:560px;'>{html_escape(log_text)}</div>").classes("w-full")


def render_raw_state_tab(report_dir_name: str, report_dir: Path, summary: dict[str, Any], nodes: list[dict[str, Any]], state_dump: dict[str, Any]) -> None:
    ui.markdown("### 原始状态").classes("m-0")
    render_download_buttons(report_dir_name, report_dir)

    with ui.expansion("summary.json 预览", value=True).classes("w-full"):
        ui.html(f"<pre class='mini-log' style='max-height:520px;'>{pretty_json_for_html(summary, max_items=60, max_depth=6)}</pre>").classes("w-full")
    with ui.expansion("nodes_data.json 预览（截断）", value=False).classes("w-full"):
        ui.html(f"<pre class='mini-log' style='max-height:520px;'>{pretty_json_for_html(nodes, max_items=10, max_str_len=220, max_depth=4)}</pre>").classes("w-full")
    if state_dump:
        with ui.expansion("state_dump.json 预览（真实运行完整状态，截断显示）", value=False).classes("w-full"):
            ui.html(f"<pre class='mini-log' style='max-height:520px;'>{pretty_json_for_html(state_dump, max_items=30, max_str_len=220, max_depth=4)}</pre>").classes("w-full")
    else:
        ui.markdown("未找到 `state_dump.json`。")


def render_report_v5(result_area: Any, report_dir_name: str) -> None:
    report_dir = OUTPUT_DIR / report_dir_name
    summary, nodes, state_dump = _safe_load_report_files(report_dir)
    summary["_report_dir"] = str(report_dir)
    overview = summarize_nodes(nodes, summary)

    result_area.clear()
    result_area.set_visibility(True)

    with result_area:
        with ui.row().classes("w-full items-center justify-between mb-2"):
            ui.markdown("### 分析结果").classes("m-0")
            render_download_buttons(report_dir_name, report_dir)

        tabs = ui.tabs().classes("w-full mt-4")
        with tabs:
            tab_report = ui.tab("报告")
            tab_graph = ui.tab("拓扑图")
            tab_search = ui.tab("检索结果")
            tab_logs = ui.tab("日志")
            tab_raw = ui.tab("原始状态")

        with ui.tab_panels(tabs, value=tab_report).classes("w-full"):
            with ui.tab_panel(tab_report):
                render_report_tab(summary, nodes, overview)
            with ui.tab_panel(tab_graph):
                render_topology_tab(report_dir_name, report_dir, summary, nodes, overview)
            with ui.tab_panel(tab_search):
                render_search_results_tab(nodes)
            with ui.tab_panel(tab_logs):
                render_logs_tab(report_dir_name, report_dir, summary)
            with ui.tab_panel(tab_raw):
                render_raw_state_tab(report_dir_name, report_dir, summary, nodes, state_dump)


# 覆盖旧版 render_report，主页面和真实工作流完成后都会进入这套 5 栏结果渲染。
render_report = render_report_v5

# =============================================================================
# 主页面
# =============================================================================



@ui.page("/")
def index() -> None:
    RUNTIME.active_report_dir = None
    write_status(status="idle", step="idle", message="Idle", progress=0.0, report_dir=None)

    with ui.column().classes("w-full max-w-5xl mx-auto p-4"):
        ui.label("Image Provenance Agent").classes("text-h4")
        ui.label("Upload -> Search -> Validate -> Analyze -> Report")

        if IMPORT_ERROR is not None:
            ui.warning(f"Import error: {IMPORT_ERROR}")

        with ui.row().classes("w-full gap-4 mt-4"):
            with ui.card().classes("w-full md:w-1/3 p-4"):
                ui.label("Upload Image").classes("text-h6")
                preview = ui.image().classes("w-full")
                preview.set_visibility(False)
                file_info = ui.label("No file selected")

                def handle_upload(e: Any) -> None:
                    file = e.file
                    data = getattr(file, "_data", None)
                    if not data:
                        try: data = file.read()
                        except Exception: data = b""
                    if not data or not isinstance(data, bytes):
                        ui.notify("No file content", type="negative"); return
                    RUNTIME.uploaded_content = data
                    RUNTIME.uploaded_filename = file.name or "uploaded.jpg"
                    mime = getattr(file, "content_type", None) or "image/jpeg"
                    b64 = base64.b64encode(data).decode("ascii")
                    preview.set_source(f"data:{mime};base64,{b64}")
                    preview.set_visibility(True)
                    file_info.set_text(f"{RUNTIME.uploaded_filename} ({len(data)/1024:.1f} KB)")
                    ui.notify("Image uploaded", type="positive")

                upload = ui.upload(label="Select image", auto_upload=True, on_upload=handle_upload).classes("w-full")
                upload.props("accept=.png,.jpg,.jpeg,.webp max-files=1")

            with ui.card().classes("w-full md:w-2/3 p-4"):
                ui.label("Pipeline").classes("text-h6")
                with ui.row().classes("gap-2 flex-wrap"):
                    step_labels = {}
                    for step in STEP_ORDER:
                        step_labels[step] = ui.label(f"O {STEP_LABELS[step]}")
                progress_bar = ui.linear_progress(0).classes("w-full mt-2")
                status_label = ui.label("Idle")
                with ui.row().classes("gap-2 mt-2"):
                    start_btn = ui.button("Start", icon="play_arrow", color="primary")
                    load_latest_btn = ui.button("Load Report", icon="folder_open")
                    test_btn = ui.button("Import Test", icon="science")
                mini_log = ui.log().classes("w-full mt-2")

        result_area = ui.column().classes("w-full mt-4")
        result_area.set_visibility(False)

        def on_start_click() -> None:
            if RUNTIME.running: ui.notify("Already running", type="warning"); return
            if not RUNTIME.uploaded_content: ui.notify("Upload image first", type="warning"); return
            RUNTIME.running = True; start_btn.disable(); result_area.set_visibility(False)
            write_status(status="running", step="upload", message="Starting", progress=0.01)
            asyncio.create_task(_run_workflow())

        async def _run_workflow() -> None:
            try: await execute_workflow(RUNTIME.uploaded_content, RUNTIME.uploaded_filename)
            finally: RUNTIME.running = False; start_btn.enable()

        def load_latest() -> None:
            report = latest_report_dir()
            if not report: ui.notify("No report found", type="warning"); return
            RUNTIME.active_report_dir = report.name
            render_report(result_area, report.name)
            write_status(status="done", step="report", message=f"Loaded: {report.name}", progress=1.0, report_dir=report.name)

        def import_test() -> None:
            name = copy_uploaded_test_files_to_output()
            if not name: ui.notify("No test data", type="warning"); return
            RUNTIME.active_report_dir = name
            render_report(result_area, name)
            write_status(status="done", step="report", message=f"Imported: {name}", progress=1.0, report_dir=name)
            ui.notify("Test data loaded", type="positive")

        start_btn.on_click(on_start_click)
        load_latest_btn.on_click(load_latest)
        test_btn.on_click(import_test)

        def refresh_status() -> None:
            s = read_status(); status = s.get("status", "idle"); step = s.get("step", "idle")
            msg = s.get("message", ""); progress = float(s.get("progress") or 0.0)
            status_label.set_text(msg or "Idle"); progress_bar.value = progress
            update_step_labels(step_labels, step, status)
            logs = s.get("recent_logs") or RUNTIME.latest_log_lines or []
            if logs: mini_log.set_content(chr(10).join(str(x) for x in logs[-40:]))
            report_dir_name = s.get("report_dir")
            if status == "done" and report_dir_name and report_dir_name != RUNTIME.active_report_dir:
                RUNTIME.active_report_dir = report_dir_name
                render_report(result_area, report_dir_name)
                ui.notify("Analysis complete", type="positive")
            if status == "error": start_btn.enable(); RUNTIME.running = False

        ui.timer(1.0, refresh_status)

        if "--test" in sys.argv:
            def auto_test_load() -> None:
                report = latest_report_dir()
                if not report:
                    imported = copy_uploaded_test_files_to_output()
                    report = OUTPUT_DIR / imported if imported else None
                if report and report.exists():
                    RUNTIME.active_report_dir = report.name
                    render_report(result_area, report.name)
                    write_status(status="done", step="report", message=f"Test: {report.name}", progress=1.0, report_dir=report.name)
            ui.timer(0.4, auto_test_load, once=True)


if __name__ in {"__main__", "__mp_main__"}:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    app.add_static_files("/report_files", OUTPUT_DIR.as_posix())
    # 如果你需要局域网访问，把 host 改成 0.0.0.0。
    ui.run(
        title="图片溯源智能体",
        favicon="🔍",
        host=os.getenv("NICEGUI_HOST", "127.0.0.1"),
        port=int(os.getenv("NICEGUI_PORT", "8502")),
        reload=False,
        show=True,
        reconnect_timeout=30.0,
        binding_refresh_interval=0.8,
    )
