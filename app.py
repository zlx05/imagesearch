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
CURRENT_JOB_ID = ""


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
    from agents.analyzer import parse_node, TimeSpaceAnalyzerAgent
    from core.state import AgentState, build_initial_state
    from core.visualization import write_g6_html
except Exception as import_error:  # 页面仍可启动，用于只展示已有报告
    report_node = upload_node = retrieve_node = validate_node = parse_node = TimeSpaceAnalyzerAgent = None  # type: ignore[assignment]
    AgentState = dict  # type: ignore[misc,assignment]
    build_initial_state = None  # type: ignore[assignment]
    write_g6_html = None  # type: ignore[assignment]
    IMPORT_错误 = import_error
else:
    IMPORT_错误 = None

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
    current_job_id: str = ""
    phase_data: dict[str, Any] | None = None


RUNTIME = AppRuntime(latest_log_lines=[], phase_data={})


def _json_default(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


def _compact_phase_for_status(phase: Any, *, list_limit: int = 320, event_limit: int = 32) -> dict[str, Any]:
    """Keep status polling light while preserving live UI progress and samples."""
    if not isinstance(phase, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in (
        "target_image",
        "retrieval_summary",
        "validation_summary",
        "analysis_summary",
        "topology_data",
        "overview",
        "validation_started_at",
        "analysis_started_at",
        "final_report",
    ):
        if key in phase:
            compact[key] = phase[key]
    for key in ("candidates", "nodes", "analyzed_nodes"):
        value = phase.get(key)
        if isinstance(value, list):
            compact[key] = value[:list_limit]
            if len(value) > list_limit:
                compact[f"{key}_truncated"] = len(value) - list_limit
        elif value is not None:
            compact[key] = value
    for key in ("retrieval_progress", "analysis_progress"):
        if isinstance(phase.get(key), dict):
            compact[key] = phase[key]
    validation_progress = phase.get("validation_progress")
    if isinstance(validation_progress, dict):
        progress = dict(validation_progress)
        events = progress.get("recent_events")
        if isinstance(events, list):
            compact_events = []
            for event in events[-event_limit:]:
                if not isinstance(event, dict):
                    continue
                node = event.get("node") if isinstance(event.get("node"), dict) else {}
                compact_events.append({
                    **{k: v for k, v in event.items() if k != "node"},
                    "node": _compact_node_for_ui(node, int(event.get("index") or 0)),
                    "reason": _short_text(event.get("reason"), 120),
                })
            progress["recent_events"] = compact_events
        compact["validation_progress"] = progress
    return compact


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
        "job_id": globals().get("CURRENT_JOB_ID", ""),
        "phase_data": _compact_phase_for_status(RUNTIME.phase_data or {}, list_limit=120, event_limit=12),
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

    def on_progress(step: str, current: int, total: int, sub_total: Any, message: str) -> None:
        now = time.time()
        full_logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # 降频写 status：只更新进度与一句状态，不更新前端关键日志。
        try:
            cur = float(current) if current is not None else 0.0
            tot = float(total) if total is not None else 0.0
        except (TypeError, ValueError):
            cur, tot = 0.0, 0.0
        progress_event = sub_total if isinstance(sub_total, dict) else {}
        if tot > 0:
            pct = min(max(cur / tot, 0.0), 1.0)
            phase = RUNTIME.phase_data or {}
            if "validate" in step:
                progress = 0.30 + pct * 0.34
                logical_step = "validate"
                previous = phase.get("validation_progress") if isinstance(phase.get("validation_progress"), dict) else {}
                recent_events = list(previous.get("recent_events") or [])
                keep_count = int(previous.get("keep_count") or 0)
                reject_count = int(previous.get("reject_count") or 0)
                if progress_event:
                    event_seq = int(previous.get("last_event_seq") or 0) + 1
                    event = {
                        **progress_event,
                        "seq": event_seq,
                        "current": int(cur),
                        "total": int(tot),
                        "updated_at": now,
                    }
                    recent_events.append(event)
                    recent_events = recent_events[-80:]
                    keep_count = int(event.get("keep_count") or keep_count)
                    reject_count = int(event.get("reject_count") or reject_count)
                else:
                    event_seq = int(previous.get("last_event_seq") or 0)
                phase["validation_progress"] = {
                    "current": int(cur),
                    "total": int(tot),
                    "updated_at": now,
                    "keep_count": keep_count,
                    "reject_count": reject_count,
                    "recent_events": recent_events,
                    "last_event_seq": event_seq,
                }
                RUNTIME.phase_data = phase
            elif "retrieve" in step:
                progress = 0.10 + pct * 0.18
                logical_step = "retrieve"
                phase["retrieval_progress"] = {"current": int(cur), "total": int(tot), "updated_at": now}
                RUNTIME.phase_data = phase
            elif "parse" in step or "analyze" in step:
                progress = 0.70 + pct * 0.16
                logical_step = "analyze"
                phase["analysis_progress"] = {"current": int(cur), "total": int(tot), "updated_at": now}
                RUNTIME.phase_data = phase
            else:
                progress = None
                logical_step = "validate"
        else:
            progress = None
            logical_step = "validate"
        emit_interval = 0.14 if "validate" in str(step) else 2.5
        if now - last_emit["t"] < emit_interval and current != total:
            return
        last_emit["t"] = now
        last_emit["count"] += 1
        write_status(
            status="running",
            step=logical_step,
            message=message[:120],
            progress=progress,
            extra={"recent_logs": (key_logs or [])},
        )

    return on_progress


def _merge_state_from_node(state: dict[str, Any], result: Any, fallback_state: dict[str, Any] | None = None) -> None:
    """兼容智能体节点返回 None 或原地修改 state 的情况，避免 dict.update(None) 直接炸掉。"""
    if isinstance(result, dict):
        state.update(result)
        return
    if fallback_state is not None and isinstance(fallback_state, dict):
        for key, value in fallback_state.items():
            if key != "_progress_callback":
                state[key] = value


def _read_image_meta(path: Path) -> dict[str, Any]:
    """尽量读取目标图片尺寸和格式；Pillow 不可用时不影响主流程。"""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return {"width": img.width, "height": img.height, "format": img.format or ""}
    except Exception:
        return {"width": "-", "height": "-", "format": ""}


def _short_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _compact_node_for_ui(node: Any, idx: int = 0) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {"id": str(idx), "title": _short_text(node), "platform": "未知"}
    url = str(node.get("url") or node.get("page_url") or node.get("source_url") or "")
    title = node.get("title") or node.get("description") or node.get("snippet") or url or f"候选资源 {idx + 1}"
    return {
        "id": str(node.get("id") or node.get("node_id") or node.get("url") or node.get("page_url") or idx),
        "idx": idx + 1,
        "title": _short_text(title, 260),
        "url": url,
        "page_url": str(node.get("page_url") or url),
        "image_url": str(node.get("image_url") or node.get("thumbnail_url") or ""),
        "thumbnail_url": str(node.get("thumbnail_url") or node.get("image_url") or ""),
        "platform": str(node.get("platform") or node.get("engine") or node.get("source") or node.get("platform_family") or "未知"),
        "engine": str(node.get("engine") or ""),
        "source": str(node.get("source") or ""),
        "publisher": _short_text(node.get("publisher") or node.get("author") or node.get("metadata_author") or node.get("domain") or "未知", 80),
        "author": _short_text(node.get("author") or node.get("publisher") or node.get("metadata_author") or "未知", 80),
        "published_at": str(node.get("published_at") or node.get("publish_time") or ""),
        "propagation_role": str(node.get("propagation_role") or node.get("page_type") or "候选节点"),
        "similarity": node.get("similarity"),
        "clip_similarity": node.get("clip_similarity") or node.get("clip_score"),
        "phash_distance": node.get("phash_distance") or node.get("hash_distance"),
        "ocr_overlap": node.get("ocr_overlap") or node.get("ocr_text_overlap"),
        "like_count": node.get("like_count"),
        "comment_count": node.get("comment_count"),
        "repost_count": node.get("repost_count"),
        "view_count": node.get("view_count"),
        "is_suspected_source": bool(node.get("is_suspected_source")),
        "is_key_node": bool(node.get("is_key_node")),
        "suspected_tampering": bool(node.get("suspected_tampering")),
    }


def _compact_nodes_for_ui(nodes: Any, limit: int = 2000) -> list[dict[str, Any]]:
    if not isinstance(nodes, list):
        return []
    return [_compact_node_for_ui(n, i) for i, n in enumerate(nodes[:limit])]


def _refresh_phase_overview() -> None:
    phase = RUNTIME.phase_data or {}
    nodes = phase.get("nodes") or phase.get("analyzed_nodes") or []
    summary = {
        "retrieval_summary": phase.get("retrieval_summary") or {},
        "validation_summary": phase.get("validation_summary") or {},
        "analysis_summary": phase.get("analysis_summary") or {},
        "topology_data": phase.get("topology_data") or {},
    }
    try:
        phase["overview"] = summarize_nodes(nodes if isinstance(nodes, list) else [], summary)
    except Exception:
        phase["overview"] = {}
    RUNTIME.phase_data = phase


async def _run_analyzer_progressive(state: dict[str, Any], key_logs: list[str], full_logs: list[str]) -> dict[str, Any]:
    """Run Analyzer with per-node UI updates while preserving Analyzer's core logic.

    parse_node itself only returns after all nodes are done. This wrapper uses the
    same TimeSpaceAnalyzerAgent methods, but publishes each analyzed node into
    RUNTIME.phase_data as soon as it is completed so the card-flip UI can update
    progressively.
    """
    if TimeSpaceAnalyzerAgent is None:
        return await run.io_bound(parse_node, state)  # type: ignore[misc]

    agent = TimeSpaceAnalyzerAgent()  # type: ignore[operator]
    logs = list(state.get("execution_logs") or [])
    started_at = time.perf_counter()
    input_nodes = agent.preprocess_input_nodes(state.get("nodes_data") or state.get("nodes") or [])
    total = len(input_nodes)
    phase = RUNTIME.phase_data or {}
    phase["analysis_input_count"] = total
    phase["analyzed_nodes"] = []
    phase["analysis_progress"] = {"current": 0, "total": total, "updated_at": time.time()}
    RUNTIME.phase_data = phase
    _refresh_phase_overview()
    write_status(status="running", step="analyze", message=f"正在逐条提取网页信息：0/{total}", progress=0.70, extra={"recent_logs": key_logs})

    analyzed_nodes: list[dict[str, Any]] = []
    for index, node in enumerate(input_nodes):
        analyzed = await run.io_bound(agent._analyze_node_safely, node, index)
        if analyzed is None or not isinstance(analyzed, dict):
            analyzed = dict(node) if isinstance(node, dict) else {"id": str(index)}
            analyzed["crawl_status"] = "failed"
            analyzed["crawl_source"] = "analyzer_error"
        analyzed_nodes.append(analyzed)
        done = index + 1
        log_line = f"analyzer: extracted node {done}/{total}, id={analyzed.get('id')}"
        full_logs.append(f"[{time.strftime('%H:%M:%S')}] {log_line}")
        phase = RUNTIME.phase_data or {}
        phase["analyzed_nodes"] = _compact_nodes_for_ui(analyzed_nodes, limit=3000)
        phase["analysis_progress"] = {"current": done, "total": total, "updated_at": time.time()}
        RUNTIME.phase_data = phase
        _refresh_phase_overview()
        # 写 status 会把 phase_data 一起发给前端，卡片可逐张翻面。
        write_status(
            status="running",
            step="analyze",
            message=f"正在逐条提取网页信息：{done}/{total}",
            progress=0.70 + (done / max(total, 1)) * 0.14,
            extra={"recent_logs": key_logs},
        )

    analyzed_nodes.sort(key=agent._sort_key)
    agent._assign_topology(analyzed_nodes)
    agent._assign_matrix_candidates(analyzed_nodes)
    agent._assign_duplicate_clusters(analyzed_nodes)
    topology_data = agent.build_topology_data(analyzed_nodes)
    analysis_summary = agent.build_analysis_summary(analyzed_nodes, topology_data)
    mermaid_graph = agent.build_mermaid_graph(analyzed_nodes)
    elapsed_seconds = round(time.perf_counter() - started_at, 2)
    topology_data["runtime"] = {
        "elapsed_seconds": elapsed_seconds,
        "elapsed_human": agent._format_elapsed(elapsed_seconds),
        "node_count": len(analyzed_nodes),
        "edge_count": len(topology_data.get("edges") or []),
    }
    return {
        "nodes_data": analyzed_nodes,
        "mermaid_graph": mermaid_graph,
        "topology_data": topology_data,
        "analysis_summary": analysis_summary,
        "execution_logs": logs,
    }


async def execute_workflow(content: bytes, filename: str) -> str | None:
    """真实运行工作流。

    日志分两套：
    - key_logs：前端展示用关键里程碑日志，写入 logs.txt。
    - full_logs：后台排错用完整细粒度日志，写入 full_logs.txt。
    """
    if IMPORT_错误 is not None or build_initial_state is None:
        write_status(status="error", step="idle", message=f"业务模块导入失败：{IMPORT_错误}", progress=0.0)
        return None

    key_logs: list[str] = []
    full_logs: list[str] = []
    report_dir: Path | None = None
    current_stage = "upload"

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
        image_meta = await run.io_bound(_read_image_meta, saved_path)
        key_log(f"上传图片：保存完成，文件名={filename}，大小={len(content) / 1024:.1f}KB")
        RUNTIME.phase_data = {
            "target_image": {
                "filename": filename,
                "size_bytes": len(content),
                "width": image_meta.get("width"),
                "height": image_meta.get("height"),
                "format": image_meta.get("format"),
                "local_path": str(saved_path),
            },
            "candidates": [],
            "nodes": [],
        }
        _refresh_phase_overview()

        state: AgentState = build_initial_state({
            "filename": filename,
            "content_type": "",
            "size_bytes": len(content),
            "local_path": str(saved_path),
            "width": image_meta.get("width"),
            "height": image_meta.get("height"),
            "format": image_meta.get("format"),
        })
        if isinstance(state, dict) and isinstance(state.get("target_image"), dict):
            state["target_image"].update({
                "width": image_meta.get("width"),
                "height": image_meta.get("height"),
                "format": image_meta.get("format"),
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
        _merge_state_from_node(state, r)
        if isinstance(state.get("target_image"), dict):
            for _k in ("width", "height", "format"):
                if state["target_image"].get(_k) in (None, "", "-"):
                    state["target_image"][_k] = image_meta.get(_k)
            if not state["target_image"].get("local_path"):
                state["target_image"]["local_path"] = str(saved_path)
        target_image = state.get("target_image", {}) if isinstance(state, dict) else {}
        if isinstance(target_image, dict):
            width = target_image.get("width") or target_image.get("image_width") or "-"
            height = target_image.get("height") or target_image.get("image_height") or "-"
            local_path = target_image.get("local_path") or str(saved_path)
            key_log(f"图片信息提取：完成，尺寸={width}×{height}，路径={local_path}")
            phase = RUNTIME.phase_data or {}
            phase["target_image"] = {**phase.get("target_image", {}), **target_image}
            RUNTIME.phase_data = phase
            _refresh_phase_overview()
            write_status(status="running", step="upload", message="图片信息提取完成，准备进入多平台检索。", progress=0.10, extra={"recent_logs": key_logs})
        else:
            key_log("图片信息提取：完成")

        # 3. 以图搜图
        key_log(f"以图搜图：开始调用搜索引擎（{', '.join(env_engines)}）")
        write_status(status="running", step="retrieve", message="正在以图搜图，收集候选链接...", progress=0.12, extra={"recent_logs": key_logs})
        current_stage = "retrieve"
        r = await run.io_bound(retrieve_node, state, progress_callback=_progress_recorder(full_logs, key_logs))  # type: ignore[misc]
        _merge_state_from_node(state, r)
        rsummary = state.get("retrieval_summary", {})
        per_engine = rsummary.get("per_engine_counts", {}) if isinstance(rsummary, dict) else {}
        engine_text = ", ".join(f"{k}:{v}" for k, v in per_engine.items()) if isinstance(per_engine, dict) and per_engine else "-"
        retrieved_nodes_snapshot = state.get("nodes_data", [])
        if isinstance(retrieved_nodes_snapshot, list):
            state["retrieved_nodes"] = [dict(n) if isinstance(n, dict) else n for n in retrieved_nodes_snapshot]
        else:
            state["retrieved_nodes"] = []
        candidate_count = rsummary.get("result_count", len(state.get("retrieved_nodes", []))) if isinstance(rsummary, dict) else len(state.get("retrieved_nodes", []))
        key_log(f"以图搜图：完成，候选结果={candidate_count}条，引擎返回={engine_text}")
        phase = RUNTIME.phase_data or {}
        phase["retrieval_summary"] = rsummary if isinstance(rsummary, dict) else {}
        phase["candidates"] = _compact_nodes_for_ui(state.get("retrieved_nodes", []), limit=3000)
        RUNTIME.phase_data = phase
        _refresh_phase_overview()
        write_status(status="running", step="retrieve", message=f"多平台检索完成：候选资源 {candidate_count} 条。", progress=0.30, extra={"recent_logs": key_logs})

        # 4. 相似度校验与去重
        key_log("相似度校验/去重：开始处理候选结果")
        phase = RUNTIME.phase_data or {}
        phase["validation_started_at"] = time.time()
        RUNTIME.phase_data = phase
        write_status(status="running", step="validate", message="正在进行相似度校验和去重...", progress=0.32, extra={"recent_logs": key_logs})
        vstate = dict(state)
        vstate["_progress_callback"] = _progress_recorder(full_logs, key_logs)
        current_stage = "validate"
        r = await run.io_bound(validate_node, vstate)  # type: ignore[misc]
        _merge_state_from_node(state, r, vstate)
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
        phase = RUNTIME.phase_data or {}
        phase["validation_summary"] = vsummary if isinstance(vsummary, dict) else {}
        phase["nodes"] = _compact_nodes_for_ui(state.get("nodes_data", []), limit=3000)
        RUNTIME.phase_data = phase
        _refresh_phase_overview()
        write_status(status="running", step="validate", message="相似度校验/去重完成，准备提取网页信息。", progress=0.66, extra={"recent_logs": key_logs})

        # 5. 内容提取与传播分析
        key_log("内容提取/传播分析：开始提取发布时间、账号、互动量与传播关系")
        phase = RUNTIME.phase_data or {}
        phase["analysis_started_at"] = time.time()
        RUNTIME.phase_data = phase
        write_status(status="running", step="analyze", message="正在提取内容并分析传播关系...", progress=0.70, extra={"recent_logs": key_logs})
        current_stage = "analyze"
        if os.getenv("APP_PROGRESSIVE_ANALYZER", "true").strip().lower() in {"1", "true", "yes", "on"}:
            r = await _run_analyzer_progressive(state, key_logs, full_logs)
        else:
            r = await run.io_bound(parse_node, state)  # type: ignore[misc]
        _merge_state_from_node(state, r)
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
        phase = RUNTIME.phase_data or {}
        phase["analysis_summary"] = analysis_summary if isinstance(analysis_summary, dict) else {}
        phase["nodes"] = _compact_nodes_for_ui(state.get("nodes_data", []), limit=3000)
        phase["topology_data"] = state.get("topology_data", {}) if isinstance(state.get("topology_data", {}), dict) else {}
        RUNTIME.phase_data = phase
        _refresh_phase_overview()
        write_status(status="running", step="analyze", message="网页信息提取与传播分析完成，准备生成报告。", progress=0.86, extra={"recent_logs": key_logs})

        # 6. 生成报告
        key_log("生成报告：开始生成文字报告与拓扑图")
        write_status(status="running", step="report", message="正在生成报告和拓扑图...", progress=0.94, extra={"recent_logs": key_logs})
        current_stage = "report"
        r = await run.io_bound(report_node, state)  # type: ignore[misc]
        _merge_state_from_node(state, r)
        key_log("生成报告：文字报告生成完成")
        phase = RUNTIME.phase_data or {}
        phase["final_report"] = state.get("final_report", "")
        phase["topology_data"] = state.get("topology_data", {}) if isinstance(state.get("topology_data", {}), dict) else {}
        RUNTIME.phase_data = phase
        _refresh_phase_overview()
        write_status(status="running", step="report", message="文字报告已生成，正在写出最终报告文件。", progress=0.97, extra={"recent_logs": key_logs})

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
        retrieved_nodes_data = state.get("retrieved_nodes", [])
        if not isinstance(retrieved_nodes_data, list):
            retrieved_nodes_data = []
        summary_payload = {
            "report": state.get("final_report", ""),
            "insight_report": state.get("insight_report", ""),
            "confidence_scores": state.get("confidence_scores", {}),
            "analysis_summary": state.get("analysis_summary", {}),
            "validation_summary": state.get("validation_summary", {}),
            "retrieval_summary": state.get("retrieval_summary", {}),
            "topology_data": state.get("topology_data", {}),
            "key_logs": key_logs,
        }
        (report_dir / "nodes_data.json").write_text(json.dumps(nodes_data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        (report_dir / "retrieved_nodes.json").write_text(json.dumps(retrieved_nodes_data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
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
        print(f"[错误] 工作流异常:\n{tb_text}")
        failed_dir = OUTPUT_DIR / f"partial_failed_{time.strftime('%Y%m%d_%H%M%S')}"
        try:
            failed_dir.mkdir(parents=True, exist_ok=True)
            (failed_dir / "logs.txt").write_text("\n".join(key_logs), encoding="utf-8")
            (failed_dir / "full_logs.txt").write_text("\n".join(full_logs), encoding="utf-8")
        except Exception:
            pass
        write_status(status="error", step=current_stage or "idle", message=f"流程失败：{e}", progress=STEP_PROGRESS.get(current_stage, 0.0), extra={"recent_logs": key_logs})
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
    
    /* v7: retrieval engine dashboard and smoother live-stage animation */
    .engine-board{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:4px}
    .engine-card{min-height:230px;border:1px solid rgba(148,163,184,.25);background:linear-gradient(180deg,rgba(15,23,42,.72),rgba(2,6,23,.56));border-radius:20px;position:relative;overflow:hidden;padding:16px;display:grid;grid-template-rows:auto 1fr auto;box-shadow:0 16px 42px rgba(0,0,0,.22);opacity:0;transform:translateY(18px) scale(.985)}
    .engine-card.show{animation:popIn .55s cubic-bezier(.2,.8,.2,1) forwards}
    .engine-card::before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(56,189,248,.18),transparent);transform:translateX(-110%);animation:engineSweep 1.65s ease-in-out infinite}
    .engine-card::after{content:"";position:absolute;left:0;right:0;top:42%;height:2px;background:linear-gradient(90deg,transparent,rgba(56,189,248,.72),transparent);box-shadow:0 0 20px rgba(56,189,248,.35);animation:engineScan 1.35s ease-in-out infinite}
    @keyframes engineSweep{0%,25%{transform:translateX(-110%)}100%{transform:translateX(110%)}}
    @keyframes engineScan{0%{top:16%;opacity:.2}45%{opacity:.95}100%{top:82%;opacity:.08}}
    .engine-name{position:relative;z-index:1;display:flex;justify-content:space-between;align-items:center;gap:10px;color:#e2e8f0;font-weight:800;font-size:17px}
    .engine-name span{font-size:12px;color:var(--muted);font-weight:650}
    .engine-preview{position:relative;z-index:1;border:1px solid rgba(148,163,184,.18);border-radius:16px;background:rgba(15,23,42,.54);display:grid;place-items:center;min-height:130px;margin-top:14px;color:rgba(226,232,240,.72);font-size:42px;overflow:hidden}
    .engine-preview img{width:100%;height:100%;object-fit:cover;filter:saturate(.92) contrast(.95)}
    .engine-stats{position:relative;z-index:1;display:flex;justify-content:space-between;align-items:center;margin-top:12px;color:var(--muted);font-size:12px}
    .engine-stats strong{font-size:22px;color:#dbeafe}
    .candidate-section-title{grid-column:1/-1;display:flex;justify-content:space-between;align-items:center;border:1px solid var(--line);background:rgba(2,6,23,.30);border-radius:14px;padding:10px 14px;color:var(--muted);font-size:13px;margin-top:2px}
    .candidate-section-title strong{color:#e2e8f0}
    .validation-visual{border:1px solid rgba(56,189,248,.22);background:rgba(2,6,23,.48);border-radius:16px;padding:12px;min-height:142px;display:grid;align-content:center;gap:10px;color:#cbd5e1;font-size:13px}
    .validation-visual-bar{height:9px;border-radius:999px;background:rgba(148,163,184,.16);overflow:hidden}
    .validation-visual-bar i{display:block;height:100%;width:38%;border-radius:999px;background:linear-gradient(90deg,rgba(37,99,235,.9),rgba(56,189,248,.95));animation:barTravel 1.6s ease-in-out infinite}
    @keyframes barTravel{0%{transform:translateX(-110%)}100%{transform:translateX(280%)}}
    .extract-status{grid-column:1/-1;border:1px solid rgba(56,189,248,.22);background:rgba(2,6,23,.32);border-radius:16px;padding:12px 14px;color:#cbd5e1;font-size:13px;display:flex;justify-content:space-between;gap:16px;align-items:center}
    .extract-progress{height:8px;min-width:240px;flex:0 0 260px;border-radius:999px;background:rgba(148,163,184,.16);overflow:hidden}
    .extract-progress i{display:block;height:100%;width:0%;background:linear-gradient(90deg,rgba(16,185,129,.85),rgba(56,189,248,.85));transition:width .35s ease}

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
    ui.markdown("### 传播拓扑").classes("m-0")
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




# =============================================================================
# 自动翻页版：使用 demo 深蓝前端模板 + 原后端真实工作流
# =============================================================================

from fastapi import UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

UI_STEP_BY_BACKEND_STEP = {
    "idle": 0,
    "upload": 0,
    "retrieve": 1,
    "validate": 2,
    "analyze": 3,
    "report": 4,
}


def _safe_report_payload(report_dir_name: str) -> dict[str, Any]:
    safe_name = Path(report_dir_name).name
    report_dir = (OUTPUT_DIR / safe_name).resolve()
    output_root = OUTPUT_DIR.resolve()
    if output_root not in report_dir.parents and report_dir != output_root:
        raise HTTPException(status_code=400, detail="invalid report directory")
    if not report_dir.exists() or not report_dir.is_dir():
        raise HTTPException(status_code=404, detail="report not found")

    summary = read_json(report_dir / "summary.json", {})
    nodes = read_json(report_dir / "nodes_data.json", [])
    state_dump = read_json(report_dir / "state_dump.json", {})
    retrieved_nodes = read_json(report_dir / "retrieved_nodes.json", [])
    if not isinstance(retrieved_nodes, list):
        retrieved_nodes = []
    if not retrieved_nodes and isinstance(state_dump, dict):
        retrieved_nodes = state_dump.get("retrieved_nodes") or state_dump.get("raw_retrieved_nodes") or []
    if not isinstance(retrieved_nodes, list):
        retrieved_nodes = []
    logs_text = ""
    full_logs_text = ""
    try:
        logs_text = (report_dir / "logs.txt").read_text(encoding="utf-8")
    except Exception:
        logs_text = ""
    try:
        full_logs_text = (report_dir / "full_logs.txt").read_text(encoding="utf-8")
    except Exception:
        full_logs_text = logs_text

    overview = summarize_nodes(nodes if isinstance(nodes, list) else [], summary if isinstance(summary, dict) else {})
    return {
        "report_dir": safe_name,
        "summary": summary if isinstance(summary, dict) else {},
        "nodes": nodes if isinstance(nodes, list) else [],
        "candidates": retrieved_nodes if isinstance(retrieved_nodes, list) else [],
        "state_dump": state_dump if isinstance(state_dump, dict) else {},
        "overview": overview,
        "logs": logs_text,
        "full_logs": full_logs_text,
        "downloads": {
            "nodes": f"/report_files/{safe_name}/nodes_data.json",
            "candidates": f"/report_files/{safe_name}/retrieved_nodes.json",
            "summary": f"/report_files/{safe_name}/summary.json",
            "logs": f"/report_files/{safe_name}/logs.txt",
            "full_logs": f"/report_files/{safe_name}/full_logs.txt",
            "topology": f"/report_files/{safe_name}/topology.html",
        },
    }


async def _run_uploaded_workflow(data: bytes, filename: str) -> None:
    try:
        await execute_workflow(data, filename)
    finally:
        RUNTIME.running = False


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> JSONResponse:
    if RUNTIME.running:
        return JSONResponse({"ok": False, "error": "workflow already running"}, status_code=409)
    data = await file.read()
    if not data:
        return JSONResponse({"ok": False, "error": "empty file"}, status_code=400)
    filename = file.filename or "uploaded.jpg"
    job_id = uuid4().hex
    globals()["CURRENT_JOB_ID"] = job_id
    RUNTIME.current_job_id = job_id
    RUNTIME.uploaded_content = data
    RUNTIME.uploaded_filename = filename
    RUNTIME.active_report_dir = None
    RUNTIME.latest_log_lines = []
    RUNTIME.phase_data = {"target_image": {"filename": filename, "size_bytes": len(data)}, "candidates": [], "nodes": []}
    RUNTIME.running = True
    write_status(status="running", step="upload", message="已接收上传图片，正在启动真实溯源流程。", progress=0.01, report_dir=None)
    asyncio.create_task(_run_uploaded_workflow(data, filename))
    return JSONResponse({"ok": True, "filename": filename, "size_bytes": len(data), "job_id": job_id})


@app.get("/api/status")
async def api_status() -> JSONResponse:
    status = read_status()
    # 不把上一次遗留的 status.json 推给新打开的页面，避免一进页面就跳到步骤 3/4。
    if not RUNTIME.running and not RUNTIME.current_job_id:
        status = {"status": "idle", "step": "idle", "message": "等待上传图片", "progress": 0.0, "report_dir": None, "job_id": ""}
    elif status.get("job_id") and RUNTIME.current_job_id and status.get("job_id") != RUNTIME.current_job_id:
        status = {"status": "idle", "step": "idle", "message": "等待上传图片", "progress": 0.0, "report_dir": None, "job_id": RUNTIME.current_job_id}
    status["running"] = RUNTIME.running
    status["phase_data"] = RUNTIME.phase_data or status.get("phase_data") or {}
    status["uploaded_filename"] = RUNTIME.uploaded_filename
    status["current_job_id"] = RUNTIME.current_job_id
    status["ui_step"] = UI_STEP_BY_BACKEND_STEP.get(str(status.get("step") or "idle"), 0)
    return JSONResponse(status)


@app.get("/api/report/{report_dir_name}")
async def api_report(report_dir_name: str) -> JSONResponse:
    return JSONResponse(_safe_report_payload(report_dir_name))


@app.get("/api/latest")
async def api_latest() -> JSONResponse:
    report = latest_report_dir()
    if not report:
        return JSONResponse({"ok": False, "error": "no report found"}, status_code=404)
    return JSONResponse({"ok": True, "report_dir": report.name, **_safe_report_payload(report.name)})


@app.get("/api/import-test")
async def api_import_test() -> JSONResponse:
    name = copy_uploaded_test_files_to_output()
    if not name:
        report = latest_report_dir()
        if not report:
            return JSONResponse({"ok": False, "error": "no test report found"}, status_code=404)
        name = report.name
    RUNTIME.active_report_dir = name
    globals()["CURRENT_JOB_ID"] = "manual_load"
    RUNTIME.current_job_id = "manual_load"
    write_status(status="done", step="report", message=f"已载入测试/最近报告：{name}", progress=1.0, report_dir=name)
    return JSONResponse({"ok": True, "report_dir": name, **_safe_report_payload(name)})


INDEX_HTML = r'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>图片溯源智能体 · </title>
  <link rel="preconnect" href="https://cdnjs.cloudflare.com" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css" />
  <script src="https://unpkg.com/vis-network@9.1.6/dist/vis-network.min.js"></script>
  <style>
    :root{--bg:#0f172a;--bg2:#111827;--panel:rgba(15,23,42,.78);--panel-strong:rgba(17,24,39,.94);--card:rgba(255,255,255,.075);--card2:rgba(255,255,255,.105);--line:rgba(148,163,184,.22);--line2:rgba(148,163,184,.38);--text:#e5e7eb;--muted:#94a3b8;--subtle:#64748b;--blue:#38bdf8;--blue2:#2563eb;--green:#10b981;--red:#b91c1c;--amber:#d97706;--shadow:0 30px 80px rgba(0,0,0,.34);--radius:22px;--fast:260ms;--normal:520ms;--slow:900ms;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
    *{box-sizing:border-box}html,body{margin:0;height:100%;overflow:hidden;background:var(--bg);color:var(--text)}
    body::before{content:"";position:fixed;inset:0;background:radial-gradient(circle at 18% 12%,rgba(37,99,235,.18),transparent 34%),radial-gradient(circle at 80% 18%,rgba(14,165,233,.12),transparent 30%),linear-gradient(135deg,#0f172a 0%,#111827 50%,#020617 100%);z-index:-3}
    body::after{content:"";position:fixed;inset:0;background-image:linear-gradient(rgba(148,163,184,.055) 1px,transparent 1px),linear-gradient(90deg,rgba(148,163,184,.055) 1px,transparent 1px);background-size:34px 34px;mask-image:linear-gradient(to bottom,rgba(0,0,0,.85),rgba(0,0,0,.25));z-index:-2}
    .app-shell{height:100vh;padding:24px 30px 26px;display:grid;grid-template-rows:auto 1fr;gap:16px;max-width:1500px;margin:0 auto;width:100%}.topbar{display:grid;grid-template-columns:1fr auto auto;gap:16px;align-items:center;padding:13px 15px;border:1px solid var(--line);background:rgba(2,6,23,.46);backdrop-filter:blur(18px);border-radius:20px;box-shadow:0 18px 70px rgba(0,0,0,.18)}
    .brand{display:flex;align-items:center;gap:13px;min-width:0}.brand-mark{width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,rgba(56,189,248,.18),rgba(37,99,235,.16));border:1px solid rgba(56,189,248,.28);display:grid;place-items:center;color:var(--blue);box-shadow:inset 0 0 30px rgba(56,189,248,.06)}.brand h1{margin:0;font-size:18px;letter-spacing:.04em;font-weight:650}.brand p{margin:4px 0 0;color:var(--muted);font-size:12px;letter-spacing:.04em}.progress-meta{text-align:right;color:var(--muted);font-size:13px;white-space:nowrap}.step-pill{color:#dbeafe;border:1px solid rgba(56,189,248,.26);background:rgba(37,99,235,.13);border-radius:999px;padding:9px 14px;font-size:13px;font-weight:650;white-space:nowrap;margin-bottom:6px}.status-chip{max-width:520px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-size:12px}.nav-actions{display:flex;gap:9px;align-items:center;flex-wrap:wrap;justify-content:flex-end}.btn,.linkbtn{border:1px solid rgba(148,163,184,.26);background:rgba(255,255,255,.07);color:var(--text);border-radius:13px;padding:10px 13px;cursor:pointer;font-weight:650;letter-spacing:.02em;transition:transform var(--fast),border-color var(--fast),background var(--fast),opacity var(--fast);text-decoration:none;font-size:13px}.btn:hover,.linkbtn:hover{transform:translateY(-1px);border-color:rgba(56,189,248,.48);background:rgba(255,255,255,.105)}.btn:disabled{opacity:.45;cursor:not-allowed;transform:none}.btn.primary{background:linear-gradient(135deg,rgba(37,99,235,.82),rgba(14,116,144,.82));border-color:rgba(56,189,248,.38)}.btn.ghost{background:rgba(2,6,23,.35)}.btn.on{border-color:rgba(16,185,129,.52);background:rgba(16,185,129,.10);color:#bbf7d0}
    .stage-wrap{position:relative;min-height:0;perspective:1700px;overflow:hidden;border-radius:28px}.steps-track{height:100%;display:flex;transition:transform 720ms cubic-bezier(.18,.78,.25,1);will-change:transform}.step{min-width:100%;height:100%;padding:24px;overflow:hidden;opacity:.26;transform:scale(.985) rotateY(-4deg);transition:opacity 520ms ease,transform 720ms cubic-bezier(.18,.78,.25,1)}.step.active{opacity:1;transform:scale(1) rotateY(0);visibility:visible}.step:not(.active){visibility:hidden;pointer-events:none;opacity:0}.steps-track{width:100%}.step{flex:0 0 100%;max-width:100%;min-width:100%}.step-card{height:100%;border:1px solid var(--line);border-radius:28px;background:linear-gradient(180deg,rgba(15,23,42,.82),rgba(2,6,23,.68));box-shadow:var(--shadow);backdrop-filter:blur(22px);overflow:hidden;display:grid;grid-template-rows:auto 1fr;position:relative}.step-card::before{content:"";position:absolute;inset:0;pointer-events:none;background:linear-gradient(135deg,rgba(255,255,255,.08),transparent 30%,transparent 74%,rgba(56,189,248,.06));opacity:.66}.step-head{position:relative;z-index:1;padding:23px 27px 17px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:24px;align-items:flex-end}.kicker{color:var(--blue);font-size:12px;letter-spacing:.16em;text-transform:uppercase;font-weight:750;margin-bottom:9px}.step-title{margin:0;font-size:26px;font-weight:680;letter-spacing:.015em}.step-desc{margin:8px 0 0;color:var(--muted);font-size:14px;line-height:1.6;max-width:780px}.head-stat{display:flex;gap:10px;align-items:center;color:var(--muted);font-size:13px;white-space:nowrap}.head-stat strong{color:#e2e8f0;font-size:22px}.step-body{position:relative;z-index:1;min-height:0;padding:24px 27px 27px;overflow:hidden}.muted{color:var(--muted)}.status-note{color:var(--muted);font-size:13px;line-height:1.65}.hidden{display:none!important}
    .insight-card{border:1px solid var(--line);background:rgba(2,6,23,.22);border-radius:12px;padding:14px 16px}.insight-head{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:.9rem;color:#e2e8f0}.insight-head i{width:18px;text-align:center}.insight-body{color:#cbd5e1;font-size:.78rem;line-height:1.6;white-space:pre-wrap}.score-grid{display:grid;gap:8px;margin-top:8px}.score-item{margin-bottom:2px}.score-label{display:flex;justify-content:space-between;font-size:.74rem;margin-bottom:3px}.score-label span{color:#cbd5e1}.score-track{height:5px;border-radius:999px;background:rgba(148,163,184,.12);overflow:hidden;margin-bottom:2px}.score-track i{display:block;height:100%;border-radius:999px;transition:width .6s ease}.score-reason{font-size:.68rem;color:var(--muted);line-height:1.35}.report-metric-bar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}.rchip{border:1px solid var(--line);border-radius:999px;padding:4px 12px;background:rgba(255,255,255,.04);font-size:.72rem;color:#cbd5e1}.rchip b{color:var(--muted);font-weight:550;margin-right:4px}.upload-layout{height:100%;display:grid;grid-template-columns:1.02fr .98fr;gap:26px;align-items:center}.dropzone{height:min(450px,68vh);border:1.5px dashed rgba(148,163,184,.48);border-radius:26px;background:rgba(2,6,23,.26);display:grid;place-items:center;cursor:pointer;transition:border-color var(--fast),background var(--fast),transform var(--fast);position:relative;overflow:hidden}.dropzone:hover,.dropzone.dragover{border-color:rgba(56,189,248,.75);background:rgba(14,116,144,.11);transform:translateY(-2px)}.dropzone::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(56,189,248,.08),transparent);transform:translateX(-120%);animation:sheen 4.8s infinite}@keyframes sheen{0%,58%{transform:translateX(-120%)}100%{transform:translateX(120%)}}.upload-inner{text-align:center;padding:24px}.upload-icon{width:84px;height:84px;display:grid;place-items:center;margin:0 auto 22px;border-radius:24px;background:rgba(56,189,248,.11);color:var(--blue);font-size:34px;border:1px solid rgba(56,189,248,.24)}.upload-inner h3{margin:0;font-size:23px;font-weight:650}.upload-inner p{color:var(--muted);line-height:1.7;margin:12px auto 0;max-width:450px}.preview-panel{display:grid;gap:14px;align-content:center}.image-preview{min-height:250px;border-radius:24px;border:1px solid var(--line);background:rgba(255,255,255,.055);display:grid;place-items:center;overflow:hidden;color:var(--subtle)}.image-preview img{max-width:100%;max-height:320px;object-fit:contain;animation:fadeScale 620ms cubic-bezier(.2,.9,.2,1)}@keyframes fadeScale{from{opacity:0;transform:scale(.94)}to{opacity:1;transform:scale(1)}}.info-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.info-card{border:1px solid var(--line);background:rgba(255,255,255,.06);border-radius:17px;padding:15px;transform:translateY(18px);opacity:0}.info-card.reveal{animation:slideUp .55s cubic-bezier(.2,.85,.22,1.18) forwards}@keyframes slideUp{to{transform:translateY(0);opacity:1}}.label{color:var(--muted);font-size:12px;margin-bottom:7px;display:flex;gap:7px;align-items:center}.value{font-weight:650;color:#f1f5f9;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .scan-overlay{position:absolute;inset:0;background:linear-gradient(180deg,transparent,rgba(56,189,248,.16),transparent);transform:translateY(-100%);opacity:0;pointer-events:none;z-index:5}.scan-overlay.run{animation:scanLine .8s ease-out}@keyframes scanLine{0%{opacity:1;transform:translateY(-100%)}100%{opacity:.05;transform:translateY(100%)}}.result-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:16px;align-content:start;height:100%;overflow-y:auto;overflow-x:hidden;padding:0 6px 24px 0;scroll-behavior:smooth}.result-card{border:1px solid var(--line);background:rgba(255,255,255,.064);border-radius:18px;overflow:hidden;min-height:194px;opacity:0;transform:translateY(16px) scale(.985);transition:transform var(--fast),box-shadow var(--fast),border-color var(--fast)}.result-card.show{animation:popIn .45s cubic-bezier(.2,.8,.2,1) forwards}.result-card:hover{transform:translateY(-4px);box-shadow:0 18px 44px rgba(0,0,0,.28);border-color:rgba(56,189,248,.38)}@keyframes popIn{to{opacity:1;transform:translateY(0) scale(1)}}.thumb{height:92px;background:linear-gradient(135deg,rgba(30,41,59,.92),rgba(51,65,85,.64));display:grid;place-items:center;color:rgba(226,232,240,.7);font-size:26px;position:relative;overflow:hidden}.thumb img{width:100%;height:100%;object-fit:cover}.thumb::after{content:"";position:absolute;inset:0;background:linear-gradient(135deg,transparent,rgba(56,189,248,.08))}.result-content{padding:14px}.platform{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:750;border-radius:999px;padding:5px 8px;margin-bottom:10px;letter-spacing:.03em}.p-baidu{background:rgba(30,64,175,.18);color:#bfdbfe;border:1px solid rgba(59,130,246,.26)}.p-yandex{background:rgba(146,64,14,.18);color:#fed7aa;border:1px solid rgba(217,119,6,.24)}.p-google{background:rgba(21,128,61,.18);color:#bbf7d0;border:1px solid rgba(34,197,94,.22)}.p-xhs{background:rgba(127,29,29,.18);color:#fecaca;border:1px solid rgba(185,28,28,.24)}.p-weibo{background:rgba(51,65,85,.42);color:#cbd5e1;border:1px solid rgba(148,163,184,.22)}.p-other{background:rgba(15,23,42,.58);color:#e2e8f0;border:1px solid rgba(148,163,184,.22)}.result-title{font-weight:650;font-size:14px;line-height:1.45;margin-bottom:8px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}.result-meta{color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:10px;min-width:0}.result-meta span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.result-note{grid-column:1/-1;border:1px solid var(--line);background:rgba(2,6,23,.30);border-radius:14px;padding:10px 14px;color:var(--muted);font-size:13px}.skeleton-card{position:relative;overflow:hidden}.skeleton-card::before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(226,232,240,.08),transparent);transform:translateX(-100%);animation:loadingBeam 1.2s infinite}@keyframes loadingBeam{to{transform:translateX(100%)}}
    .validate-layout{height:100%;display:grid;grid-template-columns:1fr 1.18fr 1fr;gap:18px;min-height:0}.zone{border:1px solid var(--line);background:rgba(2,6,23,.24);border-radius:22px;padding:18px;min-height:0;overflow:hidden;position:relative}.zone-title{display:flex;justify-content:space-between;align-items:center;color:#e2e8f0;font-size:15px;font-weight:700;margin-bottom:14px}.small-count{color:var(--blue);font-variant-numeric:tabular-nums}.queue-list{display:grid;gap:10px;overflow:auto;max-height:calc(100% - 34px);padding-right:3px}.mini-card{display:grid;grid-template-columns:48px 1fr auto;gap:10px;align-items:center;border:1px solid var(--line);border-radius:14px;padding:9px;background:rgba(255,255,255,.055);transition:opacity var(--normal),transform var(--normal),border-color var(--fast)}.mini-card.processing{border-color:rgba(56,189,248,.52);box-shadow:0 0 0 1px rgba(56,189,248,.16)}.mini-card.done{opacity:.35;transform:translateX(10px)}.mini-thumb{width:48px;height:42px;display:grid;place-items:center;border-radius:10px;background:rgba(51,65,85,.6);color:#cbd5e1;overflow:hidden}.mini-thumb img{width:100%;height:100%;object-fit:cover}.mini-title{font-size:12px;color:#dbeafe;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.mini-sub{font-size:11px;color:var(--muted);margin-top:4px}.center-lab{display:grid;grid-template-rows:auto 1fr auto;gap:14px}.scan-lab{min-height:230px;border-radius:22px;background:rgba(15,23,42,.48);border:1px solid var(--line);display:grid;place-items:center;position:relative;overflow:hidden}.pulse-ring{position:absolute;width:170px;height:170px;border-radius:50%;border:1px solid rgba(56,189,248,.32);opacity:0}.scan-lab.running .pulse-ring:nth-child(1){animation:pulse 1.4s infinite}.scan-lab.running .pulse-ring:nth-child(2){animation:pulse 1.4s .42s infinite}@keyframes pulse{0%{transform:scale(.62);opacity:.72}100%{transform:scale(1.35);opacity:0}}.scan-beam{position:absolute;inset:-20%;background:linear-gradient(100deg,transparent 38%,rgba(56,189,248,.18),transparent 62%);transform:translateX(-80%)}.scan-lab.running .scan-beam{animation:beam 1s linear infinite}@keyframes beam{to{transform:translateX(80%)}}.processing-card{width:220px;border:1px solid rgba(56,189,248,.32);border-radius:18px;overflow:hidden;background:rgba(15,23,42,.92);box-shadow:0 20px 45px rgba(0,0,0,.34);transform:translateY(0);transition:transform 520ms ease,opacity 520ms ease;z-index:2}.processing-card.pass{transform:translate(265px,90px) scale(.42);opacity:0}.processing-card.fail{transform:translate(250px,210px) rotate(10deg) scale(.35);opacity:0}.metric-console{border:1px solid var(--line);background:rgba(2,6,23,.58);border-radius:17px;padding:14px;min-height:142px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#cbd5e1;line-height:1.8;white-space:pre-wrap}.metric-console .accent{color:var(--blue)}.metric-console .ok{color:#86efac}.metric-console .bad{color:#fca5a5}.bucket{min-height:44%;border:1px solid var(--line);background:rgba(255,255,255,.04);border-radius:18px;padding:14px;margin-bottom:14px;transition:border-color var(--fast),background var(--fast)}.bucket.highlight{border-color:rgba(16,185,129,.65);background:rgba(16,185,129,.10)}.bucket.reject.flash{border-color:rgba(185,28,28,.72);background:rgba(185,28,28,.12);animation:rejectFlash .46s ease}@keyframes rejectFlash{0%,100%{transform:translateX(0)}35%{transform:translateX(-3px)}70%{transform:translateX(3px)}}.bucket-head{display:flex;justify-content:space-between;align-items:center;font-size:14px;color:#e2e8f0;font-weight:700;margin-bottom:10px}.bucket-grid{display:flex;flex-wrap:wrap;gap:8px;align-content:flex-start}.kept-chip{display:flex;align-items:center;gap:6px;max-width:160px;border:1px solid rgba(16,185,129,.24);color:#bbf7d0;background:rgba(16,185,129,.08);border-radius:999px;padding:7px 9px;font-size:11px;animation:fadeScale .3s ease}
    .extract-list{height:100%;overflow-y:auto;overflow-x:hidden;display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:18px;align-content:start;padding:0 8px 28px 0;scroll-behavior:smooth;perspective:1600px}.extract-card{min-height:238px;opacity:0;transform:translateY(20px) rotateX(4deg);position:relative;perspective:1300px}.extract-card.show{animation:slideUp .42s ease forwards}.extract-inner{position:relative;width:100%;height:100%;min-height:238px;transition:transform .72s cubic-bezier(.2,.78,.24,1);transform-style:preserve-3d}.extract-card.flipped .extract-inner{transform:rotateY(180deg)}.extract-face{position:absolute;inset:0;backface-visibility:hidden;border:1px solid var(--line);border-radius:20px;overflow:hidden}.extract-back{display:grid;place-items:center;align-content:center;gap:10px;background:linear-gradient(135deg,rgba(15,23,42,.92),rgba(30,41,59,.72));color:#cbd5e1}.extract-back::before{content:"";position:absolute;inset:16px;border:1px dashed rgba(148,163,184,.25);border-radius:16px}.extract-back i{font-size:30px;color:var(--blue);z-index:1}.extract-back strong{font-size:16px;z-index:1}.extract-back span{font-size:12px;color:var(--muted);z-index:1}.extract-front{transform:rotateY(180deg);background:rgba(255,255,255,.06);display:grid;grid-template-columns:122px minmax(0,1fr);gap:16px;padding:16px;overflow:visible}.extract-front::after{content:"";position:absolute;top:0;bottom:0;width:2px;left:0;background:rgba(56,189,248,.44);opacity:.55}.extract-thumb{height:122px;border-radius:15px;display:grid;place-items:center;background:rgba(51,65,85,.65);color:#cbd5e1;font-size:26px;overflow:hidden}.extract-thumb img{width:100%;height:100%;object-fit:cover}.real-content{transition:opacity .35s ease;min-width:0}.skeleton{position:absolute;inset:16px;left:152px}.sk{height:13px;border-radius:999px;background:linear-gradient(90deg,rgba(148,163,184,.14),rgba(226,232,240,.22),rgba(148,163,184,.14));background-size:220% 100%;animation:loading 1s infinite;margin-bottom:13px}.sk.w1{width:82%}.sk.w2{width:58%}.sk.w3{width:72%}.sk.w4{width:45%}@keyframes loading{to{background-position:-220% 0}}.extract-title{font-weight:700;font-size:15px;margin-bottom:10px;line-height:1.45;white-space:normal;word-break:break-word}.extract-row{color:var(--muted);font-size:13px;margin:7px 0;display:flex;gap:8px;align-items:flex-start;min-width:0;word-break:break-word}.extract-row i{width:16px;color:var(--blue)}.metrics-line{margin-top:10px;display:flex;gap:10px;flex-wrap:wrap;color:#cbd5e1;font-size:12px}.metrics-line span{border:1px solid var(--line);border-radius:999px;padding:5px 8px;background:rgba(255,255,255,.045)}
    .report-layout{height:100%;display:flex;flex-direction:column;gap:0;min-height:0;overflow-y:auto}.report-box{min-height:0;overflow:auto;color:#e2e8f0;line-height:1.72;font-size:.88rem;padding:4px 0}.report-metric-bar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}.rchip{border:1px solid var(--line);border-radius:999px;padding:4px 12px;background:rgba(255,255,255,.04);font-size:.7rem;color:#cbd5e1}.rchip b{color:var(--muted);font-weight:550;margin-right:4px}
    .topology-layout{height:100%;display:grid;grid-template-columns:1fr 240px;gap:16px;min-height:0}#network{height:100%;min-height:480px;border:1px solid var(--line);border-radius:22px;background:rgba(2,6,23,.30);position:relative}.legend{border:1px solid var(--line);border-radius:22px;padding:18px;background:rgba(2,6,23,.26);align-self:end}.legend h4{margin:0 0 14px;font-size:15px}.legend-item{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:13px;margin:12px 0}.dot{width:11px;height:11px;border-radius:50%}
    .topology-open{position:absolute;right:16px;top:16px;z-index:5;display:inline-flex;align-items:center;gap:8px;border:1px solid rgba(37,99,235,.34);border-radius:999px;background:#2563eb;color:#ffffff;text-decoration:none;font-weight:800;font-size:13px;padding:11px 15px;box-shadow:0 14px 34px rgba(37,99,235,.26)}.topology-open:hover{background:#1d4ed8;color:#ffffff}
    .data-layout{height:100%;display:grid;grid-template-rows:auto 1fr auto;gap:14px;min-height:0}.tabs{display:flex;gap:10px;border-bottom:1px solid var(--line);padding-bottom:12px}.tab{padding:10px 13px;border:1px solid var(--line);border-radius:999px;color:var(--muted);background:rgba(255,255,255,.045);cursor:pointer;transition:all var(--fast)}.tab.active{color:#dbeafe;border-color:rgba(56,189,248,.42);background:rgba(37,99,235,.16)}.tab-panel{min-height:0;overflow:hidden;position:relative}.tab-content{display:none;height:100%;overflow:auto;opacity:0;animation:fadeIn .28s ease forwards}.tab-content.active{display:block}@keyframes fadeIn{to{opacity:1}}table{width:100%;border-collapse:collapse;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:11px 10px;text-align:left;color:#cbd5e1;vertical-align:top}th{color:#e5e7eb;background:rgba(255,255,255,.045);position:sticky;top:0;backdrop-filter:blur(12px)}.terminal{min-height:100%;background:#020617;color:#86efac;border:1px solid rgba(34,197,94,.22);border-radius:18px;padding:16px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;line-height:1.8;white-space:pre-wrap}.json-block{background:rgba(2,6,23,.55);border:1px solid var(--line);border-radius:18px;padding:16px;color:#cbd5e1;font-size:12px;line-height:1.7;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.download-row{display:flex;gap:10px;justify-content:flex-end}.empty{height:100%;display:grid;place-items:center;color:var(--muted);border:1px dashed var(--line);border-radius:18px}.scrollbar::-webkit-scrollbar,.result-grid::-webkit-scrollbar,.extract-list::-webkit-scrollbar,.report-box::-webkit-scrollbar,.tab-content::-webkit-scrollbar,.queue-list::-webkit-scrollbar,.score-panel::-webkit-scrollbar{width:8px;height:8px}.scrollbar::-webkit-scrollbar-thumb,.result-grid::-webkit-scrollbar-thumb,.extract-list::-webkit-scrollbar-thumb,.report-box::-webkit-scrollbar-thumb,.tab-content::-webkit-scrollbar-thumb,.queue-list::-webkit-scrollbar-thumb,.score-panel::-webkit-scrollbar-thumb{background:rgba(148,163,184,.22);border-radius:999px}@media(max-width:1199px){body{overflow:auto}.app-shell{min-width:1180px}}

    /* Layout and print-safe light theme overrides. Keep this block last. */
    :root{--bg:#ffffff;--bg2:#ffffff;--panel:#ffffff;--panel-strong:#ffffff;--card:#ffffff;--card2:#f8fafc;--line:#e2e8f0;--line2:#cbd5e1;--text:#0f172a;--muted:#475569;--subtle:#64748b;--shadow:0 14px 38px rgba(15,23,42,.08)}
    html,body{width:100%;height:100%;margin:0;background:#ffffff!important;color:var(--text)!important;overflow:hidden}
    body::before,body::after{display:none!important}
    .app-shell{width:min(100%,1500px);max-width:1500px;min-width:0;margin:0 auto;padding:24px clamp(16px,2.4vw,30px) 26px;justify-self:center}
    .topbar,.step-card,.zone,.insight-card,.image-preview,.info-card,.result-card,.legend,.tab,.json-block,.dropzone,.bucket,#network{background:#ffffff!important;color:var(--text)!important;border-color:var(--line)!important;box-shadow:var(--shadow)}
    .stage-wrap,.steps-track,.step,.step-card{width:100%;max-width:100%;min-width:0}
    .stage-wrap{margin:0 auto}
    .step{padding:clamp(12px,1.6vw,22px)}
    .step-card{background:#ffffff!important;backdrop-filter:none!important}
    .step-card::before{display:none!important}
    .step-head{border-bottom-color:var(--line)!important}
    .brand h1,.step-title,.zone-title,.bucket-head,.head-stat strong,.upload-inner h3,.result-title,.extract-title,.insight-head,.score-label span,th,td{color:#0f172a!important}
    .brand p,.status-chip,.progress-meta,.step-desc,.status-note,.result-meta,.extract-row,.score-reason,.muted{color:var(--muted)!important}
    .brand-mark,.upload-icon{background:#eef6ff!important;border-color:#bfdbfe!important;color:#2563eb!important}
    .btn,.linkbtn,.step-pill,.tab{background:#ffffff!important;color:#0f172a!important;border-color:#cbd5e1!important}
    .btn:hover,.linkbtn:hover,.tab:hover{background:#f8fafc!important;border-color:#94a3b8!important}
    .btn.primary{background:#2563eb!important;border-color:#2563eb!important;color:#ffffff!important}
    .btn.ghost,.btn.on{background:#f8fafc!important;color:#0f172a!important}
    .dropzone:hover,.dropzone.dragover{background:#f8fbff!important;border-color:#2563eb!important}
    .scan-lab,.metric-console,.extract-back,.processing-card,.engine-card,.validation-visual,.extract-status{background:#f8fafc!important;color:#0f172a!important;border-color:var(--line)!important;box-shadow:none!important}
    .extract-front,.terminal{border-color:var(--line)!important}
    .terminal{background:#0f172a!important;color:#bbf7d0!important}
    .result-note,.candidate-section-title,.rchip,.metrics-line span{background:#f8fafc!important;color:#334155!important;border-color:var(--line)!important}
    .report-box,.insight-body{color:#0f172a!important}
    .score-track{background:#e2e8f0!important}
    table{background:#ffffff!important}
    th{background:#f8fafc!important;color:#0f172a!important;backdrop-filter:none!important}
    td{color:#334155!important}
    .kicker{color:#0284c7!important}.small-count{color:#2563eb!important}.value{color:#0f172a!important}.label{color:#475569!important}
    .mini-title{color:#0f172a!important}.mini-sub{color:#64748b!important}.mini-thumb,.thumb,.extract-thumb{background:#eef2f7!important;color:#2563eb!important}
    .p-baidu{background:#eff6ff!important;color:#1d4ed8!important;border-color:#bfdbfe!important}.p-yandex{background:#fff7ed!important;color:#c2410c!important;border-color:#fed7aa!important}.p-google{background:#f0fdf4!important;color:#15803d!important;border-color:#bbf7d0!important}.p-xhs{background:#fef2f2!important;color:#b91c1c!important;border-color:#fecaca!important}.p-weibo{background:#f8fafc!important;color:#334155!important;border-color:#cbd5e1!important}.p-other{background:#f1f5f9!important;color:#334155!important;border-color:#cbd5e1!important}
    .search-summary{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:2px}.search-stat{border:1px solid var(--line);border-radius:16px;background:#f8fafc;padding:13px 14px}.search-stat b{display:block;color:#0f172a;font-size:20px}.search-stat span{color:#64748b;font-size:12px}
    .engine-board{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin-bottom:4px}.engine-card{min-height:154px;border:1px solid var(--line)!important;background:#ffffff!important;border-radius:18px!important;position:relative;overflow:hidden;padding:15px;display:grid;grid-template-rows:auto 1fr auto;box-shadow:0 12px 28px rgba(15,23,42,.06)!important;opacity:1!important;transform:none!important}.engine-card::before{content:"";position:absolute;inset:0;background:linear-gradient(100deg,transparent 18%,rgba(37,99,235,.08),transparent 42%);transform:translateX(-120%);animation:engineSweep 1.9s ease-in-out infinite}.engine-card.done::before{display:none}.engine-name{position:relative;z-index:1;display:flex;justify-content:space-between;align-items:center;gap:10px;color:#0f172a!important;font-weight:800;font-size:15px}.engine-preview{position:relative;z-index:1;border:1px dashed #cbd5e1;border-radius:14px;background:#f8fafc;display:grid;place-items:center;min-height:64px;margin-top:12px;color:#2563eb;font-size:26px;overflow:hidden}.engine-stats{position:relative;z-index:1;display:flex;justify-content:space-between;align-items:center;margin-top:11px;color:#475569!important;font-size:12px}.engine-meter{height:6px;border-radius:999px;background:#e2e8f0;overflow:hidden;margin-top:10px}.engine-meter i{display:block;height:100%;width:45%;border-radius:999px;background:#2563eb;animation:meterPulse 1.4s ease-in-out infinite}@keyframes meterPulse{0%{transform:translateX(-60%)}100%{transform:translateX(240%)}}
    .validation-progress{display:grid;gap:10px}.validation-progress-head{display:flex;justify-content:space-between;gap:10px;font-size:13px;color:#334155}.validation-progress-track{height:9px;border-radius:999px;background:#e2e8f0;overflow:hidden}.validation-progress-track i{display:block;height:100%;width:0%;background:linear-gradient(90deg,#2563eb,#10b981);transition:width .4s ease}.decision-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}.decision-chip{border:1px solid #cbd5e1;border-radius:999px;background:#f8fafc;color:#334155;padding:5px 9px;font-size:11px}.decision-chip.keep{border-color:#86efac;background:#f0fdf4;color:#15803d}.decision-chip.reject{border-color:#fecaca;background:#fef2f2;color:#b91c1c}
    .flight-deck{position:absolute;inset:0;pointer-events:none;overflow:visible;z-index:3}.flight-card{--tx:285px;--dy:-92px;position:absolute;left:50%;top:50%;width:178px;border:1px solid #93c5fd;border-radius:16px;background:#ffffff;box-shadow:0 18px 42px rgba(15,23,42,.20);overflow:hidden;transform:translate(-50%,-50%) scale(.86);animation:flyKeep .82s cubic-bezier(.2,.75,.18,1) forwards}.flight-card::after{content:"";position:absolute;left:18px;right:18px;bottom:8px;height:2px;border-radius:999px;background:linear-gradient(90deg,transparent,#2563eb,transparent);opacity:.55}.flight-card.reject{--dy:96px;border-color:#fecaca;animation-name:flyReject}.flight-card.keep{border-color:#86efac}.flight-card.lane1{--dy:-36px}.flight-card.lane2{--dy:38px}.flight-card.lane3{--dy:118px}.flight-card .thumb{height:58px}.flight-mini{padding:9px 10px}.flight-mini b{display:block;color:#0f172a;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.flight-mini span{display:block;margin-top:5px;color:#64748b;font-size:11px}.flight-card.keep .flight-mini span{color:#15803d}.flight-card.reject .flight-mini span{color:#b91c1c}@keyframes flyKeep{0%{opacity:0;transform:translate(-50%,-50%) scale(.75)}16%{opacity:1;transform:translate(calc(-50% - 8px),calc(-50% - 8px)) scale(.94)}68%{opacity:.95;transform:translate(calc(-50% + var(--tx) * .62),calc(-50% + var(--dy) * .62)) scale(.74)}100%{opacity:0;transform:translate(calc(-50% + var(--tx)),calc(-50% + var(--dy))) scale(.34)}}@keyframes flyReject{0%{opacity:0;transform:translate(-50%,-50%) rotate(0) scale(.75)}16%{opacity:1;transform:translate(calc(-50% - 6px),calc(-50% + 6px)) rotate(-2deg) scale(.94)}68%{opacity:.95;transform:translate(calc(-50% + var(--tx) * .62),calc(-50% + var(--dy) * .62)) rotate(6deg) scale(.72)}100%{opacity:0;transform:translate(calc(-50% + var(--tx)),calc(-50% + var(--dy))) rotate(12deg) scale(.32)}}
    .report-hero{border:1px solid var(--line);border-radius:18px;background:#f8fafc;padding:16px 18px;margin-bottom:14px}.report-hero-title{font-size:18px;font-weight:800;color:#0f172a;margin-bottom:8px}.report-hero-body{font-size:14px;line-height:1.7;color:#334155}.report-stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}.report-stat{border:1px solid var(--line);border-radius:15px;background:#ffffff;padding:12px 13px}.report-stat span{display:block;color:#64748b;font-size:12px;margin-bottom:4px}.report-stat strong{display:block;color:#0f172a;font-size:17px;line-height:1.25}.report-section{border:1px solid var(--line);border-radius:16px;background:#ffffff;padding:15px 16px;margin-bottom:12px;break-inside:avoid}.report-section h3{margin:0 0 10px;color:#0f172a;font-size:16px}.report-section p{margin:8px 0;color:#334155;line-height:1.75}.report-list{display:grid;gap:8px;margin:8px 0 0;padding:0;list-style:none}.report-list li{position:relative;padding-left:17px;color:#334155;line-height:1.65}.report-list li::before{content:"";position:absolute;left:0;top:.72em;width:6px;height:6px;border-radius:50%;background:#2563eb}
    .topology-layout,.upload-layout,.validate-layout,.data-layout{min-width:0}
    .stage-wrap{perspective:none!important;overflow:hidden!important}.steps-track{position:relative!important;display:block!important;height:100%!important;width:100%!important;transform:none!important;transition:none!important;transform-style:flat!important}.step{position:absolute!important;inset:0!important;width:100%!important;max-width:100%!important;min-width:0!important;height:100%!important;transform:none!important;opacity:0!important;visibility:hidden!important;transition:opacity 180ms ease!important;pointer-events:none!important}.step.active{transform:none!important;opacity:1!important;visibility:visible!important;pointer-events:auto!important}
    @media(max-width:1199px){html,body{overflow:auto}.app-shell{width:100%;min-width:0;height:auto;min-height:100vh}.stage-wrap{min-height:760px}.topbar{grid-template-columns:1fr;align-items:start}.progress-meta{text-align:left}.nav-actions{justify-content:flex-start}.upload-layout,.validate-layout,.topology-layout{grid-template-columns:1fr}.extract-list{grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}}
    @media print{html,body{height:auto!important;overflow:visible!important;background:#ffffff!important}.app-shell{display:block;width:100%;max-width:none;height:auto;padding:0}.topbar,.tabs,.download-row,.nav-actions,.step:not(.active),.legend{display:none!important}.stage-wrap,.steps-track,.step,.step-card,.step-body,.report-layout,.report-box{display:block!important;height:auto!important;min-height:0!important;overflow:visible!important;transform:none!important;background:#ffffff!important;box-shadow:none!important;border:0!important;padding:0!important}.step-head{border:0!important;padding:0 0 12px!important}.insight-card,.rchip{break-inside:avoid;background:#ffffff!important;color:#0f172a!important;box-shadow:none!important}}
  </style>
</head>
<body>
<div class="app-shell">
<header class="topbar">
  <div class="brand"><div class="brand-mark"><i class="fa-solid fa-diagram-project"></i></div><div><h1>图片溯源智能体系统</h1><p>Retrieval · Validation · Analysis · Orchestration</p></div></div>
  <div class="progress-meta"><div class="step-pill" id="stepPill">步骤 1 / 7</div><div class="status-chip" id="statusChip">等待上传图片</div></div>
  <div class="nav-actions"><button class="btn ghost" id="prevBtn"><i class="fa-solid fa-arrow-left"></i> 上一步</button><button class="btn ghost on" id="autoBtn"><i class="fa-solid fa-wand-magic-sparkles"></i> 自动跟随</button><button class="btn primary" id="nextBtn">下一步 <i class="fa-solid fa-arrow-right"></i></button><button class="btn ghost" id="loadLatestBtn">载入最近报告</button><button class="btn ghost" id="testBtn">测试预览</button></div>
</header>
<main class="stage-wrap"><div class="steps-track" id="stepsTrack">
<section class="step active"><div class="step-card"><div class="step-head"><div><div class="kicker">Step 01</div><h2 class="step-title">上传图片</h2><p class="step-desc"></p></div><div class="head-stat"><strong id="uploadState">待上传</strong></div></div><div class="step-body"><div class="upload-layout"><label class="dropzone" id="dropzone"><input id="fileInput" type="file" accept="image/*" hidden/><div class="upload-inner"><div class="upload-icon"><i class="fa-solid fa-cloud-arrow-up"></i></div><h3>点击选择图片文件上传</h3><p></p></div></label><div class="preview-panel"><div class="image-preview" id="imagePreview"></div><div class="info-grid" id="infoGrid"></div></div></div></div></div></section>
<section class="step"><div class="step-card"><div class="scan-overlay" id="searchScan"></div><div class="step-head"><div><div class="kicker">Step 02</div><h2 class="step-title">多平台检索结果 <span style="font-size:11px;color:#94a3b8;font-weight:400">张林翔</span></h2><p class="step-desc"></p></div><div class="head-stat"><span>候选资源</span><strong id="resultCount">--</strong></div></div><div class="step-body"><div class="result-grid" id="resultGrid"></div></div></div></section>
<section class="step"><div class="step-card"><div class="step-head"><div><div class="kicker">Step 03</div><h2 class="step-title">去重校验 <span style="font-size:11px;color:#94a3b8;font-weight:400">黄子倩</span></h2><p class="step-desc"></p></div><div class="head-stat"><span>校验状态</span><strong id="validateState">等待</strong></div></div><div class="step-body"><div class="validate-layout"><div class="zone"><div class="zone-title">待校验队列 <span class="small-count" id="queueCount">0</span></div><div class="queue-list" id="queueList"></div></div><div class="zone center-lab"><div class="zone-title">校验中 <span class="small-count" id="currentName">等待任务</span></div><div class="scan-lab" id="scanLab"><div class="pulse-ring"></div><div class="pulse-ring"></div><div class="scan-beam"></div><div id="processingMount" class="status-note"></div></div><div class="metric-console" id="metricConsole"></div></div><div class="zone"><div class="bucket keep" id="keepBucket"><div class="bucket-head"><span><i class="fa-regular fa-folder-open"></i> 保留池</span><span id="keepCount">0</span></div><div class="bucket-grid" id="keepGrid"></div></div><div class="bucket reject" id="rejectBucket"><div class="bucket-head"><span><i class="fa-regular fa-trash-can"></i> 丢弃</span><span id="rejectCount">0</span></div><p class="status-note"></p></div></div></div></div></div></section>
<section class="step"><div class="step-card"><div class="step-head"><div><div class="kicker">Step 04</div><h2 class="step-title">网页信息提取 <span style="font-size:11px;color:#94a3b8;font-weight:400">秦家欣</span></h2><p class="step-desc"></p></div><div class="head-stat"><span>有效节点</span><strong id="extractCount">--</strong></div></div><div class="step-body"><div class="extract-list" id="extractList"></div></div></div></section>
<section class="step"><div class="step-card"><div class="step-head"><div><div class="kicker">Step 05</div><h2 class="step-title">传播拓扑 <span style="font-size:11px;color:#94a3b8;font-weight:400">秦家欣</span></h2><p class="step-desc">G6 交互式传播拓扑</p></div><div class="head-stat"><span>节点 / 边</span><strong id="topologyCount">--</strong></div></div><div class="step-body"><div class="topology-layout"><div id="network"></div><aside class="legend"><h4>图例</h4><div class="legend-item"><span class="dot" style="background:#10b981"></span> 疑似源头</div><div class="legend-item"><span class="dot" style="background:#2563eb"></span> 关键节点</div><div class="legend-item"><span class="dot" style="background:#d97706"></span> 传播节点</div><div class="legend-item"><span class="dot" style="background:#64748b"></span> 普通节点</div><p class="status-note"></p></aside></div></div></div></section>
<section class="step"><div class="step-card"><div class="step-head"><div><div class="kicker">Step 06</div><h2 class="step-title">溯源分析报告</h2><p class="step-desc">AI 综合生成洞察结论与多维评估</p></div></div><div class="step-body"><div class="report-layout"><article class="report-box" id="reportBox"><div class="status-note">分析中...</div></article></div></div></div></section>
<section class="step"><div class="step-card"><div class="step-head"><div><div class="kicker">Step 07 · Data</div><h2 class="step-title">节点数据与运行日志</h2><p class="step-desc">查看过滤节点表格、关键日志、原始状态，并下载 JSON/TXT 文件。</p></div><div class="head-stat"><span>导出</span><strong>3</strong></div></div><div class="step-body"><div class="data-layout"><nav class="tabs"><button class="tab active" data-tab="table">过滤节点数据</button><button class="tab" data-tab="logs">运行日志</button><button class="tab" data-tab="raw">原始信息</button></nav><div class="tab-panel"><div class="tab-content active" id="tab-table"><div class="empty">等待数据生成</div></div><div class="tab-content" id="tab-logs"><div class="terminal" id="terminal日志"></div></div><div class="tab-content" id="tab-raw"><pre class="json-block" id="rawJson">{}</pre></div></div><div class="download-row"><a class="linkbtn" id="downloadJson" download><i class="fa-solid fa-file-code"></i> 下载 JSON</a><a class="linkbtn" id="downloadTxt" download><i class="fa-solid fa-file-lines"></i> 下载日志 TXT</a><a class="linkbtn" id="downloadReport" download><i class="fa-solid fa-file-arrow-down"></i> 下载报告</a></div></div></div></div></section>
</div></main></div>
<script>
const TOTAL_STEPS=7;const $=id=>document.getElementById(id);let currentStep=0,uploadedInfo=null,activeReport=null,reportRendered=false,network=null,currentStatus={},activeJobId=null,pollTimer=null,hasUploadedThisPage=false;let autoFollow=true,manualPauseUntil=0,queuedAutoTimer=null,lastAutoStepAt=0;let validationLiveLoop=false,validationLiveToken=0,validationFinalRendered=false,extractionLiveRendered=false,analyzeStartedAt=0;let validationSeenEvents=new Set();let validationVisual={processed:0,keep:0,reject:0,input:0,queueData:[],finalizing:false,startedAt:0};const rendered={search:false,validate:false,extract:false,report:false,network:false,data:false};
function esc(v){return String(v??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}function wait(ms){return new Promise(r=>setTimeout(r,ms))}
function setStep(index,source='manual'){currentStep=Math.max(0,Math.min(TOTAL_STEPS-1,index));$('stepsTrack').style.transform=`translateX(-${currentStep*100}%)`;document.querySelectorAll('.step').forEach((s,i)=>s.classList.toggle('active',i===currentStep));$('stepPill').textContent=`步骤 ${currentStep+1} / ${TOTAL_STEPS}`;$('prevBtn').disabled=currentStep===0;$('nextBtn').disabled=currentStep===TOTAL_STEPS-1;if(source==='manual')manualPauseUntil=Date.now()+9000;runStepEnter(currentStep)}
function autoStep(target){if(!autoFollow)return;if(Date.now()<manualPauseUntil)return;target=Math.max(0,Math.min(6,target));if(target>currentStep+1)target=currentStep+1;if(target===currentStep)return;clearTimeout(queuedAutoTimer);const delay=Math.max(900,4800-(Date.now()-lastAutoStepAt));queuedAutoTimer=setTimeout(()=>{if(autoFollow&&Date.now()>=manualPauseUntil&&target!==currentStep){lastAutoStepAt=Date.now();setStep(target,'auto')}},delay)}
$('prevBtn').onclick=()=>setStep(currentStep-1,'manual');$('nextBtn').onclick=()=>setStep(currentStep+1,'manual');$('autoBtn').onclick=()=>{autoFollow=!autoFollow;$('autoBtn').classList.toggle('on',autoFollow);$('autoBtn').innerHTML=autoFollow?'<i class="fa-solid fa-wand-magic-sparkles"></i> 自动跟随':'<i class="fa-solid fa-pause"></i> 手动浏览'};
function runStepEnter(step){
  if(step===1){$('searchScan').classList.remove('run');void $('searchScan').offsetWidth;$('searchScan').classList.add('run');if(activeReport)renderSearchResults(activeReport,true);else renderSearchSkeleton()}
  if(step===2){$('scanLab').classList.add('running');if(activeReport){if(validationHasFinalSummary(activeReport))animateValidation(activeReport,true);else animateValidationLive(activeReport)}else renderValidationSkeleton()}
  else if(step!==2){$('scanLab').classList.remove('running')}
  if(step===3){if(activeReport){const hasAnalysis=!!(activeReport.summary?.analysis_summary&&Object.keys(activeReport.summary.analysis_summary).length)||!!(activeReport.summary?.report);if(hasAnalysis)renderExtraction(activeReport,true);else renderExtractionSkeleton(activeReport)}else renderExtractionSkeleton()}
  if(step===4&&activeReport)renderReport(activeReport,true);if(step===5&&activeReport)drawNetwork(activeReport,true);if(step===6&&activeReport)renderDataTabs(activeReport)}
const dropzone=$('dropzone'),fileInput=$('fileInput');dropzone.addEventListener('dragover',e=>{e.preventDefault();dropzone.classList.add('dragover')});dropzone.addEventListener('dragleave',()=>dropzone.classList.remove('dragover'));dropzone.addEventListener('drop',e=>{e.preventDefault();dropzone.classList.remove('dragover');handleFile(e.dataTransfer.files[0])});fileInput.addEventListener('change',e=>handleFile(e.target.files[0]));
function fakeMd5(name,size){let base=`${name}-${size}-${Date.now()}`,hash=0;for(let i=0;i<base.length;i++)hash=((hash<<5)-hash)+base.charCodeAt(i)|0;return Math.abs(hash).toString(16).padStart(8,'0')+'a9c2d4e7f031b6c8'.slice(0,24)}
function renderInfoCards(){const icons=['fa-file-signature','fa-hard-drive','fa-expand','fa-code','fa-fingerprint','fa-circle-check'];$('infoGrid').innerHTML=Object.entries(uploadedInfo).map(([k,v],i)=>`<div class="info-card" style="animation-delay:${i*90}ms"><div class="label"><i class="fa-solid ${icons[i]||'fa-circle-info'}"></i>${k}</div><div class="value" title="${esc(v)}">${esc(v)}</div></div>`).join('');setTimeout(()=>document.querySelectorAll('.info-card').forEach(c=>c.classList.add('reveal')),30)}
async function handleFile(file){if(!file)return;resetForNewRun();const url=URL.createObjectURL(file);const img=new Image();img.onload=()=>{uploadedInfo={文件名:file.name,文件大小:`${(file.size/1024).toFixed(1)} KB`,图片尺寸:`${img.width} × ${img.height}`,文件格式:file.type||'image/*',模拟MD5:fakeMd5(file.name,file.size),状态:'已上传，真实流程运行中'};$('imagePreview').innerHTML=`<img src="${url}" alt="preview"/>`;$('uploadState').textContent='已上传';renderInfoCards()};img.src=url;$('statusChip').textContent='上传中，正在启动真实流程...';const fd=new FormData();fd.append('file',file,file.name);try{const res=await fetch('/api/upload',{method:'POST',body:fd});const data=await res.json();if(!res.ok||!data.ok)throw new Error(data.error||'上传失败');activeJobId=data.job_id||null;hasUploadedThisPage=true;manualPauseUntil=0;$('statusChip').textContent='真实流程已启动';setStep(0,'auto');startPolling();pollStatus(true)}catch(e){$('statusChip').textContent='上传失败：'+e.message;$('uploadState').textContent='上传失败'}}
function resetForNewRun(){activeReport=null;reportRendered=false;validationLiveLoop=false;validationLiveToken++;validationFinalRendered=false;extractionLiveRendered=false;analyzeStartedAt=0;validationSeenEvents=new Set();validationVisual={processed:0,keep:0,reject:0,input:0,queueData:[],finalizing:false,startedAt:0};network&&network.destroy();network=null;Object.keys(rendered).forEach(k=>rendered[k]=false);['resultGrid','queueList','keepGrid','extractList'].forEach(id=>$(id).innerHTML='');$('processingMount').className='status-note';$('processingMount').innerHTML='';$('resultCount').textContent='--';$('queueCount').textContent='0';$('keepCount').textContent='0';$('rejectCount').textContent='0';$('extractCount').textContent='--';$('topologyCount').textContent='--';$('reportBox').innerHTML='分析中...';$('terminal日志').textContent='';$('rawJson').textContent='{}'}
function nodePlatform(n){const text=[n.platform,n.platform_family,n.source,n.engine,n.url].map(x=>String(x||'').toLowerCase()).join(' ');if(text.includes('baidu')||text.includes('百度'))return'百度';if(text.includes('yandex'))return'Yandex';if(text.includes('google'))return'Google';if(text.includes('xiaohongshu')||text.includes('xhs')||text.includes('小红书'))return'小红书';if(text.includes('weibo')||text.includes('微博'))return'微博';return n.platform||n.source||n.engine||'外部网页'}
function platformClass(p){p=String(p);if(p.includes('百度'))return'p-baidu';if(p.includes('Yandex'))return'p-yandex';if(p.includes('Google'))return'p-google';if(p.includes('小红书'))return'p-xhs';if(p.includes('微博'))return'p-weibo';return'p-other'}
function thumbHtml(n,mini=false){const src=n.thumbnail_url||n.image_url||n.cached_image_url||'';if(src&&String(src).startsWith('http'))return`<img src="${esc(src)}" onerror="this.replaceWith(document.createElement('i'))"/>`;return`<i class="fa-solid ${mini?'fa-image':'fa-newspaper'}"></i>`}
function candidateNodes(data,limit=null){const raw=Array.isArray(data?.candidates)&&data.candidates.length?data.candidates:(Array.isArray(data?.nodes)?data.nodes:[]);return limit?raw.slice(0,limit):raw}
function finalNodes(data){return Array.isArray(data?.nodes)?data.nodes:[]}
function nodeKey(n){return String(n?.url||n?.canonical_url||n?.page_url||n?.image_url||n?.id||'')}
function renderSearchSkeleton(){
  if($('resultGrid').children.length)return;
  const names=['baidu','yandex','serpapi_lens','tineye','bing','google','saucenao','ascii2d','mitmproxy'];
  $('resultGrid').innerHTML='<div class="search-summary"><div class="search-stat"><b>多源检索</b><span>按引擎并行收集候选链接</span></div><div class="search-stat"><b>实时归并</b><span>同源 URL 与图片候选会进入统一池</span></div><div class="search-stat"><b>等待结果</b><span>完成后按平台展示候选资源</span></div></div><div class="engine-board">'+names.map(name=>`<article class="engine-card"><div class="engine-name"><span><i class="fa-solid ${engineIcon(name)}"></i> ${engineDisplay(name)}</span><span class="platform ${platformClass(engineDisplay(name))}">检索中</span></div><div class="engine-preview"><i class="fa-solid ${engineIcon(name)}"></i></div><div><div class="engine-stats"><span>searching</span><span>--</span></div><div class="engine-meter"><i></i></div></div></article>`).join('')+'</div>';
}
function engineDisplay(name){const t=String(name||'').toLowerCase();if(t.includes('baidu'))return'百度';if(t.includes('yandex'))return'Yandex';if(t.includes('google')||t.includes('serpapi'))return t.includes('serpapi')?'SerpApi Lens':'Google';if(t.includes('tineye'))return'TinEye';if(t.includes('bing'))return'Bing';if(t.includes('saucenao'))return'SauceNAO';if(t.includes('ascii'))return'Ascii2D';if(t.includes('mitm'))return'Mitmproxy';if(t.includes('weibo'))return'微博';if(t.includes('xiaohongshu')||t.includes('xhs'))return'小红书';return name||'外部引擎'}
function engineIcon(name){const t=String(name||'').toLowerCase();if(t.includes('baidu'))return'fa-magnifying-glass-chart';if(t.includes('yandex'))return'fa-globe';if(t.includes('google')||t.includes('serpapi'))return'fa-camera-retro';if(t.includes('tineye'))return'fa-eye';if(t.includes('bing'))return'fa-magnifying-glass';if(t.includes('saucenao')||t.includes('ascii'))return'fa-fingerprint';if(t.includes('mitm'))return'fa-mobile-screen-button';return'fa-diagram-project'}
function groupCandidatesByEngine(nodes){const m=new Map();nodes.forEach(n=>{const key=String(n.engine||n.source||nodePlatform(n)||'external').trim()||'external';if(!m.has(key))m.set(key,[]);m.get(key).push(n)});return m}
function renderSearchResults(data,replay=false){
  if(rendered.search&&!replay)return;
  const nodes=candidateNodes(data);const rs=data.summary?.retrieval_summary||{};const count=rs.result_count??rs.candidate_count??nodes.length;$('resultCount').textContent=count||nodes.length||0;const grid=$('resultGrid');
  if(!nodes.length){renderSearchSkeleton();return}
  const groups=groupCandidatesByEngine(nodes);const diag=rs.engine_diagnostics||{};const used=Array.isArray(rs.engines_used)?rs.engines_used:[...groups.keys()];
  const engineCards=used.map((name,i)=>{const items=groups.get(name)||[];const d=diag[name]||{};const returned=d.returned??items.length;const raw=d.raw_count??d.normalized_count??returned;const status=d.status||'success';return `<article class="engine-card done" style="animation-delay:${i*70}ms"><div class="engine-name"><span><i class="fa-solid ${engineIcon(name)}"></i> ${engineDisplay(name)}</span><span class="platform ${platformClass(engineDisplay(name))}">${esc(status)}</span></div><div class="engine-preview"><strong>${returned}</strong></div><div class="engine-stats"><span>raw ${raw}</span><span>returned ${returned}</span></div></article>`}).join('');
  const stats='<div class="search-summary"><div class="search-stat"><b>'+esc(count||nodes.length)+'</b><span>候选资源总数</span></div><div class="search-stat"><b>'+esc(used.length)+'</b><span>有效检索引擎</span></div><div class="search-stat"><b>'+esc(Object.keys(diag).length||used.length)+'</b><span>引擎诊断记录</span></div></div>';
  const resultCards=nodes.slice(0,260).map((n,i)=>`<article class="result-card" style="animation-delay:${Math.min(i,60)*32}ms"><div class="thumb">${thumbHtml(n)}</div><div class="result-content"><span class="platform ${platformClass(nodePlatform(n))}">${esc(nodePlatform(n))}</span><div class="result-title" title="${esc(n.title||n.description||'无标题候选资源')}">${esc(n.title||n.description||'无标题候选资源')}</div><div class="result-meta"><span title="${esc(n.engine||n.source||n.domain||'source')}">${esc(n.engine||n.source||n.domain||'source')}</span><span>${n.similarity!=null?Number(n.similarity).toFixed(2):'候选'}</span></div></div></article>`).join('');
  grid.innerHTML=stats+'<div class="engine-board">'+engineCards+'</div>'+resultCards;
  setTimeout(()=>{grid.querySelectorAll('.result-card').forEach((c,i)=>{if(i<80)c.classList.add('show');else{c.classList.add('show');c.style.animationDelay=(80*25)+'ms'}})},80);
  rendered.search=true
}
function renderValidationSkeleton(){
  if($('queueList').children.length)return;const names=['百度候选资源','Yandex 图片索引','Google 页面结果','小红书笔记','微博转发节点','外部网页'];
  $('queueList').innerHTML=names.map((t,i)=>`<div class="mini-card"><div class="mini-thumb"><i class="fa-solid fa-image"></i></div><div><div class="mini-title">${t}</div><div class="mini-sub">等待 validate_node</div></div><i class="fa-solid fa-chevron-right muted"></i></div>`).join('');
  $('queueCount').textContent=names.length;$('processingMount').innerHTML='<div class="status-note"></div>';$('metricConsole').className='validation-visual';$('metricConsole').innerHTML='<div><strong></strong></div><div class="validation-visual-bar"><i></i></div><div></div>'
}

function validationHasFinalSummary(data){const vs=data.summary?.validation_summary||{};return !!(Object.keys(vs).length&&(vs.validated_count!=null||vs.rejected_count!=null||vs.deduplicated_count!=null||Array.isArray(data.nodes)&&data.nodes.length))}
function resetValidationVisual(inputCount,queueData){validationSeenEvents=new Set();validationVisual={processed:0,keep:0,reject:0,input:inputCount,queueData:queueData||[],finalizing:false,startedAt:Date.now()};$('processingMount').className='status-note';$('processingMount').innerHTML='';$('keepGrid').innerHTML='';$('keepCount').textContent='0';$('rejectCount').textContent='0';$('queueCount').textContent=inputCount||0}
function validationStatusBlock(text,current=validationVisual.processed,total=validationVisual.input,keep=validationVisual.keep,reject=validationVisual.reject,decisions={}){const pct=total?Math.min(100,Math.round(current/Math.max(1,total)*100)):0;const chips=Object.entries(decisions||{}).slice(0,5).map(([k,v])=>`<span class="decision-chip">${esc(k)} ${esc(v)}</span>`).join('');$('metricConsole').className='validation-progress';$('metricConsole').innerHTML=`<div class="validation-progress-head"><strong>${esc(text||'校验中')}</strong><span>${current||0}/${total||0} · ${pct}%</span></div><div class="validation-progress-track"><i style="width:${pct}%"></i></div><div class="decision-row"><span class="decision-chip keep">保留 ${keep||0}</span><span class="decision-chip reject">筛除 ${reject||0}</span>${chips}</div><div class="status-note">进度来自后端 validate_node；动画只抽样展示，最终数量以校验汇总为准。</div>`}
function validationQueueFrom(data){const candidates=candidateNodes(data);const inputCount=candidates.length||Number(data.summary?.validation_summary?.input_count||0)||0;const queueData=candidates.length?candidates:Array.from({length:Math.min(inputCount||24,120)},(_,i)=>({title:`候选资源 ${i+1}`,platform:'candidate',engine:'validator'}));return {inputCount,queueData}}
function renderValidationQueue(queueData,inputCount){const display=queueData.slice(0,220);$('queueList').innerHTML=display.map((n,i)=>`<div class="mini-card" data-idx="${i}"><div class="mini-thumb">${thumbHtml(n,true)}</div><div><div class="mini-title" title="${esc(n.title||n.description||('候选资源 '+(i+1)))}">${esc(n.title||n.description||('候选资源 '+(i+1)))}</div><div class="mini-sub">${esc(nodePlatform(n))} · ${esc(n.engine||n.source||'candidate')}</div></div><i class="fa-solid fa-chevron-right muted"></i></div>`).join('')+(queueData.length>220?`<div class="status-note">还有 ${queueData.length-220} 条候选不展开，仍计入后端校验。</div>`:'');$('queueCount').textContent=inputCount||queueData.length}
function showValidationCard(n,idx,total){$('currentName').textContent=`${Math.min(idx+1,total)}/${total||'-'}`;$('processingMount').innerHTML=`<div class="processing-card" id="procCard"><div class="thumb">${thumbHtml(n)}</div><div class="result-content"><span class="platform ${platformClass(nodePlatform(n))}">${esc(nodePlatform(n))}</span><div class="result-title">${esc(n.title||n.description||('候选资源 '+(idx+1)))}</div></div></div>`;document.querySelectorAll('.mini-card.processing').forEach(x=>x.classList.remove('processing'));const q=document.querySelector(`.mini-card[data-idx="${idx}"]`);if(q){q.classList.add('processing');if(idx%18===0)q.scrollIntoView({block:'nearest'});}return q}
function ensureFlightDeck(){const mount=$('processingMount');if(!mount.classList.contains('flight-deck')){mount.className='flight-deck';mount.innerHTML=''}return mount}
function validationEventKey(ev){return String(ev.seq??ev.updated_at??'')+'-'+String(ev.current??ev.index??'')+'-'+String(ev.decision??'')}
function launchValidationFlight(ev,total){const n=ev.node||{};const idx=Number(ev.index??((ev.current||1)-1));const keep=String(ev.decision||'keep')==='keep';const deck=ensureFlightDeck();$('currentName').textContent=`${Math.min((ev.current||idx+1),total||'-')}/${total||'-'}`;document.querySelectorAll('.mini-card.processing').forEach(x=>x.classList.remove('processing'));const q=document.querySelector(`.mini-card[data-idx="${idx}"]`);if(q){q.classList.add('processing');setTimeout(()=>{q.classList.remove('processing');q.classList.add('done')},420);if(idx%18===0)q.scrollIntoView({block:'nearest'});}const card=document.createElement('article');card.className=`flight-card ${keep?'keep':'reject'} lane${Math.abs(idx)%4}`;card.innerHTML=`<div class="thumb">${thumbHtml(n)}</div><div class="flight-mini"><b title="${esc(n.title||n.description||('候选资源 '+(idx+1)))}">${esc(n.title||n.description||('候选资源 '+(idx+1)))}</b><span><i class="fa-solid ${keep?'fa-circle-check':'fa-circle-xmark'}"></i> ${keep?'保留':'丢弃'} · ${esc(nodePlatform(n))}</span></div>`;deck.appendChild(card);setTimeout(()=>card.remove(),980);validationVisual.processed=Math.max(validationVisual.processed,Number(ev.current||idx+1)||0);validationVisual.keep=Number(ev.keep_count??validationVisual.keep)||0;validationVisual.reject=Number(ev.reject_count??validationVisual.reject)||0;$('keepCount').textContent=validationVisual.keep;$('rejectCount').textContent=validationVisual.reject;$('queueCount').textContent=Math.max(0,(total||validationVisual.input||0)-validationVisual.processed);if(keep){$('keepBucket').classList.add('highlight');setTimeout(()=>$('keepBucket').classList.remove('highlight'),420);if(validationVisual.keep<=160)$('keepGrid').insertAdjacentHTML('beforeend',`<span class="kept-chip" title="${esc(n.title||'')}"><i class="fa-solid fa-circle-check"></i>${esc(nodePlatform(n))}</span>`)}else{$('rejectBucket').classList.add('flash');setTimeout(()=>$('rejectBucket').classList.remove('flash'),420)}}
async function playValidationDecision(n,idx,total,pass,delay,updateCounters=true){const q=showValidationCard(n,idx,total);validationStatusBlock('校验中');await wait(delay);const proc=$('procCard');if(pass){proc&&proc.classList.add('pass');$('keepBucket').classList.add('highlight');if(updateCounters)validationVisual.keep++;if(validationVisual.keep<=160||!updateCounters)$('keepGrid').insertAdjacentHTML('beforeend',`<span class="kept-chip" title="${esc(n.title||'')}"><i class="fa-solid fa-circle-check"></i>${esc(nodePlatform(n))}</span>`);await wait(Math.max(40,delay*.35));$('keepBucket').classList.remove('highlight')}else{proc&&proc.classList.add('fail');$('rejectBucket').classList.add('flash');if(updateCounters)validationVisual.reject++;await wait(Math.max(35,delay*.25));$('rejectBucket').classList.remove('flash')}validationVisual.processed=Math.max(validationVisual.processed,idx+1);if(q)q.classList.add('done');if(updateCounters){$('keepCount').textContent=validationVisual.keep;$('rejectCount').textContent=validationVisual.reject;$('queueCount').textContent=Math.max(0,total-validationVisual.processed)}}
async function animateValidationLive(data){
  if(validationLiveLoop||validationFinalRendered)return;
  validationLiveLoop=true;const token=++validationLiveToken;
  const {inputCount,queueData}=validationQueueFrom(data);
  if(!validationVisual.startedAt||validationVisual.input!==inputCount){resetValidationVisual(inputCount,queueData);renderValidationQueue(queueData,inputCount)}
  $('validateState').textContent='Processing';$('scanLab').classList.add('running');
  const total=inputCount||queueData.length;let shown=0;
  while(validationLiveLoop&&token===validationLiveToken&&!validationFinalRendered&&!validationVisual.finalizing){
    const p=(currentStatus&&currentStatus.phase_data)||{};
    const prog=p.validation_progress||{};
    const realProcessed=Math.max(0,Number(prog.current||prog.processed||shown));
    const events=Array.isArray(prog.recent_events)?prog.recent_events:[];
    const fresh=[];
    for(const ev of events){const key=validationEventKey(ev);if(!validationSeenEvents.has(key)){validationSeenEvents.add(key);fresh.push(ev)}}
    const burst=fresh.slice(0,18);
    if(burst.length){
      burst.forEach((ev,i)=>setTimeout(()=>launchValidationFlight(ev,total),i*34));
      shown=Math.max(shown,realProcessed);
      validationVisual.processed=realProcessed;
      validationVisual.keep=Number(prog.keep_count??validationVisual.keep)||0;
      validationVisual.reject=Number(prog.reject_count??validationVisual.reject)||0;
      validationStatusBlock('实时校验/去重中',realProcessed,total,validationVisual.keep,validationVisual.reject);
      await wait(220);
      continue;
    }
    const visualTarget=total?Math.min(queueData.length,Math.floor(realProcessed/Math.max(1,total)*Math.min(queueData.length,90))):0;
    while(shown<visualTarget&&shown<queueData.length){
      const q=document.querySelector(`.mini-card[data-idx="${shown}"]`);if(q)q.classList.add('done');
      shown++;validationVisual.processed=realProcessed;
      validationStatusBlock('后端校验进行中',realProcessed,total,validationVisual.keep,validationVisual.reject);
      await wait(45);
    }
    validationVisual.processed=realProcessed;validationStatusBlock('后端校验进行中',realProcessed,total,validationVisual.keep,validationVisual.reject);
    if(realProcessed>=total)break;
    await wait(260);
  }
}
async function animateValidation(data,replay=false){
  if(validationFinalRendered&&!replay)return;validationVisual.finalizing=true;validationLiveLoop=false;const {inputCount,queueData}=validationQueueFrom(data);const final=finalNodes(data);const vs=data.summary?.validation_summary||{};const total=Number(vs.input_count??inputCount??queueData.length??0);const finalKeep=Number(vs.deduplicated_count??final.length??vs.validated_count??0);const finalReject=Math.max(0,total-finalKeep);const realKeepNodes=final.length?final:queueData.slice(0,finalKeep);
  const decisionCounts=vs.decision_counts||{};$('validateState').textContent='校验完成';$('scanLab').classList.add('running');validationStatusBlock('校验完成',total,total,finalKeep,finalReject,decisionCounts);
  if(!validationVisual.startedAt||validationVisual.input!==total||!$('queueList').children.length){resetValidationVisual(total,queueData);renderValidationQueue(queueData,total)}
  $('keepGrid').innerHTML='';validationVisual.keep=finalKeep;validationVisual.reject=finalReject;validationVisual.processed=total;$('keepCount').textContent=finalKeep;$('rejectCount').textContent=finalReject;$('queueCount').textContent='0';document.querySelectorAll('.mini-card.done').forEach(x=>x.classList.remove('done'));
  if(!validationSeenEvents.size){
    const finalKeys=new Set(realKeepNodes.map(nodeKey));const sample=[];for(const n of queueData){const keep=finalKeys.has(nodeKey(n));if(keep&&sample.filter(x=>x.keep).length<24)sample.push({node:n,keep:true});else if(!keep&&sample.filter(x=>!x.keep).length<24)sample.push({node:n,keep:false});if(sample.length>=48)break}
    for(let i=0;i<sample.length;i++){await playValidationDecision(sample[i].node,i,total,sample[i].keep,total>300?65:95,false);$('keepCount').textContent=finalKeep;$('rejectCount').textContent=finalReject;$('queueCount').textContent='0'}
  }
  $('scanLab').classList.remove('running');$('currentName').textContent='完成';$('validateState').textContent='完成';$('keepCount').textContent=finalKeep;$('rejectCount').textContent=finalReject;$('queueCount').textContent='0';
  $('keepGrid').innerHTML=realKeepNodes.slice(0,80).map(n=>`<span class="kept-chip" title="${esc(n.title||'')}"><i class="fa-solid fa-circle-check"></i>${esc(nodePlatform(n))}</span>`).join('')+(finalKeep>80?`<span class="kept-chip"><i class="fa-solid fa-database"></i>+${finalKeep-80}</span>`:'');
  $('processingMount').innerHTML='<div class="status-note"><i class="fa-solid fa-check"></i> 校验完成，结果已写入保留池。</div>';validationStatusBlock('校验结果已完成',total,total,finalKeep,finalReject,decisionCounts);validationFinalRendered=true;rendered.validate=true;
}

function renderExtractionSkeleton(data){
  const base=(data&&((data.nodes&&data.nodes.length?data.nodes:data.candidates)||[]))||[];const count=Math.max(8,Math.min(base.length||12,260));$('extractCount').textContent=data.analysis_progress?.total||base.length||'解析中';
  if(extractionLiveRendered)return;extractionLiveRendered=true;
  $('extractList').innerHTML=`<div class="extract-status"><span><i class="fa-solid fa-spinner fa-spin"></i> 网页信息抽取正在进行，完成一个节点就翻开一张牌。</span><div class="extract-progress"><i id="extractProgressBar" style="width:0%"></i></div></div>`+Array.from({length:count},(_,i)=>{const n=base[i]||{};return `<article class="extract-card show" data-idx="${i}"><div class="extract-inner"><div class="extract-face extract-back"><i class="fa-solid fa-file-lines"></i><strong>${esc(nodePlatform(n)||'待解析资源')}</strong><span title="${esc(n.title||n.description||'')}">${esc((n.title||n.description||('metadata pending #'+(i+1))).slice(0,42))}</span></div><div class="extract-face extract-front"><div class="skeleton"><div class="sk w1"></div><div class="sk w2"></div><div class="sk w3"></div><div class="sk w4"></div></div></div></div></article>`}).join('')
}
function updateExtractionProgress(data){
  const nodes=Array.isArray(data.analyzed_nodes)&&data.analyzed_nodes.length?data.analyzed_nodes:(data.nodes||[]);
  if(!nodes.length){renderExtractionSkeleton(data);return}
  if(!extractionLiveRendered)renderExtractionSkeleton({candidates:data.candidates||nodes,analysis_progress:data.analysis_progress||{}});
  $('extractCount').textContent=(data.analysis_progress?.total?`${nodes.length}/${data.analysis_progress.total}`:nodes.length);
  const list=$('extractList');
  nodes.forEach((n,i)=>{
    let card=list.querySelector(`.extract-card[data-idx="${i}"]`);
    if(!card){list.insertAdjacentHTML('beforeend',`<article class="extract-card show" data-idx="${i}"><div class="extract-inner"><div class="extract-face extract-back"><i class="fa-solid fa-file-lines"></i><strong>${esc(nodePlatform(n))}</strong><span>${esc((n.title||n.description||('节点 '+(i+1))).slice(0,42))}</span></div><div class="extract-face extract-front"></div></div></article>`);card=list.querySelector(`.extract-card[data-idx="${i}"]`)}
    const front=card.querySelector('.extract-front');
    if(front&&!card.classList.contains('flipped')){front.innerHTML=nodeCardFront(n,i);setTimeout(()=>{card.classList.add('flipped');if(i%6===0)card.scrollIntoView({block:'nearest'});},80)}
  });
  const total=Number(data.analysis_progress?.total||nodes.length||1);const bar=$('extractProgressBar');if(bar)bar.style.width=Math.min(100,Math.round(nodes.length/Math.max(1,total)*100))+'%';
}
function nodeCardFront(n,i){return `<div class="extract-thumb">${thumbHtml(n)}</div><div class="real-content"><div class="platform ${platformClass(nodePlatform(n))}">${esc(nodePlatform(n))}</div><div class="extract-title" title="${esc(n.title||n.description||'无标题')}">${esc(n.title||n.description||'无标题')}</div><div class="extract-row"><i class="fa-regular fa-user"></i>${esc(n.publisher||n.author||n.metadata_author||'未知')}</div><div class="extract-row"><i class="fa-regular fa-clock"></i>${esc(n.published_at||n.publish_time||'未知')}</div><div class="extract-row"><i class="fa-solid fa-network-wired"></i>${esc(n.propagation_role||n.page_type||'候选节点')}</div><div class="metrics-line"><span><i class="fa-regular fa-eye"></i> ${n.view_count??'-'}</span><span><i class="fa-regular fa-thumbs-up"></i> ${n.like_count??'-'}</span><span><i class="fa-regular fa-comment"></i> ${n.comment_count??'-'}</span><span><i class="fa-solid fa-share"></i> ${n.repost_count??'-'}</span><span>相似度 ${n.similarity!=null?Number(n.similarity).toFixed(3):'-'}</span></div></div>`}
function renderExtraction(data,replay=false){
  if(rendered.extract&&!replay)return;
  const nodes=(data.nodes||[]);$('extractCount').textContent=nodes.length;if(!nodes.length){updateExtractionProgress(data);return}
  updateExtractionProgress({candidates:data.candidates||nodes,analyzed_nodes:nodes,analysis_progress:{current:nodes.length,total:nodes.length}});rendered.extract=true;
}
function stripMd(s){return String(s||'').replace(/[#*_`>-]/g,'').replace(/\s+/g,' ').trim()}
function cleanReportLine(s){return String(s||'').replace(/^[-*]\s*/,'').replace(/\*\*/g,'').trim()}
function parseReportSections(report){
  const raw=String(report||'');const start=raw.indexOf('## 结论')>=0?raw.indexOf('## 结论'):(raw.indexOf('分析洞察')>=0?raw.indexOf('分析洞察'):0);
  const text=raw.slice(start).trim()||raw.trim();const sections=[];let cur={title:'分析摘要',lines:[],scores:[]};
  for(const line of text.split('\n')){
    const trimmed=line.trim();if(!trimmed)continue;
    if(trimmed.startsWith('## ')){if(cur.lines.length||cur.scores.length)sections.push(cur);cur={title:trimmed.replace(/^##\s+/,''),lines:[],scores:[]};continue}
    const score=trimmed.match(/^\*\*([^*]+)\*\*\s*[:：]\s*(\d+)分\s*[-—]\s*(.+)/);
    if(score){cur.scores.push({label:score[1],value:Math.min(100,Math.max(0,parseInt(score[2])||0)),reason:score[3]});continue}
    if(!trimmed.startsWith('#'))cur.lines.push(cleanReportLine(trimmed));
  }
  if(cur.lines.length||cur.scores.length)sections.push(cur);
  return sections.length?sections:[{title:'分析摘要',lines:[stripMd(raw)||'报告为空或尚未生成。'],scores:[]}];
}
function renderReport(data,replay=false){
  if(rendered.report&&!replay)return;
  const overview=data.overview||{},summary=data.summary||{},report=summary.report||summary.final_report||'';
  const sections=parseReportSections(report);const conclusion=sections.find(s=>s.title.includes('结论'))||sections[0];
  const hero=(conclusion.lines||[]).slice(0,2).join(' ')||'报告已生成，建议结合拓扑图与节点表交叉核验。';
  const vs=summary.validation_summary||{},rs=summary.retrieval_summary||{},as=summary.analysis_summary||{};
  const metrics=[['检索候选',rs.result_count??data.candidates?.length??'-'],['通过校验',vs.deduplicated_count??vs.validated_count??data.nodes?.length??'-'],['筛除节点',vs.rejected_count??'-'],['时间证据',as.with_time_count??overview.with_time_count??'-'],['疑似出处',overview.earliest_publisher||'-'],['最早时间',overview.earliest_time||'-']];
  let html='<div class="report-hero"><div class="report-hero-title">溯源结论摘要</div><div class="report-hero-body">'+esc(hero)+'</div></div>';
  html+='<div class="report-stat-grid">'+metrics.map(([k,v])=>'<div class="report-stat"><span>'+esc(k)+'</span><strong>'+esc(String(v).slice(0,32))+'</strong></div>').join('')+'</div>';
  const icons={'结论':'fa-lightbulb','核心发现':'fa-magnifying-glass-chart','多维评估':'fa-chart-simple','建议与对策':'fa-shield-halved','分析摘要':'fa-file-lines'};
  for(const s of sections){
    const icon=icons[s.title]||'fa-circle-info';
    html+='<section class="report-section"><h3><i class="fa-solid '+icon+'" style="color:#2563eb;margin-right:8px"></i>'+esc(s.title)+'</h3>';
    if(s.scores&&s.scores.length){html+='<div class="score-grid">';
      for(const sc of s.scores){const barC=sc.value>=80?'#10b981':(sc.value>=60?'#f59e0b':(sc.value>=40?'#f97316':'#ef4444'));
        html+='<div class="score-item"><div class="score-label"><span>'+esc(sc.label)+'</span><strong style="color:'+barC+'">'+sc.value+'分</strong></div><div class="score-track"><i style="width:'+sc.value+'%;background:'+barC+'"></i></div><div class="score-reason">'+esc(sc.reason)+'</div></div>';}
      html+='</div>';
    }
    const lines=(s.lines||[]).filter(Boolean);
    if(lines.length<=2){for(const line of lines)html+='<p>'+esc(line)+'</p>';}
    else{html+='<ul class="report-list">'+lines.slice(0,8).map(line=>'<li>'+esc(line)+'</li>').join('')+'</ul>';}
    html+='</section>';
  }
  $('reportBox').innerHTML=html;
  rendered.report=true;
}
function drawNetwork(data,replay=false){
  if(rendered.network&&!replay)return;
  const container=$('network');container.innerHTML='';
  const topo=data.summary?.topology_data||data.state_dump?.topology_data||{};
  const tnodes=Array.isArray(topo.nodes)?topo.nodes:[];
  const tedges=Array.isArray(topo.edges)?topo.edges:[];
  $('topologyCount').textContent=tnodes.length+' / '+tedges.length;
  const reportDir=(currentStatus&&currentStatus.report_dir)||(activeReport?activeReport.report_dir:'')||'';
  const topoUrl=reportDir?('/report_files/'+reportDir+'/topology.html'):'';
  if(topoUrl){
    container.innerHTML='<a class="topology-open" href="'+topoUrl+'" target="_blank" rel="noreferrer"><i class="fa-solid fa-up-right-and-down-left-from-center"></i> 全屏拓扑交互</a><iframe src="'+topoUrl+'" style="width:100%;height:100%;min-height:580px;border:0;border-radius:12px;background:#0f172a;"></iframe>';
  }else if(tnodes.length){
    container.innerHTML='<div class="status-note" style="padding:18px">Topology: '+tnodes.length+' nodes, '+tedges.length+' edges. Open output/report_*/topology.html for G6 view.</div>';
  }else{
    container.innerHTML='<div class="status-note" style="padding:22px">No topology data. Check analyzer output.</div>';
  }
  rendered.network=true;
}
function renderDataTabs(data){
  const nodes=data.nodes||[];
  const groups=new Map();
  nodes.forEach(n=>{const plat=nodePlatform(n)||'other';if(!groups.has(plat))groups.set(plat,[]);groups.get(plat).push(n)});
  const order=['微博','小红书','百度','Yandex','Google','TinEye','Bing'];
  const sorted=[...groups.entries()].sort((a,b)=>{const ai=order.indexOf(a[0]),bi=order.indexOf(b[0]);return(ai===-1?99:ai)-(bi===-1?99:bi)});
  let html='';
  for(const [plat,items] of sorted){
    html+='<div style="margin-bottom:16px"><div style="font-size:.78rem;font-weight:700;color:#38bdf8;margin-bottom:6px;display:flex;align-items:center;gap:8px"><span class="platform '+platformClass(plat)+'" style="font-size:.68rem">'+esc(plat)+'</span><span style="color:var(--muted);font-weight:400">'+items.length+' nodes</span></div><table><thead><tr><th>#</th><th>Title</th><th>Author</th><th>Time</th><th>Role</th><th>Similarity</th></tr></thead><tbody>';
    items.forEach((n,i)=>{html+='<tr><td>'+(i+1)+'</td><td title="'+esc(n.title||'')+'" style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+esc((n.title||'?').slice(0,55))+'</td><td>'+esc((n.publisher||n.author||'?').slice(0,14))+'</td><td style="font-size:.7rem">'+esc((n.published_at||'?').slice(0,16))+'</td><td style="font-size:.7rem">'+esc((n.propagation_role||'-').slice(0,10))+'</td><td style="font-family:monospace;font-size:.7rem">'+(n.similarity!=null?Number(n.similarity).toFixed(2):'-')+'</td></tr>';});
    html+='</tbody></table></div>';
  }
  if(!sorted.length)html='<div class="status-note">暂无数据</div>';
  $('tab-table').innerHTML=html;
  $('terminal日志').textContent=data.logs||data.full_logs||'No logs';
  $('rawJson').textContent=JSON.stringify({overview:data.overview,summary:data.summary,candidates:data.candidates,nodes:data.nodes,state_dump:data.state_dump},null,2);
  $('downloadJson').href=data.downloads?.nodes||'#';$('downloadTxt').href=data.downloads?.logs||'#';$('downloadReport').href=data.downloads?.summary||'#';
  rendered.data=true;
}
function renderAll(data){activeReport=data;reportRendered=true;Object.assign(rendered,{search:false,validate:false,extract:false,report:false,network:false,data:false});renderSearchResults(data,false);renderDataTabs(data);if(currentStep===2)animateValidation(data,true);if(currentStep===3)renderExtraction(data,true);if(currentStep===4)renderReport(data,true);if(currentStep===5)drawNetwork(data,true)}
async function loadReport(name){if(!name)return;try{const res=await fetch('/api/report/'+encodeURIComponent(name));if(!res.ok)throw new Error('report not found');const data=await res.json();renderAll(data)}catch(e){$('statusChip').textContent='读取报告失败：'+e.message}}
function livePayloadFromStatus(s){const p=s.phase_data||{};const summary={retrieval_summary:p.retrieval_summary||{},validation_summary:p.validation_summary||{},analysis_summary:p.analysis_summary||{},topology_data:p.topology_data||{},report:p.final_report||''};return {summary:summary,candidates:Array.isArray(p.candidates)?p.candidates:[],nodes:Array.isArray(p.nodes)?p.nodes:[],analyzed_nodes:Array.isArray(p.analyzed_nodes)?p.analyzed_nodes:[],overview:p.overview||{},validation_started_at:p.validation_started_at||0,analysis_started_at:p.analysis_started_at||0,validation_progress:p.validation_progress||{},analysis_progress:p.analysis_progress||{},logs:(s.recent_logs||[]).join('\n'),full_logs:(s.recent_logs||[]).join('\n'),downloads:{}}}

function renderLiveFromStatus(s){
  if(reportRendered)return;const data=livePayloadFromStatus(s);activeReport=data;
  const hasAnalysis=!!(data.summary.analysis_summary&&Object.keys(data.summary.analysis_summary).length)||(s.step==='report')||(s.status==='done');
  if(data.candidates.length&&!rendered.search)renderSearchResults(data,false);
  if(s.step==='validate'&&data.candidates.length&&!validationFinalRendered){if(data.validation_started_at&&!validationVisual.startedAt)validationVisual.startedAt=data.validation_started_at*1000;animateValidationLive(data)}
  if(validationHasFinalSummary(data)&&!validationFinalRendered)animateValidation(data,true);
  if(s.step==='analyze'){if(!analyzeStartedAt)analyzeStartedAt=(data.analysis_started_at?data.analysis_started_at*1000:Date.now());if(!extractionLiveRendered)renderExtractionSkeleton(data);if(data.analyzed_nodes.length)updateExtractionProgress(data)}
  if(data.nodes.length&&hasAnalysis&&!rendered.extract)renderExtraction(data,true);
  if((data.summary.report||'').trim()&&!rendered.report)renderReport(data,true);
  if((data.summary.topology_data?.nodes||[]).length&&!rendered.network&&currentStep===5)drawNetwork(data,true);
  if(currentStep===6)renderDataTabs(data)
}
function startPolling(){if(!pollTimer)pollTimer=setInterval(()=>pollStatus(false),1000)}function stopPolling(){if(pollTimer){clearInterval(pollTimer);pollTimer=null}}
async function pollStatus(force=false){if(!hasUploadedThisPage&&!activeJobId)return;try{const res=await fetch('/api/status');const s=await res.json();if(activeJobId&&s.job_id&&s.job_id!==activeJobId)return;currentStatus=s;$('statusChip').textContent=s.message||s.status||'运行中';const logs=s.recent_logs||[];if(logs.length&&!reportRendered){$('terminal日志').textContent=logs.join('\n')}renderLiveFromStatus(s);let step=s.ui_step||0;if(s.status==='error'){$('statusChip').textContent=s.message||'流程失败';autoStep(step);stopPolling();return}if(s.status==='done'&&s.report_dir){if(!reportRendered)await loadReport(s.report_dir);autoStep(4);setTimeout(()=>autoStep(5),5200);setTimeout(()=>autoStep(6),10800);stopPolling();return}autoStep(step)}catch(e){$('statusChip').textContent='状态连接失败：'+e.message}}
$('loadLatestBtn').addEventListener('click',async()=>{try{const res=await fetch('/api/latest');const d=await res.json();if(!res.ok)throw new Error(d.error||'没有报告');renderAll(d);manualPauseUntil=0;autoStep(4)}catch(e){$('statusChip').textContent=e.message}});$('testBtn').addEventListener('click',async()=>{try{const res=await fetch('/api/import-test');const d=await res.json();if(!res.ok)throw new Error(d.error||'没有测试报告');renderAll(d);manualPauseUntil=0;autoStep(4)}catch(e){$('statusChip').textContent=e.message}});
document.querySelectorAll('.tab').forEach(btn=>btn.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));btn.classList.add('active');$('tab-'+btn.dataset.tab).classList.add('active')}));setStep(0,'auto');
</script>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
async def index_html() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


if __name__ in {"__main__", "__mp_main__"}:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    globals()["CURRENT_JOB_ID"] = ""
    write_status(status="idle", step="idle", message="等待上传图片", progress=0.0, report_dir=None)
    app.add_static_files("/report_files", OUTPUT_DIR.as_posix())
    ui.run(
        title="图片溯源智能体",
        favicon="🔍",
        host=os.getenv("NICEGUI_HOST", "127.0.0.1"),
        port=int(os.getenv("NICEGUI_PORT", "8502")),
        reload=False,
    )
