"""
G6 拓扑图可视化模块 — 从 AgentState 生成 G6 节点/边数据并渲染为 HTML。

该模块从 test.py 提取而来，供 app.py（Streamlit）和 test.py（CLI）共用。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from core.state import AgentState

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


# ═══════════════════════════════════════════════════════════════════════
# 平台 / 分类标签常量
# ═══════════════════════════════════════════════════════════════════════

PLATFORM_LABELS = {
    "weibo": "微博",
    "xiaohongshu": "小红书",
    "other": "其他平台",
}

EXTERNAL_FAMILY_LABELS = {
    "news": "新闻",
    "forum": "论坛",
    "blog": "博客",
    "media": "媒体",
    "baidu_media": "百度系媒体",
    "video": "视频",
    "social": "社交平台",
    "generic": "通用网页",
    "unknown": "未知来源",
}


# ═══════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════

def short_text(value: Any, limit: int = 32) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else f"{text[:limit - 1]}..."


def numeric_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def metric_value(node: Dict[str, Any], key: str) -> int | None:
    metrics = node.get("metrics") if isinstance(node.get("metrics"), dict) else {}
    value = metrics.get(key) if isinstance(metrics, dict) else node.get(key)
    if value in (None, "", "null"):
        return None
    try:
        return int(float(str(value or 0)))
    except ValueError:
        return None


def metric_label(value: Any) -> str:
    return "无" if value in (None, "", "null") else str(value)


def coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def external_url(value: Any) -> str:
    url = str(value or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith(("http://", "https://")):
        return url
    return ""


def platform_label(platform: Any) -> str:
    return PLATFORM_LABELS.get(str(platform or "other"), str(platform or "other"))


def external_family_label(family: Any) -> str:
    key = str(family or "unknown").strip() or "unknown"
    return EXTERNAL_FAMILY_LABELS.get(key, key)


def is_mainstream_platform(node: Dict[str, Any]) -> bool:
    return str(node.get("platform") or "other") in {"weibo", "xiaohongshu"}


def external_node_allowed(node: Dict[str, Any]) -> bool:
    return is_mainstream_platform(node) or bool(node.get("allow_in_external_timeline"))


def lane_key_for_node(node: Dict[str, Any]) -> str:
    platform = str(node.get("platform") or "other")
    if platform in {"weibo", "xiaohongshu"}:
        return platform
    family = str(node.get("platform_family") or "unknown").strip() or "unknown"
    return f"external:{family}"


def lane_label(lane_key: Any) -> str:
    key = str(lane_key or "other")
    if key.startswith("external:"):
        return f"其他平台 / {external_family_label(key.split(':', 1)[1])}"
    return platform_label(key)


def parse_publish_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    formats = (
        ("%Y-%m-%d %H:%M:%S", 19),
        ("%Y-%m-%d %H:%M", 16),
        ("%Y-%m-%d", 10),
    )
    for fmt, length in formats:
        try:
            return datetime.strptime(text[:length], fmt)
        except ValueError:
            continue
    return None


def display_publish_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "时间未知"
    parsed = parse_publish_time(text)
    if parsed:
        return parsed.strftime("%Y-%m-%d %H:%M")
    return text[:16] if len(text) > 16 else text


def fetch_image_as_data_url(url: str, headers: Dict[str, str]) -> str:
    if requests is None:
        return ""
    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
    except Exception:
        return ""
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if not content_type.startswith("image/"):
        content_type = mimetypes.guess_type(url)[0] or "image/jpeg"
    if len(response.content) > 5 * 1024 * 1024:
        return ""
    encoded = base64.b64encode(response.content).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def display_image_url(value: Any) -> str:
    url = str(value or "").strip()
    if not url:
        return ""
    if re.search(r".*\.sinaimg\.cn.*", url, flags=re.I):
        embedded = fetch_image_as_data_url(url, {"Referer": "https://weibo.com/"})
        if embedded:
            return embedded
    return url


def page_url_for_node(node: Dict[str, Any]) -> str:
    raw_url = external_url(node.get("url"))
    canonical_url = external_url(node.get("canonical_url"))
    if not canonical_url:
        return raw_url
    if "baijiahao.baidu.com/s" in canonical_url and "id=" not in canonical_url and raw_url:
        return raw_url
    if raw_url and "id=" in raw_url and "id=" not in canonical_url:
        return raw_url
    return canonical_url


def first_image_url(node: Dict[str, Any]) -> str:
    for key in ("thumbnail_url", "image_url"):
        url = str(node.get(key) or "").strip()
        if url:
            return url
    for item in coerce_list(node.get("image_urls")):
        url = str(item or "").strip()
        if url:
            return url
    normalized = node.get("normalized_record") if isinstance(node.get("normalized_record"), dict) else {}
    post = normalized.get("post") if isinstance(normalized.get("post"), dict) else {}
    for item in coerce_list(post.get("image_urls")):
        url = str(item or "").strip()
        if url:
            return url
    return ""


def field_evidence_summary(value: Any, limit: int = 4) -> List[Dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for field_name, evidence in value.items():
        if not isinstance(evidence, dict):
            continue
        rows.append(
            {
                "field": str(field_name),
                "status": str(evidence.get("status") or ""),
                "source": str(evidence.get("source") or ""),
                "confidence": numeric_value(evidence.get("confidence")),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def duplicate_center_id(node: Dict[str, Any]) -> str:
    duplicate = node.get("duplicate_analysis") if isinstance(node.get("duplicate_analysis"), dict) else {}
    return str(duplicate.get("center_node_id") or "")


def is_folded_duplicate_child(node: Dict[str, Any]) -> bool:
    node_id = str(node.get("id") or "")
    duplicate = node.get("duplicate_analysis") if isinstance(node.get("duplicate_analysis"), dict) else {}
    center_id = duplicate_center_id(node)
    member_ids = [str(item) for item in coerce_list(duplicate.get("cluster_member_ids")) if str(item)]
    return bool(len(member_ids) > 1 and center_id and node_id and center_id != node_id)


# ═══════════════════════════════════════════════════════════════════════
# G6 拓扑数据构建
# ═══════════════════════════════════════════════════════════════════════

def build_g6_payload(result: AgentState) -> Dict[str, Any]:
    nodes_data = result.get("nodes_data", [])
    topology_data = result.get("topology_data", {})
    topo_nodes = topology_data.get("nodes", []) if isinstance(topology_data, dict) else []
    topo_edges = topology_data.get("edges", []) if isinstance(topology_data, dict) else []
    platform_timelines = topology_data.get("platform_timelines", {}) if isinstance(topology_data, dict) else {}
    external_timelines = topology_data.get("external_timelines", {}) if isinstance(topology_data, dict) else {}

    runtime_by_id = {str(node.get("id") or ""): node for node in nodes_data if isinstance(node, dict)}
    graph_nodes: List[Dict[str, Any]] = []
    if isinstance(topo_nodes, list) and topo_nodes:
        for topo_node in topo_nodes:
            if not isinstance(topo_node, dict):
                continue
            node_id = str(topo_node.get("id") or "")
            graph_nodes.append({**runtime_by_id.get(node_id, {}), **topo_node})
    else:
        graph_nodes = [node for node in nodes_data if isinstance(node, dict)]

    graph_node_ids = {str(node.get("id") or "") for node in graph_nodes}
    graph_node_by_id = {str(node.get("id") or ""): node for node in graph_nodes}
    graph_edges = [
        edge
        for edge in topo_edges
        if isinstance(edge, dict)
        and str(edge.get("source") or edge.get("from") or "") in graph_node_ids
        and str(edge.get("target") or edge.get("to") or "") in graph_node_ids
        and (
            str(edge.get("edge_type") or "") == "duplicate_cluster"
            or (
                external_node_allowed(graph_node_by_id[str(edge.get("source") or edge.get("from") or "")])
                and external_node_allowed(graph_node_by_id[str(edge.get("target") or edge.get("to") or "")])
            )
        )
        and str(edge.get("edge_type") or "") not in {"text_source_mention", "time_image_text_inferred"}
        and str(edge.get("method") or "") not in {"text", "time+image+text", "time+image+text+platform"}
    ]

    seen_lanes = {lane_key_for_node(node) for node in graph_nodes}
    preferred_order = ["weibo", "xiaohongshu"]
    platform_order = [item for item in preferred_order if item in seen_lanes]
    platform_order.extend(
        f"external:{family}"
        for family in sorted(external_timelines)
        if f"external:{family}" in seen_lanes and f"external:{family}" not in platform_order
    )
    platform_order.extend(sorted(seen_lanes - set(platform_order)))
    if isinstance(platform_timelines, dict):
        platform_order = [
            platform for platform in platform_order
            if platform in platform_timelines
            or platform.startswith("external:")
            or any(lane_key_for_node(node) == platform for node in graph_nodes)
        ]

    lane_gap = 220
    x_gap = 230
    x_start = 300
    y_start = 280
    max_x = x_start
    positioned_nodes: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}

    lane_y_by_platform = {}
    platform_counts = {platform: 0 for platform in platform_order}
    for node in graph_nodes:
        key = lane_key_for_node(node)
        platform_counts[key] = platform_counts.get(key, 0) + 1
    for lane_index, platform in enumerate(platform_order):
        lane_y = y_start + lane_index * lane_gap
        lane_y_by_platform[platform] = lane_y
        lanes = [{"platform": platform, "label": lane_label(platform), "y": lane_y, "count": platform_counts.get(platform, 0)}]

    # rebuild lanes list properly
    lanes = []
    for lane_index, platform in enumerate(platform_order):
        lane_y = y_start + lane_index * lane_gap
        lane_y_by_platform[platform] = lane_y
        lanes.append({"platform": platform, "label": lane_label(platform), "y": lane_y, "count": platform_counts.get(platform, 0)})

    main_nodes = [node for node in graph_nodes if not is_folded_duplicate_child(node)]
    main_nodes.sort(
        key=lambda item: (
            parse_publish_time(item.get("publish_time") or item.get("published_at")) or datetime.max,
            int(item.get("input_order") or 0),
        )
    )
    main_position_by_id: Dict[str, tuple[float, float]] = {}
    for order, node in enumerate(main_nodes):
        platform = lane_key_for_node(node)
        x = x_start + order * x_gap
        y = lane_y_by_platform.get(platform, y_start)
        main_position_by_id[str(node.get("id") or "")] = (x, y)
        max_x = max(max_x, x)

    duplicate_children_by_center: Dict[str, List[Dict[str, Any]]] = {}
    for node in graph_nodes:
        if is_folded_duplicate_child(node):
            duplicate_children_by_center.setdefault(duplicate_center_id(node), []).append(node)

    duplicate_offsets = [
        (-145, -120), (0, -170), (145, -120),
        (-165, 0), (165, 0),
        (-145, 120), (0, 170), (145, 120),
    ]
    node_render_order = [*main_nodes]
    for center_id, children in duplicate_children_by_center.items():
        center_pos = main_position_by_id.get(center_id)
        if not center_pos:
            continue
        children.sort(key=lambda item: int(item.get("input_order") or 0))
        for child_index, child in enumerate(children):
            offset = duplicate_offsets[child_index % len(duplicate_offsets)]
            ring = child_index // len(duplicate_offsets)
            child["_folded_x"] = center_pos[0] + offset[0] + ring * (32 if offset[0] >= 0 else -32)
            child["_folded_y"] = center_pos[1] + offset[1] + ring * (26 if offset[1] >= 0 else -26)
            node_render_order.append(child)

    for node in node_render_order:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        platform = str(node.get("platform") or "other")
        lane_key = lane_key_for_node(node)
        external_allowed = external_node_allowed(node)
        platform_family = str(node.get("platform_family") or "unknown")
        evidence_status = str(node.get("evidence_node_status") or "")
        page_type = str(node.get("page_type") or "")
        tamper = node.get("tamper_analysis") if isinstance(node.get("tamper_analysis"), dict) else {}
        validator = node.get("validator") if isinstance(node.get("validator"), dict) else {}
        duplicate = node.get("duplicate_analysis") if isinstance(node.get("duplicate_analysis"), dict) else {}
        influence = node.get("influence_analysis") if isinstance(node.get("influence_analysis"), dict) else {}
        provenance = node.get("provenance") if isinstance(node.get("provenance"), dict) else {}
        image_occurrence = node.get("image_occurrence") if isinstance(node.get("image_occurrence"), dict) else {}
        node_decision = node.get("node_decision") if isinstance(node.get("node_decision"), dict) else {}
        source_score = numeric_value(node.get("source_score"))
        node_weight = numeric_value(node.get("node_weight"))
        influence_score = numeric_value(influence.get("influence_score"))
        folded_child = is_folded_duplicate_child(node)
        size = 30 if not external_allowed else (34 if folded_child else max(42, min(82, 42 + node_weight * 26 + influence_score * 16)))

        if not external_allowed:
            fill, stroke = "#eef2f6", "#98a2b3"
        elif folded_child:
            fill, stroke = "#fff7ed", "#ff4d64"
        elif node.get("is_suspected_source"):
            fill, stroke = "#dff7ea", "#1f9d68"
        elif tamper.get("is_tampered") or node.get("is_tampered"):
            fill, stroke = "#fff0e8", "#d95f35"
        elif node.get("is_key_node"):
            fill, stroke = "#e7f1ff", "#1d5f99"
        elif duplicate.get("is_possible_duplicate"):
            fill, stroke = "#fff9db", "#b7791f"
        else:
            fill, stroke = "#f8fafc", "#5b7c99"

        badges = node.get("badges") if isinstance(node.get("badges"), list) else []
        macro_badges = [str(item) for item in badges if item in {"疑似源头", "关键节点", "大V/认证", "疑似篡改", "疑似重复"}]
        duplicate_role = str(duplicate.get("cluster_role") or "")
        if duplicate_role == "center" and len(coerce_list(duplicate.get("cluster_member_ids"))) > 1:
            macro_badges.append("重复簇中心")
        if folded_child and "疑似重复" not in macro_badges:
            macro_badges.append("疑似重复")
        badge_text = " / ".join(macro_badges[:3]) or "普通节点"
        publish_time = str(node.get("publish_time") or node.get("published_at") or "")
        time_text = display_publish_time(publish_time)
        repost = metric_value(node, "repost_count")
        like = metric_value(node, "like_count")
        comment = metric_value(node, "comment_count")
        x, y = (
            (float(node.get("_folded_x")), float(node.get("_folded_y")))
            if folded_child
            else main_position_by_id.get(node_id, (x_start, lane_y_by_platform.get(lane_key, y_start)))
        )
        max_x = max(max_x, x)
        raw_image_url = first_image_url(node)
        image_url = display_image_url(raw_image_url)
        author_info = {
            "name": node.get("publisher") or node.get("author") or "",
            "followers": node.get("follower_count"),
            "following": node.get("following_count"),
            "is_big_v": bool(node.get("is_big_v")),
        }
        reason = node.get("reason") or validator.get("reason") or validator.get("validation_reason") or tamper.get("reason") or ""
        data = {
            "id": node_id,
            "platform": lane_label(lane_key),
            "publish_time": publish_time,
            "badges": macro_badges,
            "repost_count": repost,
            "like_count": like,
            "comment_count": comment,
            "image_url": image_url,
            "raw_image_url": raw_image_url,
            "url": page_url_for_node(node),
            "similarity": numeric_value(node.get("similarity") if node.get("similarity") is not None else validator.get("similarity")),
            "reason": reason,
            "author": author_info,
            "source_score": round(source_score, 2),
            "node_weight": round(node_weight, 2),
            "duplicate": duplicate,
            "is_folded_duplicate": folded_child,
            "title": short_text(node.get("title"), 80),
            "platform_family": external_family_label(platform_family),
            "page_type": page_type,
            "evidence_node_status": evidence_status,
            "allow_in_external_timeline": external_allowed,
            "allow_cross_platform_relation_candidate": bool(node.get("allow_cross_platform_relation_candidate")),
            "node_decision_reason": node.get("node_decision_reason") or node_decision.get("reason") or "",
            "image_occurrence": image_occurrence,
            "provenance": provenance,
            "source_url": external_url(node.get("source_url") or provenance.get("source_url")),
            "source_text": short_text(node.get("source_text") or provenance.get("source_text"), 180),
            "source_platform_hint": node.get("source_platform_hint") or provenance.get("source_platform_hint") or "",
            "source_account_hint": node.get("source_account_hint") or provenance.get("source_account_hint") or "",
            "field_evidence": field_evidence_summary(node.get("field_evidence")),
        }
        positioned_nodes.append(
            {
                "id": node_id,
                "x": x,
                "y": y,
                "label": (
                    f"疑似重复\n{time_text}"
                    if folded_child
                    else f"#{len([item for item in main_nodes if str(item.get('id') or '') in main_position_by_id and main_position_by_id[str(item.get('id') or '')][0] <= x])} {lane_label(lane_key)} | {badge_text}\n{time_text}  转{metric_label(repost)}  赞{metric_label(like)}  评{metric_label(comment)}"
                ),
                "size": round(size, 1),
                "style": {
                    "fill": fill,
                    "stroke": stroke,
                    "lineWidth": 3.0 if folded_child or node.get("is_suspected_source") else 1.8,
                    "opacity": 0.72 if not external_allowed else 1.0,
                },
                "labelCfg": {"position": "bottom", "offset": 9, "style": {"fill": "#172033", "fontSize": 11, "lineHeight": 16}},
                "data": data,
            }
        )
        details[node_id] = data

    edge_style_by_type = {
        "REPOST": {"stroke": "#1f9d68", "lineWidth": 3.2, "lineDash": None, "directed": True},
        "explicit_repost": {"stroke": "#1f9d68", "lineWidth": 3.2, "lineDash": None, "directed": True},
        "explicit_source_url": {"stroke": "#1f9d68", "lineWidth": 3.0, "lineDash": None, "directed": True},
        "CROSS_PLATFORM": {"stroke": "#cf4f29", "lineWidth": 2.7, "lineDash": [8, 4], "directed": False},
        "cross_platform_watermark": {"stroke": "#cf4f29", "lineWidth": 2.7, "lineDash": [8, 4], "directed": False},
        "duplicate_cluster": {"stroke": "#ff4d64", "lineWidth": 2.6, "lineDash": [6, 6], "directed": False},
        "watermark_account_match": {"stroke": "#b87912", "lineWidth": 2.4, "lineDash": [4, 4], "directed": False},
        "ocr_account_match": {"stroke": "#946322", "lineWidth": 2.4, "lineDash": [6, 3], "directed": False},
    }
    default_edge_style = {"stroke": "#94a3b8", "lineWidth": 1.6, "lineDash": [4, 4], "directed": False}
    g6_edges: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    for index, edge in enumerate(graph_edges):
        source = str(edge.get("source") or edge.get("from") or "")
        target = str(edge.get("target") or edge.get("to") or "")
        edge_type = str(edge.get("edge_type") or "inferred")
        style = edge_style_by_type.get(edge_type, default_edge_style)
        evidence = edge.get("evidence") if isinstance(edge.get("evidence"), list) else []
        row = {
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "confidence": numeric_value(edge.get("confidence", edge.get("edge_weight"))),
            "method": edge.get("method") or edge_type,
            "evidence": "; ".join(str(item) for item in evidence[:4]) if evidence else "no evidence",
            "directed": bool(style.get("directed")),
        }
        g6_edges.append(
            {
                "id": f"edge-{index}",
                "source": source,
                "target": target,
                "label": "",
                "type": "line" if edge_type == "duplicate_cluster" else "cubic-horizontal",
                "style": {
                    "stroke": style["stroke"],
                    "lineWidth": style["lineWidth"],
                    "lineDash": style["lineDash"],
                    "endArrow": True if style.get("directed") else False,
                },
                "data": row,
            }
        )
        edge_rows.append(row)

    canvas_width = max(1180, max_x + 460)
    canvas_height = max(620, y_start + max(len(platform_order), 1) * lane_gap + 80)
    return {
        "nodes": positioned_nodes,
        "edges": g6_edges,
        "details": details,
        "edgeRows": edge_rows,
        "lanes": lanes,
        "canvasWidth": canvas_width,
        "canvasHeight": canvas_height,
        "runtime": topology_data.get("runtime") if isinstance(topology_data, dict) else {},
    }


# ═══════════════════════════════════════════════════════════════════════
# HTML 模板 & 输出
# ═══════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Analyzer G6 Topology</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: #172033;
      background: #f4f7fb;
    }
    .graph-shell {
      position: relative;
      min-height: __HEIGHT__px;
      margin: 16px;
      padding: 16px;
      border: 1px solid #d9e1ea;
      border-radius: 8px;
      background: #ffffff;
      box-sizing: border-box;
    }
    .graph-toolbar {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 12px;
    }
    .graph-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      margin-top: 10px;
    }
    .graph-action-btn {
      border: 1px solid #2563eb;
      border-radius: 999px;
      background: #2563eb;
      color: #ffffff;
      cursor: pointer;
      font-size: 13px;
      font-weight: 700;
      padding: 9px 13px;
      box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
    }
    .graph-action-btn.secondary {
      border-color: #cbd5e1;
      background: #ffffff;
      color: #172033;
      box-shadow: none;
    }
    h3 { margin: 0 0 4px; font-size: 18px; line-height: 1.3; }
    p { margin: 0; color: #667085; font-size: 13px; }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
      font-size: 12px;
      color: #344054;
    }
    .legend span { display: inline-flex; align-items: center; gap: 5px; }
    .legend i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .source { background: #1f9d68; }
    .risk { background: #d95f35; }
    .key { background: #1d5f99; }
    .duplicate { background: #b7791f; }
    #g6-topology {
      height: calc(__HEIGHT__px - 230px);
      min-height: 560px;
      border: 1px solid #edf1f5;
      border-radius: 8px;
      background: #fbfdff;
    }
    .detail-card {
      position: absolute;
      right: 28px;
      top: 274px;
      width: 330px;
      max-height: 430px;
      overflow: auto;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 14px 28px rgba(17, 24, 39, 0.10);
      padding: 12px;
      box-sizing: border-box;
      pointer-events: auto;
    }
    .detail-card.is-hidden { display: none; }
    .detail-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .detail-close, .detail-open {
      border: 1px solid #d8dee8;
      border-radius: 6px;
      background: #ffffff;
      color: #344054;
      cursor: pointer;
      font-size: 12px;
      line-height: 1;
    }
    .detail-close { width: 24px; height: 24px; }
    .detail-open {
      position: absolute;
      right: 28px;
      top: 274px;
      padding: 8px 10px;
      display: none;
      box-shadow: 0 8px 18px rgba(17, 24, 39, 0.10);
    }
    .detail-open.is-visible { display: block; }
    .legend-card {
      position: absolute;
      right: 28px;
      top: 92px;
      width: 330px;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 10px 22px rgba(17, 24, 39, 0.08);
      padding: 12px;
      box-sizing: border-box;
      font-size: 12px;
      color: #344054;
      pointer-events: auto;
    }
    .legend-title { font-weight: 700; margin-bottom: 8px; color: #172033; }
    .sample-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px 10px; }
    .node-sample, .edge-sample { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
    .node-dot { width: 13px; height: 13px; border-radius: 50%; border: 2px solid #5b7c99; background: #f8fafc; display: inline-block; }
    .dot-source { background: #dff7ea; border-color: #1f9d68; }
    .dot-key { background: #e7f1ff; border-color: #1d5f99; }
    .dot-risk { background: #fff0e8; border-color: #d95f35; }
    .dot-dup { background: #fff7ed; border-color: #ff4d64; }
    .dot-muted { background: #eef2f6; border-color: #98a2b3; }
    .edge-line { width: 24px; height: 0; border-top: 2px solid #64748b; display: inline-block; }
    .edge-dash { border-top-style: dashed; }
    .edge-relation { border-color: #ff4d64; }
    .detail-title { font-weight: 700; font-size: 15px; line-height: 1.35; margin-bottom: 8px; }
    .detail-line { color: #344054; font-size: 12px; line-height: 1.45; margin-top: 5px; word-break: break-word; }
    .detail-muted { color: #667085; font-size: 12px; line-height: 1.45; }
    .detail-img {
      width: 100%;
      max-height: 160px;
      object-fit: contain;
      border-radius: 6px;
      border: 1px solid #e2e8f0;
      margin-bottom: 8px;
      background: #f4f7fb;
    }
    .detail-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 9px; }
    .detail-tags span {
      display: inline-flex;
      border: 1px solid #d8dee8;
      border-radius: 999px;
      padding: 2px 7px;
      background: #f7fafc;
      color: #344054;
      font-size: 11px;
    }
    .edge-panel { margin-top: 12px; border-top: 1px solid #edf1f5; padding-top: 10px; }
    .edge-panel h4 { margin: 0 0 8px; font-size: 14px; }
    .edge-row {
      display: inline-flex;
      margin: 0 8px 8px 0;
      border: 1px solid #d8dee8;
      border-radius: 999px;
      padding: 4px 9px;
      color: #344054;
      background: #fbfdff;
      font-size: 12px;
    }
    .g6-tooltip {
      border: 1px solid #d8dee8;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.98);
      box-shadow: 0 12px 24px rgba(17, 24, 39, 0.14);
      padding: 10px;
      color: #172033;
      max-width: 290px;
      font-size: 12px;
      line-height: 1.45;
    }
    .g6-tooltip img { width: 100%; max-height: 160px; object-fit: contain; border-radius: 6px; margin-bottom: 6px; background: #f4f7fb; }
  </style>
</head>
<body>
  <main class="graph-shell">
    <div class="graph-toolbar">
      <div>
        <h3>传播拓扑图</h3>
        <p>按平台泳道和发布时间展开；节点可拖拽，hover/click 查看缩略图、相似度、reason 和作者信息。</p>
      </div>
      <div class="legend">
        <span><i class="source"></i>疑似源头</span>
        <span><i class="risk"></i>疑似篡改</span>
        <span><i class="key"></i>关键节点</span>
        <span><i class="duplicate"></i>疑似重复簇</span>
        <span><i style="background:#98a2b3"></i>外部旁证</span>
      </div>
    </div>
    <div class="graph-actions">
      <button id="fullscreen-btn" class="graph-action-btn" type="button">全屏放大拓扑</button>
      <button id="fit-btn" class="graph-action-btn secondary" type="button">适配视图</button>
    </div>
    <div id="g6-topology"></div>
    <aside class="legend-card">
      <div class="legend-title">节点与边样式</div>
      <div class="sample-grid">
        <span class="node-sample"><i class="node-dot dot-source"></i>疑似源头</span>
        <span class="node-sample"><i class="node-dot dot-key"></i>关键节点</span>
        <span class="node-sample"><i class="node-dot dot-risk"></i>疑似篡改</span>
        <span class="node-sample"><i class="node-dot dot-dup"></i>疑似重复</span>
        <span class="node-sample"><i class="node-dot dot-muted"></i>外部旁证</span>
        <span class="edge-sample"><i class="edge-line edge-relation edge-dash"></i>重复簇关系</span>
        <span class="edge-sample"><i class="edge-line edge-dash"></i>其他弱关系</span>
      </div>
    </aside>
    <aside id="node-detail" class="detail-card">
      <div class="detail-header">
        <div class="detail-title">选择一个节点</div>
        <button id="detail-close" class="detail-close" type="button">×</button>
      </div>
      <div class="detail-muted">详情仅展示图片相似度、reason 和作者信息。</div>
    </aside>
    <button id="detail-open" class="detail-open" type="button">显示详情</button>
    <section class="edge-panel">
      <h4>帖子 → 帖子边</h4>
      <div id="edge-list"></div>
    </section>
  </main>
  <script src="https://gw.alipayobjects.com/os/lib/antv/g6/4.8.24/dist/g6.min.js"></script>
  <script>
    const payload = __GRAPH_JSON__;
    const container = document.getElementById('g6-topology');
    const detail = document.getElementById('node-detail');
    const detailOpen = document.getElementById('detail-open');
    const edgeList = document.getElementById('edge-list');

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[char]));
    }

    function isMissing(value) {
      return value === null || value === undefined || value === '' || value === 'null';
    }

    function display(value, fallback = '无') {
      return isMissing(value) ? fallback : value;
    }

    function metricText(value) {
      return isMissing(value) ? '无' : String(value);
    }

    function numberText(value, digits) {
      if (isMissing(value)) return '无';
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed.toFixed(digits) : String(value);
    }

    function detailHtml(data) {
      const tags = [...new Set([...(Array.isArray(data.badges) ? data.badges : []), data.platform].filter(Boolean))];
      const author = data.author || {};
      const fieldEvidence = Array.isArray(data.field_evidence) ? data.field_evidence : [];
      const evidenceRows = fieldEvidence.map((row) => (
        `<div class="detail-line">字段证据：${esc(display(row.field))} / ${esc(display(row.status))} / ${esc(display(row.source))} / ${numberText(row.confidence, 2)}</div>`
      )).join('');
      return `
        <div class="detail-header">
          <div class="detail-title">${esc(data.id || '节点详情')}</div>
          <button class="detail-close" type="button" data-close-detail="1">×</button>
        </div>
        ${data.image_url ? `<img class="detail-img" src="${esc(data.image_url)}" alt="">` : ''}
        <div class="detail-line">图片相似度：${numberText(data.similarity, 4)}</div>
        <div class="detail-line">页面标题：${esc(display(data.title))}</div>
        <div class="detail-line">平台/页面：${esc(display(data.platform))} / ${esc(display(data.page_type))}</div>
        <div class="detail-line">外部证据状态：${esc(display(data.evidence_node_status))}</div>
        <div class="detail-line">进入主时间线：${data.allow_in_external_timeline ? '是' : '否'}</div>
        <div class="detail-line">reason：${esc(display(data.reason))}</div>
        <div class="detail-line">节点裁决：${esc(display(data.node_decision_reason))}</div>
        <div class="detail-line">作者：${esc(display(author.name))}</div>
        <div class="detail-line">粉丝：${esc(display(author.followers))}　关注：${esc(display(author.following))}　认证：${author.is_big_v ? '是' : '否'}</div>
        <div class="detail-line">宏观指标：转${metricText(data.repost_count)} / 赞${metricText(data.like_count)} / 评${metricText(data.comment_count)}</div>
        <div class="detail-line">来源提示：${esc(display(data.source_text))}</div>
        <div class="detail-line">来源账号/平台：${esc(display(data.source_account_hint))} / ${esc(display(data.source_platform_hint))}</div>
        <div class="detail-line">${data.url ? `<a href="${esc(data.url)}" target="_blank" rel="noreferrer">打开链接</a>` : ''}${data.source_url ? `　<a href="${esc(data.source_url)}" target="_blank" rel="noreferrer">来源链接</a>` : ''}</div>
        ${evidenceRows}
        <div class="detail-tags">${tags.map((tag) => `<span>${esc(tag)}</span>`).join('')}</div>
      `;
    }

    const tooltip = new G6.Tooltip({
      offsetX: 12,
      offsetY: 16,
      itemTypes: ['node'],
      getContent(evt) {
        const data = evt.item.getModel().data || {};
        const div = document.createElement('div');
        div.innerHTML = detailHtml(data);
        return div;
      }
    });

    edgeList.innerHTML = payload.edgeRows.length
      ? payload.edgeRows.slice(0, 24).map((edge) => `<span class="edge-row">${esc(edge.source)} → ${esc(edge.target)}｜${esc(edge.method)}｜${Number(edge.confidence || 0).toFixed(2)}</span>`).join('')
      : '<span class="edge-row">暂无可展示的帖子 → 帖子边</span>';

    const graph = new G6.Graph({
      container: 'g6-topology',
      width: container.clientWidth,
      height: container.clientHeight,
      fitView: false,
      plugins: [tooltip],
      modes: { default: ['drag-canvas', 'zoom-canvas', 'drag-node', 'activate-relations'] },
      defaultNode: {
        type: 'circle',
        style: { fill: '#f8fafc', stroke: '#5b7c99', lineWidth: 1.8 },
        labelCfg: { position: 'bottom', offset: 9, style: { fill: '#172033', fontSize: 11, lineHeight: 16 } }
      },
      defaultEdge: {
        type: 'cubic-horizontal',
        style: { stroke: '#94a3b8', lineWidth: 1.6, endArrow: true },
        labelCfg: { autoRotate: true, style: { fill: '#344054', fontSize: 10 } }
      },
      nodeStateStyles: {
        selected: { lineWidth: 4, shadowColor: '#94a3b8', shadowBlur: 12 },
        hover: { shadowColor: '#94a3b8', shadowBlur: 10 }
      },
      edgeStateStyles: { hover: { lineWidth: 3.5 } }
    });

    graph.data({ nodes: payload.nodes, edges: payload.edges });
    graph.render();
    document.getElementById('fullscreen-btn').addEventListener('click', async () => {
      const shell = document.querySelector('.graph-shell');
      try {
        if (!document.fullscreenElement && shell.requestFullscreen) {
          await shell.requestFullscreen();
        }
      } catch (err) {}
      setTimeout(() => {
        graph.changeSize(container.clientWidth, container.clientHeight);
        graph.fitView(30);
      }, 180);
    });
    document.getElementById('fit-btn').addEventListener('click', () => {
      graph.fitView(30);
    });

    const group = graph.get('group');
    const laneGroup = group.addGroup({ id: 'platform-lanes' });
    payload.lanes.forEach((lane, index) => {
      laneGroup.addShape('line', {
        attrs: {
          x1: 72,
          y1: lane.y,
          x2: Math.max(payload.canvasWidth - 80, 960),
          y2: lane.y,
          stroke: '#e5edf5',
          lineWidth: 1.2
        },
        name: 'platform-lane-line'
      });
      laneGroup.addShape('text', {
        attrs: {
          x: 72,
          y: lane.y - 72,
          text: `${lane.label}（${lane.count}）`,
          fill: '#344054',
          fontSize: 13,
          fontWeight: 700
        },
        name: 'platform-lane-label'
      });
    });
    laneGroup.toBack();
    graph.translate(10, 20);

    function setDetail(data) {
      detail.innerHTML = detailHtml(data || {});
      detail.classList.remove('is-hidden');
      detailOpen.classList.remove('is-visible');
    }

    detail.addEventListener('click', (evt) => {
      if (evt.target && evt.target.dataset && evt.target.dataset.closeDetail) {
        detail.classList.add('is-hidden');
        detailOpen.classList.add('is-visible');
      }
    });
    detailOpen.addEventListener('click', () => {
      detail.classList.remove('is-hidden');
      detailOpen.classList.remove('is-visible');
    });

    graph.on('node:mouseenter', (evt) => graph.setItemState(evt.item, 'hover', true));
    graph.on('node:mouseleave', (evt) => graph.setItemState(evt.item, 'hover', false));
    graph.on('edge:mouseenter', (evt) => graph.setItemState(evt.item, 'hover', true));
    graph.on('edge:mouseleave', (evt) => graph.setItemState(evt.item, 'hover', false));
    graph.on('node:click', (evt) => {
      graph.getNodes().forEach((node) => graph.clearItemStates(node, ['selected']));
      graph.setItemState(evt.item, 'selected', true);
      setDetail(evt.item.getModel().data || {});
    });
    graph.on('edge:click', (evt) => {
      const edge = evt.item.getModel().data || {};
      detail.innerHTML = `
        <div class="detail-header">
          <div class="detail-title">${esc(edge.source)} → ${esc(edge.target)}</div>
          <button class="detail-close" type="button" data-close-detail="1">×</button>
        </div>
        <div class="detail-line">confidence：${Number(edge.confidence || 0).toFixed(2)}</div>
        <div class="detail-line">method：${esc(edge.method || '-')}</div>
        <div class="detail-line">evidence：${esc(edge.evidence || 'no evidence')}</div>
      `;
      detail.classList.remove('is-hidden');
      detailOpen.classList.remove('is-visible');
    });
    window.addEventListener('resize', () => {
      graph.changeSize(container.clientWidth, container.clientHeight);
    });
    if (payload.nodes.length) {
      setDetail(payload.nodes[0].data || {});
    }
  </script>
</body>
</html>
"""


def write_g6_html(result: AgentState, output_path: Path) -> None:
    """将 AgentState 渲染为 G6 拓扑图 HTML 文件。"""
    payload = build_g6_payload(result)
    graph_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    height = max(760, int(payload.get("canvasHeight") or 760) + 120)
    html = HTML_TEMPLATE.replace("__GRAPH_JSON__", graph_json).replace("__HEIGHT__", str(height))
    output_path.write_text(html, encoding="utf-8")
