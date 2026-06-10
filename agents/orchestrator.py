from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.state import AgentState, DEFAULT_MERMAID_GRAPH, append_log

# ── LLM insight client ─────────────────────────────────────────────────

try:
    import requests as _requests
except ImportError:
    _requests = None


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name, str(default).lower()).strip().lower()
    return val in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(int(os.getenv(name, str(default))), minimum)
    except ValueError:
        return default


def _looks_like_placeholder(value: str) -> bool:
    markers = ("xxxx", "your-", "replace-", "sk-xxxx", "fc-xxxx")
    v = value.strip().lower()
    return not v or any(m in v for m in markers)


class InsightLLMClient:
    """Lightweight OpenAI-compatible client for generating report insights."""

    def __init__(self) -> None:
        self.api_key = os.getenv("ORCHESTRATOR_INSIGHT_API_KEY",
                                 os.getenv("LLM_API_KEY", ""))
        self.base_url = os.getenv("ORCHESTRATOR_INSIGHT_BASE_URL",
                                  os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
        self.model = os.getenv("ORCHESTRATOR_INSIGHT_MODEL",
                               os.getenv("LLM_MODEL", "gpt-4o"))
        self.timeout = _env_int("ORCHESTRATOR_INSIGHT_TIMEOUT_SECONDS", 45, 5)
        self.max_tokens = _env_int("ORCHESTRATOR_INSIGHT_MAX_TOKENS", 1200, 256)
        self.enabled = False
        self.reason = ""

        if not _env_flag("ORCHESTRATOR_ENABLE_INSIGHT", True):
            self.reason = "disabled by ORCHESTRATOR_ENABLE_INSIGHT"
            return
        if _looks_like_placeholder(self.api_key):
            self.reason = "missing or placeholder API key"
            return
        if _requests is None:
            self.reason = "requests not installed"
            return
        self.enabled = True

    def generate_insight(self, report_data: Dict[str, Any]) -> str:
        """Call LLM to generate an insight report from structured data."""
        if not self.enabled:
            return ""

        prompt = self._build_prompt(report_data)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _INSIGHT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.4,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = _requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                body = resp.json()
                return body["choices"][0]["message"]["content"].strip()
            return f"[洞察生成失败: HTTP {resp.status_code}]"
        except Exception as exc:
            return f"[洞察生成异常: {exc}]"

    def _build_prompt(self, data: Dict[str, Any]) -> str:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)


_INSIGHT_SYSTEM_PROMPT = """你是一个图片溯源分析专家。用户会提供一份结构化的溯源数据，请基于这些数据生成一份"分析洞察报告"。

要求：
1. 用中文写，语言专业但不生硬，像是一位分析专家在向决策者做口头汇报
2. 从数据中提炼出 3-5 个核心洞察点，每个洞察要有数据支撑
3. 回答"这份报告告诉我们什么"——不要重复罗列数据，要给出判断和建议
4. 如果数据不足，诚实指出不确定性，不要编造
5. 结构（严格按照以下格式，每个 ## 标题不可省略）：

## 结论
（一句话总结本次溯源的核心结论，30字以内）

## 核心发现
（3-5 条，每条用 **粗体关键词**：开头，后面跟 1-2 句解释。每条之间空一行）

## 多维评估
（用以下固定格式给出 0-100 分的评估，分数必须基于数据合理推断，不可随意给满分或零分：
**溯源可信度**：XX分 — （一句话理由，说明为什么是这个分数）
**证据完整性**：XX分 — （一句话理由）
**时间线一致性**：XX分 — （一句话理由）
**来源多样性**：XX分 — （一句话理由）
**传播影响力**：XX分 — （一句话理由）
）

## 建议与对策
（3 条具体可操作的建议，面向决策者）

输出纯文本。评分基于数据本身质量，不要编造。如果数据不足，相应维度应该低分并说明原因。"""


# ── 常量 ────────────────────────────────────────────────────────────────

_INSIGHT_SYSTEM_PROMPT = """你是图片传播链溯源系统中的编排分析智能体。用户会提供结构化溯源数据，你要把数据转成面向决策者的结论，而不是复述字段。

请用中文输出，严格使用以下 Markdown 结构，标题不可省略：

## 结论
用 1 句话说明本次溯源最重要的判断：能否定位疑似源头、可信度高低、还需要什么复核。

## 核心发现
输出 3-5 条。每条用 **加粗关键词**：开头，后面说明这个发现意味着什么、对判断有什么作用。必须带数据依据。

## 多维评估
按固定格式输出 0-100 分，分数要基于数据质量，不能全部高分：
**溯源可信度**：XX分 — 一句话理由。
**证据完整度**：XX分 — 一句话理由。
**时间线一致性**：XX分 — 一句话理由。
**来源多样性**：XX分 — 一句话理由。
**传播影响力**：XX分 — 一句话理由。

## 作用与风险
说明这份结果能帮助我们做什么，例如锁定核查对象、判断传播路径、识别篡改/重复风险；同时说明不能直接下结论的风险。

## 建议与对策
输出 4-6 条可执行建议，面向业务/风控/内容审核/人工复核人员。建议要具体，不要泛泛而谈。

只基于输入数据做推断。数据不足时明确说不确定，并降低对应评分。"""

SOURCE_LABELS = {
    "search_result": "搜索结果",
    "page_metadata": "页面元数据",
    "time_tag": "HTML time 标签",
    "visible_text": "页面可见文本",
    "url_pattern": "URL 日期模式",
    "http_last_modified": "HTTP Last-Modified",
    "llm": "LLM 结构化提取",
    "missing": "缺失",
}


def upload_node(state: AgentState) -> AgentState:
    target = state.get("target_image", {})
    logs = append_log(
        state,
        f"upload_node: 已接收目标图片 {target.get('filename', 'unknown file')}",
    )

    return {
        "target_image": target,
        "search_engines": state.get("search_engines", []),
        "retriever_max_results": state.get("retriever_max_results"),
        "retriever_max_results_per_engine": state.get("retriever_max_results_per_engine"),
        "nodes_data": [],
        "retrieval_summary": state.get("retrieval_summary", {}),
        "mermaid_graph": "",
        "final_report": "",
        "execution_logs": logs,
    }


def _dated_nodes(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [node for node in nodes if node.get("published_at")]


def _author_line(node: Dict[str, Any]) -> str:
    author = node.get("author") or node.get("metadata_author") or node.get("publisher")
    return author or "未知"


def _source_label(source: str) -> str:
    return SOURCE_LABELS.get(source, source or "missing")


def _time_detail(node: Dict[str, Any]) -> str:
    evidence = node.get("time_evidence", {})
    parts = [
        f"搜索结果: {evidence.get('search_result') or '-'}",
        f"页面元数据: {evidence.get('page_metadata') or '-'}",
        f"time 标签: {evidence.get('time_tag') or '-'}",
        f"可见文本: {evidence.get('visible_text') or '-'}",
        f"URL: {evidence.get('url_pattern') or '-'}",
        f"Last-Modified: {evidence.get('http_last_modified') or '-'}",
        f"LLM: {evidence.get('llm') or '-'}",
    ]
    return "; ".join(parts)


def report_node(state: AgentState) -> AgentState:
    logs = append_log(state, "report_node: 正在生成最终报告和拓扑图。")
    nodes = state.get("nodes_data", [])
    dated = _dated_nodes(nodes)
    retrieval_summary = state.get("retrieval_summary", {})
    validation_summary = state.get("validation_summary", {})
    analysis_summary = state.get("analysis_summary", {})

    # ── 事实报告 ──
    report_lines = [
        "## 图片溯源报告",
        "",
        f"- 搜索引擎: {', '.join(state.get('search_engines', [])) or '无'}",
        f"- 通过相似度校验的候选节点: {len(nodes)}",
        f"- 提取到时间证据的节点: {len(dated)}",
        f"- 检索状态: {retrieval_summary.get('status', 'unknown')}",
        "",
    ]

    if dated:
        origin = dated[0]
        report_lines.extend([
            "### 疑似最早来源",
            f"- 发布时间: {origin.get('published_at')}",
            f"- 作者 / 发布者: {_author_line(origin)}",
            f"- 标题: {origin.get('title', '无标题')}",
            f"- 链接: {origin.get('url', '')}",
            f"- 域名: {origin.get('domain', '')}",
            f"- 搜索引擎: {origin.get('engine', '')}",
            f"- 时间证据来源: {_source_label(origin.get('date_source', 'missing'))}",
            "",
            "这是当前通过校验的候选集中最早的带时间节点。"
            "如果页面元数据或抓取数据不可用，系统会退回使用搜索结果摘要、标题或 URL 中的时间线索。",
            "",
        ])
    elif nodes:
        report_lines.extend([
            "### 来源时间",
            "当前候选节点没有提取到可靠时间。"
            "可配置 Firecrawl 和 LLM 密钥，或选择能暴露发布时间元数据的搜索结果。",
            "",
        ])
    else:
        report_lines.extend([
            "### 来源时间",
            "当前没有可用于溯源的校验候选节点。",
            "",
        ])

    if nodes:
        report_lines.append("### 候选传播时间线")
        for index, node in enumerate(nodes, start=1):
            report_lines.extend([
                f"{index}. {node.get('published_at') or '未知时间'} | "
                f"{_source_label(node.get('date_source', 'missing'))} | "
                f"{node.get('engine', '未知引擎')} | "
                f"{node.get('domain', '未知域名')} | "
                f"作者: {_author_line(node)} | "
                f"[{node.get('title', '无标题')}]({node.get('url', '')})",
                f"   - 时间证据: {_time_detail(node)}",
                f"   - 相似度: {node.get('similarity', '-')}",
                f"   - Analyzer: 角色={node.get('propagation_role', '未知')} | "
                f"发布者={_author_line(node)} | "
                f"浏览={node.get('view_count', 0)} | 转发={node.get('repost_count', 0)} | "
                f"评论={node.get('comment_count', 0)} | 点赞={node.get('like_count', 0)}",
                f"   - 拓扑指标: 节点权重={node.get('node_weight', 0.0)} | "
                f"源头分数={node.get('source_score', 0.0)} | "
                f"疑似源头={node.get('is_suspected_source', False)} | "
                f"关键节点={node.get('is_key_node', False)}",
                f"   - 抓取链路: {node.get('crawl_source', 'unknown')} / "
                f"{node.get('crawl_status', 'unknown')} | LLM使用={node.get('llm_used', False)}",
                f"   - 分析说明: {node.get('analyzer_reason', '-')}",
            ])
        undated_count = len(nodes) - len(dated)
        if undated_count:
            report_lines.append("")
            report_lines.append(f"{undated_count} 个校验候选节点没有可用时间证据。")

    # ── AI 洞察报告 ──
    insight_text = ""
    confidence_scores = _compute_confidence_scores(
        state, nodes, dated, retrieval_summary, validation_summary, analysis_summary
    )
    insight_client = InsightLLMClient()
    if insight_client.enabled and nodes:
        insight_data = _build_insight_payload(state, nodes, dated, retrieval_summary,
                                              validation_summary, analysis_summary)
        insight_data["confidence_scores"] = confidence_scores
        insight_text = insight_client.generate_insight(insight_data)
        if insight_text:
            report_lines.extend([
                "",
                "---",
                "",
                "## 分析洞察",
                "",
                insight_text,
            ])
            logs.append(f"[{time.strftime('%H:%M:%S')}] report_node: AI 洞察生成完成")
        else:
            logs.append(f"[{time.strftime('%H:%M:%S')}] report_node: AI 洞察跳过 ({insight_client.reason})")
    else:
        logs.append(f"[{time.strftime('%H:%M:%S')}] report_node: AI 洞察跳过 ({insight_client.reason})")

    if nodes:
        local_insight_text = _build_local_insight_text(
            state, nodes, dated, retrieval_summary, validation_summary, analysis_summary, confidence_scores
        )
        report_lines.extend([
            "",
            "---",
            "",
            "## 决策洞察与评分",
            "",
            local_insight_text,
        ])
        insight_text = f"{insight_text}\n\n{local_insight_text}".strip()

    analyzer_graph = state.get("mermaid_graph")
    return {
        "final_report": "\n".join(report_lines),
        "insight_report": insight_text,
        "confidence_scores": confidence_scores,
        "mermaid_graph": analyzer_graph or build_mermaid_graph(nodes),
        "execution_logs": logs,
    }


def _build_insight_payload(
    state: AgentState,
    nodes: List[Dict[str, Any]],
    dated: List[Dict[str, Any]],
    retrieval: Dict[str, Any],
    validation: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Build structured data for the LLM insight prompt."""
    suspected = [n for n in nodes if n.get("is_suspected_source")]
    key_nodes = [n for n in nodes if n.get("is_key_node")]
    tampered = [n for n in nodes if n.get("suspected_tampering")
                or (n.get("tamper_analysis") or {}).get("is_tampered")]

    platform_counts: Dict[str, int] = {}
    for n in nodes:
        p = n.get("platform") or n.get("platform_family") or "unknown"
        platform_counts[p] = platform_counts.get(p, 0) + 1

    return {
        "target_image": {
            "filename": state.get("target_image", {}).get("filename", ""),
        },
        "search_engines": state.get("search_engines", []),
        "summary": {
            "total_nodes": len(nodes),
            "dated_nodes": len(dated),
            "suspected_sources": len(suspected),
            "key_nodes": len(key_nodes),
            "tampered_nodes": len(tampered),
            "platform_distribution": platform_counts,
        },
        "earliest_source": {
            "time": dated[0].get("published_at") if dated else "unknown",
            "publisher": _author_line(dated[0]) if dated else "unknown",
            "title": dated[0].get("title", "") if dated else "",
            "url": dated[0].get("url", "") if dated else "",
            "platform": dated[0].get("platform", "") if dated else "",
        } if dated else None,
        "suspected_sources": [
            {
                "time": n.get("published_at"),
                "publisher": _author_line(n),
                "title": n.get("title", ""),
                "platform": n.get("platform", ""),
                "source_score": n.get("source_score", 0),
            }
            for n in suspected[:5]
        ],
        "key_nodes": [
            {
                "time": n.get("published_at"),
                "publisher": _author_line(n),
                "title": n.get("title", ""),
                "platform": n.get("platform", ""),
                "repost_count": n.get("repost_count", 0),
            }
            for n in key_nodes[:5]
        ],
        "tampered_nodes": [
            {
                "time": n.get("published_at"),
                "publisher": _author_line(n),
                "title": n.get("title", ""),
                "tampering_type": n.get("tampering_type", ""),
            }
            for n in tampered[:5]
        ],
        "retrieval_status": retrieval.get("status", "unknown"),
        "validation_summary": {
            "validated_count": validation.get("validated_count", len(nodes)),
            "rejected_count": validation.get("rejected_count", 0),
        },
    }


def _clamp_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _platform_distribution(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for node in nodes:
        platform = str(node.get("platform") or node.get("platform_family") or node.get("engine") or "unknown")
        counts[platform] = counts.get(platform, 0) + 1
    return counts


def _compute_confidence_scores(
    state: AgentState,
    nodes: List[Dict[str, Any]],
    dated: List[Dict[str, Any]],
    retrieval: Dict[str, Any],
    validation: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    total = max(len(nodes), 1)
    dated_ratio = len(dated) / total
    suspected = [node for node in nodes if node.get("is_suspected_source")]
    key_nodes = [node for node in nodes if node.get("is_key_node")]
    platform_count = len(_platform_distribution(nodes))
    search_engine_count = len(state.get("search_engines", []) or [])
    edge_count = _safe_int(analysis.get("topology_edge_count"), 0)
    if not edge_count:
        topo = state.get("topology_data") if isinstance(state.get("topology_data"), dict) else {}
        edge_count = len(topo.get("edges") or []) if isinstance(topo, dict) else 0

    source_score_values = []
    for node in suspected or dated[:3]:
        try:
            source_score_values.append(float(node.get("source_score") or 0))
        except (TypeError, ValueError):
            pass
    avg_source_score = sum(source_score_values) / len(source_score_values) if source_score_values else 0.0

    validated_count = _safe_int(validation.get("validated_count"), len(nodes))
    rejected_count = _safe_int(validation.get("rejected_count"), 0)
    input_count = _safe_int(validation.get("input_count"), validated_count + rejected_count or total)
    validation_ratio = validated_count / max(input_count, 1)

    credibility = _clamp_score(35 + dated_ratio * 24 + min(avg_source_score, 1.0) * 24 + min(len(suspected), 3) * 5)
    completeness = _clamp_score(25 + min(total, 20) * 1.6 + dated_ratio * 22 + min(edge_count, 12) * 2 + validation_ratio * 12)
    timeline = _clamp_score(30 + dated_ratio * 45 + (15 if dated else 0) + min(len(dated), 8))
    diversity = _clamp_score(25 + min(platform_count, 5) * 10 + min(search_engine_count, 6) * 4)
    impact = _clamp_score(20 + min(len(key_nodes), 6) * 9 + min(edge_count, 12) * 2 + min(total, 30))

    return {
        "trace_credibility": {
            "score": credibility,
            "reason": f"有 {len(dated)}/{len(nodes)} 个节点具备时间证据，疑似源头 {len(suspected)} 个，源头分数均值约 {avg_source_score:.2f}。",
        },
        "evidence_completeness": {
            "score": completeness,
            "reason": f"通过校验 {validated_count} 个、过滤 {rejected_count} 个，拓扑边 {edge_count} 条，证据覆盖仍取决于页面抓取质量。",
        },
        "timeline_consistency": {
            "score": timeline,
            "reason": f"时间线可排序节点占比 {dated_ratio:.0%}；无时间节点越多，源头排序不确定性越高。",
        },
        "source_diversity": {
            "score": diversity,
            "reason": f"结果覆盖 {platform_count} 类来源、{search_engine_count} 个检索入口，可用于交叉验证但仍需人工抽检。",
        },
        "propagation_impact": {
            "score": impact,
            "reason": f"识别关键节点 {len(key_nodes)} 个、拓扑边 {edge_count} 条，可辅助判断传播扩散范围。",
        },
    }


def _build_local_insight_text(
    state: AgentState,
    nodes: List[Dict[str, Any]],
    dated: List[Dict[str, Any]],
    retrieval: Dict[str, Any],
    validation: Dict[str, Any],
    analysis: Dict[str, Any],
    scores: Dict[str, Dict[str, Any]],
) -> str:
    suspected = [node for node in nodes if node.get("is_suspected_source")]
    key_nodes = [node for node in nodes if node.get("is_key_node")]
    tampered = [
        node for node in nodes
        if node.get("suspected_tampering") or (isinstance(node.get("tamper_analysis"), dict) and node["tamper_analysis"].get("is_tampered"))
    ]
    platforms = _platform_distribution(nodes)
    earliest = dated[0] if dated else None
    earliest_title = earliest.get("title") if earliest else ""
    earliest_time = earliest.get("published_at") if earliest else "未知"
    earliest_author = _author_line(earliest) if earliest else "未知"
    credibility_score = scores["trace_credibility"]["score"]

    conclusion = (
        f"当前可定位到疑似最早节点：{earliest_author}（{earliest_time}），可信度 {credibility_score} 分，建议以人工复核确认原始发布时间。"
        if earliest
        else f"当前缺少可靠发布时间证据，可信度 {credibility_score} 分，只能作为候选传播线索而非最终源头结论。"
    )

    findings = [
        f"- **源头定位价值**：系统从 {len(nodes)} 个有效节点中提取到 {len(dated)} 个带时间证据节点，最早线索为“{earliest_title or '未知标题'}”。这可以帮助优先锁定人工核查对象。",
        f"- **校验与去重质量**：输入 {validation.get('input_count', len(nodes))} 个候选，通过 {validation.get('validated_count', len(nodes))} 个，强确定去重后输出 {validation.get('deduplicated_count', len(nodes))} 个节点；这能减少重复页面对源头排序的干扰。",
        f"- **传播结构作用**：识别到 {len(key_nodes)} 个关键节点和 {len(suspected)} 个疑似源头节点，可用于判断谁更可能是首发、谁更可能是扩散放大者。",
        f"- **来源覆盖情况**：当前覆盖 {len(platforms)} 类来源（{', '.join(list(platforms)[:5]) or '未知'}），跨平台线索越多，越有利于交叉验证发布时间与内容演变。",
    ]
    if tampered:
        findings.append(f"- **篡改风险提示**：发现 {len(tampered)} 个疑似篡改节点，需要单独核对图片裁剪、水印、文字覆盖或二次编辑痕迹。")

    score_lines = [
        ("溯源可信度", scores["trace_credibility"]),
        ("证据完整度", scores["evidence_completeness"]),
        ("时间线一致性", scores["timeline_consistency"]),
        ("来源多样性", scores["source_diversity"]),
        ("传播影响力", scores["propagation_impact"]),
    ]
    score_text = "\n".join(f"**{label}**：{item['score']}分 — {item['reason']}" for label, item in score_lines)

    actions = [
        "- 优先打开疑似最早节点原文，核对页面显示时间、源代码时间、平台接口时间是否一致。",
        "- 对疑似源头和关键节点截图留证，保留 URL、发布时间、作者、平台、互动量和图片版本。",
        "- 对无时间证据但相似度高的节点做人工抽检，避免漏掉早期搬运或缓存页面。",
        "- 对疑似篡改/重复簇单独比对图片水印、裁剪区域和 OCR 文本，确认是否属于二次传播。",
        "- 若用于风控或对外结论，建议补充平台侧原始数据或第三方网页快照作为强证据。",
    ]

    return "\n\n".join([
        "## 结论\n" + conclusion,
        "## 核心发现\n" + "\n\n".join(findings[:5]),
        "## 多维评估\n" + score_text,
        "## 作用与风险\n这份结果的作用是把海量候选收敛为可核查的源头候选、关键传播节点和风险节点；风险在于网页时间、搜索摘要和平台抓取结果可能不完整，不能把单一最早时间直接当作法律或事实最终结论。",
        "## 建议与对策\n" + "\n".join(actions),
    ])


def _label(value: str, fallback: str) -> str:
    return (value or fallback).replace('"', "'")


def build_mermaid_graph(nodes: List[Dict[str, Any]]) -> str:
    mermaid_lines = [DEFAULT_MERMAID_GRAPH]

    if not nodes:
        mermaid_lines.append('    N0["无校验候选"]')
        return "\n".join(mermaid_lines)

    graph_nodes = [node for node in nodes if node.get("is_suspected_source") or node.get("is_key_node")]
    if not graph_nodes:
        graph_nodes = nodes

    for index, node in enumerate(graph_nodes, start=1):
        label = "<br/>".join(
            [
                _label(str(node.get("published_at") or ""), "未知时间"),
                _label(str(node.get("domain") or ""), "未知域名"),
                _label(_author_line(node), "未知作者"),
                _label(str(node.get("engine") or ""), "未知引擎"),
                _label(str(node.get("propagation_role") or ""), "未知角色"),
                _label(f"sim={node.get('similarity', '-')}", "sim=-"),
            ]
        )
        mermaid_lines.append(f'    N{index}["{label}"]')

    graph_id_by_node_id = {
        str(node.get("id")): f"N{index}"
        for index, node in enumerate(graph_nodes, start=1)
    }
    source_graph_id = next(
        (
            graph_id_by_node_id[str(node.get("id"))]
            for node in graph_nodes
            if node.get("is_suspected_source")
        ),
        "N1",
    )
    edges: set[tuple[str, str]] = set()
    for index, node in enumerate(graph_nodes, start=1):
        current_graph_id = f"N{index}"
        parent_graph_id = graph_id_by_node_id.get(str(node.get("parent_id") or ""))
        if not parent_graph_id and current_graph_id != source_graph_id:
            parent_graph_id = source_graph_id
        if parent_graph_id and parent_graph_id != current_graph_id:
            edges.add((parent_graph_id, current_graph_id))

    for parent_graph_id, child_graph_id in sorted(edges):
        mermaid_lines.append(f"    {parent_graph_id} --> {child_graph_id}")

    return "\n".join(mermaid_lines)


if __name__ == "__main__":
    mock_state: AgentState = {
        "target_image": {"filename": "demo.jpg"},
        "search_engines": ["baidu", "tineye"],
        "nodes_data": [
            {
                "title": "Example source",
                "url": "https://example.com/origin",
                "published_at": "2024-03-14 09:12",
                "domain": "example.com",
                "engine": "baidu",
                "author": "example author",
                "date_source": "page_metadata",
                "similarity": 0.95,
                "time_evidence": {
                    "search_result": "",
                    "page_metadata": "2024-03-14 09:12",
                    "time_tag": "",
                    "visible_text": "",
                    "url_pattern": "",
                },
            },
        ],
        "execution_logs": [],
    }
    initialized = upload_node(mock_state)
    reported = report_node({**mock_state, "execution_logs": initialized["execution_logs"]})
    print(json.dumps(reported, ensure_ascii=False, indent=2))
