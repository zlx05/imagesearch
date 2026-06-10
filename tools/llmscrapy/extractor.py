"""LLM-based metadata extraction using DeepSeek V4 (OpenAI-compatible API).

Produces a structured EnrichedMetadata document with field_evidence for
each extracted field.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

from .config import LLMConfig
from .models import (
    ContentInfo,
    EnrichedMetadata,
    EvidenceItem,
    FieldEvidence,
    ImageOccurrence,
    MetricsInfo,
    NodeDecision,
    ProvenanceInfo,
)


# ── Extraction prompt — keeps JSON braces literal; placeholders are
#    {text} and {context_hints} (replaced via .replace(), not .format()) ─

EXTRACTION_PROMPT = """你是一个**网页内容取证**助手。你只能基于提供的网页文本提取信息，**绝不编造**。找不到的数据用 null/空值。

---

## 一、platform_family（平台家族）
根据 URL、域名、页面内容综合判断，选择**一个**：
- `"news"` — 新闻资讯平台（网易、搜狐、新浪等非百度系）
- `"baidu_media"` — 百度内容生态（百家号、百度新闻、mbd.baidu.com 等）
- `"forum"` — 论坛/社区（贴吧、知乎、Voz、Reddit、LiveJournal等）
- `"blog"` — 博客/个人站点（blog.jp、WordPress 等）
- `"telegram"` — Telegram 频道/群组
- `"dynamic_platform"` — 强动态/SPA 平台（抖音、快手、微博等，JavaScript 渲染为主）
- `"unknown"` — 无法判断

## 二、page_type（页面类型）
表示当前 URL 是内容页还是聚合页，选择**一个**：
- `"news_article"` — 新闻文章
- `"forum_post"` — 论坛帖子/讨论串
- `"blog_post"` — 博客文章
- `"channel_post"` — 频道消息（Telegram 等）
- `"aggregator"` — 聚合列表/信息流
- `"feed_page"` — RSS/Feed 页面
- `"dynamic_platform"` — 动态平台页面（内容可能不完整）
- `"unknown"` — 无法判断

## 三、content（内容信息）

**content.title**：页面自身标题。优先取网页 <title>、og:title、文章标题 H1。不要用搜索结果标题替代。找不到则为 ""。

**content.description**：页面摘要/导语。**必须从页面中直接提取**，优先 meta description、og:description、文章导语、正文前几句。**禁止自己编写总结**。找不到则为 ""。

**content.published_at**：内容发布时间。标准化为 YYYY-MM-DD HH:mm:ss。
- 优先取 article:published_time、og:published_time、jsonld:datePublished、fc:publishedAt
- 如果以上都没有但 modified_at（或 fc:dateUpdate）存在，**必须用 modified_at 的值作为 published_at**，在 field_evidence 中标注 source="fallback_from_modified"，reason 说明原因
- 如果连 modified_at 都没有，从可见文本中搜索"发布时间""发表于""时间"等
- 只有月日而无年份则返回 ""
- 不要填抓取时间、当前时间、版权年份
- 找不到则为 ""

**content.modified_at**：内容修改/更新时间。
- 优先取 article:modified_time、jsonld:dateModified、fc:dateUpdate、fc:modifiedAt、dc:date、dcterms:modified
- 不要用发布时间替代。找不到则为 ""

**content.publisher**：发布主体。新闻站填媒体名，百家号填号主名，论坛/博客填用户昵称。
- 优先级：结构化 metadata（og:site_name、jsonld:publisher）> 页面作者区 > 正文可见的"作者/发布者/来源/账号" > **站点域名推断**
- **重要 fallback**：如果以上都没有，根据域名推断（如 baijiahao.baidu.com → "百家号"、163.com → "网易"、sohu.com → "搜狐"），confidence 设为 0.5
- 找不到则为 ""

**content.author**：具体作者/编辑/发帖人。和 publisher 区别：publisher 是机构/账号，author 是写作者个人。没有明确个人作者则返回 ""。

**content.canonical_url**：规范 URL。优先 canonical link、og:url、页面声明的原文链接。不要把"来源链接"误当成 canonical。找不到则为 ""。

**content.image_urls**：正文图片 URL 列表。来自 og:image、正文 img、picture source。**过滤**：不放站点 logo、头像、广告图、图标。尽量只保留内容图片。

## 四、metrics（互动指标）
每一项都是**整数或 null**。**这是重点攻克字段**。

### 数值识别规则：
- 中文数字单位转换：万=×10000，k=×1000，亿=×100000000
- 常见格式：`1234`、`1,234`、`1.2万`、`12k`、`1.2万阅读`、`点赞 3421`
- **只取当前文章/帖子的数据**，不要取推荐阅读、评论区子项、其他文章的
- 在页面顶部（标题下方或侧边栏）找到的优先；在底部"评论"区域附近的次之
- 如果数据出现在**评论区标题旁边**（如 `评论 567`），那是评论数

### 各平台特征（帮助定位）：
- **百家号**：在标题/作者栏下方可能看到日期和地点；底部评论区标题格式为 `## 评论 N`，N 就是评论数；点赞数通常不在纯文本中显示。`2025-12-10 941阅读` 这种是推荐阅读列表的数据，不是当前文章的。
- **搜狐/网易新闻**：标题下方常见 `阅读 XXXX`、`评论 XXX`；部分文章有点赞数。
- **抖音/快手**：视频描述下方有 `点赞 XXX`、`评论 XXX`、`转发 XXX`、`收藏 XXX`。但如果 markdown 里只有 `粉丝 X.XK 获赞 X.XK`，这是**作者主页统计**不是当前视频数据，不要取。评论区数字也可能需要登录才能看到。
- **论坛（voz/LiveJournal）**：帖子信息栏有 like/reply/comment 计数，格式如 `Reactions: N`、`Comments: N`。
- **Telegram**：消息下方有 view 计数，格式如 `N views`。

### 搜索模式优先级：
1. 找 `## 评论 N` 或 `评论 N 条` 或 `N 条评论` → comment_count
2. 找 `点赞 N` 或 `N 人赞过` 或 `N 赞` → like_count
3. 找 `阅读 N` 或 `N 次阅读` 或 `N 播放` → view_count（注意排除推荐阅读区的数字）
4. 找 `转发 N` 或 `N 次转发` → repost_count
5. 找 `分享 N` 或 `N 次分享` → share_count

### 字段：
- **view_count**：浏览/阅读/播放/观看次数。注意区分"阅读"和"播放"
- **like_count**：点赞/喜欢/赞同/收藏数
- **comment_count**：评论/回复/讨论数
- **repost_count**：转发/转载/转推数
- **share_count**：分享数

**重要**：即使只有一项指标可见，也要提取。没有明确证据的返回 null，不要编造。

## 五、provenance（来源溯源）

这是**传播来源证据**，不是当前页面的自我描述。

**provenance.source_text**：页面中关于来源、转载、引用、原文、via、from 的原文片段。例如"来源：央视新闻""转载自微博用户 xxx"。找不到则为 null。

**provenance.source_url**：页面明确给出的来源 URL、原文链接、引用链接。不要把当前页面 URL、canonical URL 当作来源 URL。找不到则为 null。

**provenance.source_platform_hint**：从 source_text/source_url **推断**的来源平台。例如 "weibo"、"xiaohongshu"、"douyin"、"baijiahao"、"163.com"。必须基于来源证据，不要凭主题猜测。找不到则为 null。

**provenance.source_account_hint**：来源中的账号/作者名。例如"来源：@某某""转载自 小红书用户 xxx"。找不到则为 null。

**provenance.confidence**：来源关系置信度（0.0-1.0）。有明确来源 URL → 0.9+；有"来源：账号/媒体名" → 0.7-0.8；只有模糊文本 → 0.3-0.5；无任何来源信息 → 0.0。

**provenance.evidence**：证据片段数组。每个包含 field、value、snippet（≤100字原文）、confidence。

## 六、image_occurrence（目标图片出现判断）

检查页面中是否存在**目标图片**（target_image_url）或其变体（裁剪、缩放、水印等）的**文本线索**。

**硬性规则**：
- 你只能看到文本，**不能做像素级图片比对**
- **禁止**返回 target_or_variant_present="confirmed"
- **禁止**返回 occurrence_type="same_image"
- 纯文本 LLM 最多给 "probable"/"edited_variant"

**字段**：
- **target_or_variant_present**：`"probable"`（文本线索强烈匹配）| `"unclear"`（无法判断）| `"not_found"`（确认不存在）
- **occurrence_type**：`"edited_variant"`（可能有变体）| `"screenshot_reference"`（截图引用）| `"unrelated"`（无关图片）| `"unknown"`（无法判断）
- **caption**：图片配文/alt 文本/说明文字。没有则 null。
- **image_credit**：图片署名/供图/来源。没有则 null。
- **confidence**：0.0-1.0。基于文本线索强度，不是图片比对置信度。
- **evidence**：图片 URL 文件名、alt、图注、正文附近说明等。

## 七、node_decision（分析器建议）

这是给 analyzer 的**爬取建议**，最终裁决由 analyzer 做出。

- **evidence_node_status**：`"direct_evidence"` | `"contextual_only"` | `"no_evidence"`
- **allow_in_external_timeline**：建议是否允许加入外部时间线（true/false）。条件：页面可访问 + 有发布时间 + 是内容页而非聚合页。
- **allow_cross_platform_relation_candidate**：建议是否作为跨平台关联候选（true/false）。条件：页面明确引用/转载自其他平台，或有来源证据。
- **reason**：建议理由（中文 ≤100字）。

## 八、field_evidence（字段级证据）

一个 JSON 对象，key 是字段名，value 是单条证据记录。每个字段**只出现一次**（不是数组）。

**证据记录结构**：
```json
{
  "value": "提取到的值（字符串或null）",
  "source": "metadata | visible_text | json_ld | og_tag | llm_extraction | missing",
  "snippet": "页面中的原文片段（≤100字），source=missing时可为空",
  "reason": "只有 source=missing 时才需要填，解释为什么缺失"
}
```

**规则**：
- 每个字段给一个状态，但**不要伪造 evidence**
- 有原文证据 → source + snippet 必填
- 找不到 → value=null, source="missing", reason 解释原因
- 至少覆盖：title, published_at, publisher, author, view_count, like_count, comment_count, repost_count, share_count, source_platform, source_account
- 如果某个字段值为 null，必须用 source="missing" + reason 标记

---

## 返回格式
**只返回纯 JSON 对象**，不要 ```json 或其他文字。

```json
{
  "platform_family": "news",
  "page_type": "news_article",
  "content": {
    "title": "",
    "description": "",
    "published_at": "",
    "modified_at": "",
    "publisher": "",
    "author": "",
    "canonical_url": "",
    "image_urls": []
  },
  "metrics": {
    "view_count": null,
    "like_count": null,
    "comment_count": null,
    "repost_count": null,
    "share_count": null
  },
  "provenance": {
    "source_text": null,
    "source_url": null,
    "source_platform_hint": null,
    "source_account_hint": null,
    "confidence": 0.0,
    "evidence": []
  },
  "image_occurrence": {
    "target_or_variant_present": "unclear",
    "occurrence_type": "unknown",
    "caption": null,
    "image_credit": null,
    "confidence": 0.0,
    "evidence": []
  },
  "node_decision": {
    "evidence_node_status": "contextual_only",
    "allow_in_external_timeline": false,
    "allow_cross_platform_relation_candidate": false,
    "reason": ""
  },
  "field_evidence": {
    "title": {"value": "", "source": "missing", "snippet": "", "reason": ""}
  }
}
```

---

## 网页文本
{text}

## 上下文提示
{context_hints}

请返回 JSON:"""


class LLMExtractor:
    """Extracts structured metadata from webpage text using an LLM."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        return self._client

    def extract(
        self,
        text: str,
        title_hint: str = "",
        target_image_url: str = "",
        source_platform: str = "",
        head_metadata: dict | None = None,
    ) -> EnrichedMetadata:
        """Extract enriched metadata from webpage text using the LLM.

        Args:
            text: Cleaned webpage text.
            title_hint: Optional HTML <title> tag for guidance.
            target_image_url: The source image_url to check for occurrence.
            source_platform: The source domain from the input JSON.
            head_metadata: Structured metadata extracted from HTML <head>
                (meta tags, JSON-LD, OG tags, etc.).

        Returns:
            An EnrichedMetadata object.
        """
        # Build context hints
        hints_parts = []
        if title_hint:
            hints_parts.append(f"网页 HTML 标题：{title_hint}")
        if source_platform:
            hints_parts.append(f"输入的源平台（source字段）：{source_platform}")
        if target_image_url:
            hints_parts.append(
                f"目标图片 URL（target_image_url）：{target_image_url}"
            )

        # ── Structured metadata from <head> ────────────────────────
        if head_metadata:
            meta_lines = ["## 从 HTML <head> 提取的结构化元数据（优先参考）："]
            # Show time-related fields first (most important)
            priority_keys = [
                "article:published_time", "article:modified_time",
                "og:published_time", "published_time",
                "jsonld:datePublished", "jsonld:dateModified",
                "date", "pubdate", "publishdate",
                "dc:date", "dcterms:issued", "dcterms:modified",
                "fc:publishedAt", "fc:modifiedAt",
                "time:datetime",
            ]
            shown = set()
            for key in priority_keys:
                val = head_metadata.get(key)
                if val:
                    meta_lines.append(f"- {key}: {val}")
                    shown.add(key)
            # Then remaining fields
            for key, val in sorted(head_metadata.items()):
                if key not in shown and val:
                    meta_lines.append(f"- {key}: {val}")
            hints_parts.append("\n".join(meta_lines))

        context = "\n\n".join(hints_parts) if hints_parts else "（无额外上下文）"

        # Use replace() to avoid JSON braces conflicting with .format()
        prompt = EXTRACTION_PROMPT.replace("{text}", text[:6000]).replace(
            "{context_hints}", context
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个精确的网页取证数据提取器。"
                            "你的任务是根据提供的网页文本，提取所有指定的结构化字段。"
                            "始终返回有效的 JSON，找不到的数据用 null 或空值表示，"
                            "绝不编造数据。"
                            "field_evidence 中每个字段是一个对象（不是数组），"
                            "包含 value/source/snippet/reason。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            raw = response.choices[0].message.content or ""
            return self._parse_response(raw)

        except Exception as e:
            meta = EnrichedMetadata()
            meta.content.description = f"[LLM extraction failed: {e}]"
            return meta

    def _parse_response(self, raw: str) -> EnrichedMetadata:
        """Parse the LLM JSON response into an EnrichedMetadata object."""
        json_str = raw.strip()

        # Remove markdown code fences if present
        json_match = re.search(r"\{.*\}", json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            meta = EnrichedMetadata()
            meta.content.description = (
                f"[Parse error — raw response snippet: {raw[:300]}]"
            )
            return meta

        # ── Parse nested objects ─────────────────────────────────
        return EnrichedMetadata(
            platform_family=str(data.get("platform_family", "news") or "news"),
            page_type=str(data.get("page_type", "news_article") or "news_article"),

            content=_parse_content(data.get("content", {})),
            metrics=_parse_metrics(data.get("metrics", {})),
            provenance=_parse_provenance(data.get("provenance", {})),
            image_occurrence=_parse_image_occurrence(data.get("image_occurrence", {})),
            node_decision=_parse_node_decision(data.get("node_decision", {})),
            field_evidence=_parse_field_evidence(data.get("field_evidence", {})),
        )


# ── Sub-parsers ─────────────────────────────────────────────────────

def _parse_content(raw: dict) -> ContentInfo:
    return ContentInfo(
        title=str(raw.get("title") or "").strip(),
        description=str(raw.get("description") or "").strip(),
        published_at=str(raw.get("published_at") or "").strip(),
        modified_at=str(raw.get("modified_at") or "").strip(),
        publisher=str(raw.get("publisher") or "").strip(),
        author=str(raw.get("author") or "").strip(),
        canonical_url=str(raw.get("canonical_url") or "").strip(),
        image_urls=_parse_str_list(raw.get("image_urls")),
    )


def _parse_metrics(raw: dict) -> MetricsInfo:
    return MetricsInfo(
        view_count=_parse_int_or_none(raw.get("view_count")),
        like_count=_parse_int_or_none(raw.get("like_count")),
        comment_count=_parse_int_or_none(raw.get("comment_count")),
        repost_count=_parse_int_or_none(raw.get("repost_count")),
        share_count=_parse_int_or_none(raw.get("share_count")),
    )


def _parse_provenance(raw: dict) -> ProvenanceInfo:
    return ProvenanceInfo(
        source_text=_parse_str_or_none(raw.get("source_text")),
        source_url=_parse_str_or_none(raw.get("source_url")),
        source_platform_hint=_parse_str_or_none(raw.get("source_platform_hint")),
        source_account_hint=_parse_str_or_none(raw.get("source_account_hint")),
        confidence=float(raw.get("confidence", 0.0) or 0.0),
        evidence=_parse_evidence_list(raw.get("evidence", [])),
    )


def _parse_image_occurrence(raw: dict) -> ImageOccurrence:
    return ImageOccurrence(
        target_or_variant_present=str(
            raw.get("target_or_variant_present") or "unclear"
        ),
        occurrence_type=str(raw.get("occurrence_type") or "unknown"),
        caption=_parse_str_or_none(raw.get("caption")),
        image_credit=_parse_str_or_none(raw.get("image_credit")),
        confidence=float(raw.get("confidence", 0.0) or 0.0),
        evidence=_parse_evidence_list(raw.get("evidence", [])),
    )


def _parse_node_decision(raw: dict) -> NodeDecision:
    return NodeDecision(
        evidence_node_status=str(
            raw.get("evidence_node_status") or "contextual_only"
        ),
        allow_in_external_timeline=bool(
            raw.get("allow_in_external_timeline", False)
        ),
        allow_cross_platform_relation_candidate=bool(
            raw.get("allow_cross_platform_relation_candidate", False)
        ),
        reason=str(raw.get("reason") or "").strip(),
    )


def _parse_field_evidence(raw: dict) -> dict[str, FieldEvidence]:
    """Parse field_evidence: dict of field_name → FieldEvidence."""
    if not isinstance(raw, dict):
        return {}
    result: dict[str, FieldEvidence] = {}
    for field_name, item in raw.items():
        if isinstance(item, dict):
            result[str(field_name)] = FieldEvidence(
                value=str(item.get("value") or "") if item.get("value") is not None else None,
                source=str(item.get("source") or ""),
                snippet=str(item.get("snippet") or ""),
                reason=str(item.get("reason") or ""),
            )
        elif isinstance(item, list) and len(item) > 0:
            # Backward compat: if LLM returns array, take first element
            first = item[0]
            if isinstance(first, dict):
                result[str(field_name)] = FieldEvidence(
                    value=str(first.get("value") or "") if first.get("value") is not None else None,
                    source=str(first.get("source") or ""),
                    snippet=str(first.get("snippet") or ""),
                    reason=str(first.get("reason") or ""),
                )
    return result


def _parse_evidence_list(raw: list) -> list[EvidenceItem]:
    if not isinstance(raw, list):
        return []
    out: list[EvidenceItem] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(EvidenceItem(
                field=str(item.get("field") or ""),
                value=str(item.get("value") or ""),
                snippet=str(item.get("snippet") or ""),
                confidence=float(item.get("confidence", 0.0) or 0.0),
            ))
    return out


def _parse_str_list(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if x]


def _parse_int_or_none(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_str_or_none(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None
