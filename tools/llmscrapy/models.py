"""Pydantic data models for llmscrapy."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════
# Input / pipeline models
# ══════════════════════════════════════════════════════════════════════

class URLSource(BaseModel):
    """A single URL entry loaded from a JSON file."""

    id: str
    url: str
    title: str = ""
    source: str = ""  # e.g. "baijiahao.baidu.com"
    engine: str = ""  # e.g. "baidu"
    image_url: str = ""
    possible_duplicate: bool = False
    reason: str = ""
    similarity: float = 0.0


class FetchedPage(BaseModel):
    """The result of fetching a URL."""

    url: str
    html: str = ""
    status_code: int = 0
    error: str = ""
    fetched_at: datetime = Field(default_factory=datetime.now)


class ParsedPage(BaseModel):
    """Cleaned text extracted from HTML."""

    url: str
    title: str = ""
    text: str = ""  # cleaned body text
    text_length: int = 0
    head_metadata: dict = Field(default_factory=dict)  # meta tags, JSON-LD, OG etc.


# ══════════════════════════════════════════════════════════════════════
# LLM extraction schema — rich structured output
# ══════════════════════════════════════════════════════════════════════

class ContentInfo(BaseModel):
    """Content identity fields."""
    title: str = ""
    description: str = ""
    published_at: str = ""   # ISO 8601 / YYYY-MM-DD HH:mm:ss
    modified_at: str = ""
    publisher: str = ""      # publishing organisation
    author: str = ""         # individual author / account name
    canonical_url: str = ""
    image_urls: list[str] = Field(default_factory=list)


class MetricsInfo(BaseModel):
    """Engagement / popularity metrics."""
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    repost_count: Optional[int] = None
    share_count: Optional[int] = None


class EvidenceItem(BaseModel):
    """A single piece of evidence for a field (used in provenance, image_occurrence)."""
    field: str = ""
    value: str = ""
    snippet: str = ""        # the relevant text fragment from the page
    confidence: float = 0.0


class FieldEvidence(BaseModel):
    """Per-field evidence record for field_evidence.

    - If found:     value + source + snippet are populated
    - If missing:   value=null, source="missing", reason explains why
    """
    value: Optional[str] = None
    source: str = ""         # "metadata" | "visible_text" | "json_ld" | "og_tag" | "llm_extraction" | "missing"
    snippet: str = ""
    reason: str = ""         # only meaningful when source="missing"


class ProvenanceInfo(BaseModel):
    """Where the content came from — source attribution."""
    source_text: Optional[str] = None       # raw text indicating the source
    source_url: Optional[str] = None        # original / canonical URL if different
    source_platform_hint: Optional[str] = None  # e.g. "baijiahao", "weibo"
    source_account_hint: Optional[str] = None   # e.g. "张津涛", "人民日报"
    confidence: float = 0.0
    evidence: list[EvidenceItem] = Field(default_factory=list)


class ImageOccurrence(BaseModel):
    """Whether a specific target image (or variant) appears on the page.

    HARD RULE: LLM MUST NOT return "confirmed"/"same_image" without
    actual pixel-level image comparison. From text alone, the maximum
    is "probable"/"edited_variant".
    """
    target_or_variant_present: str = "unclear"  # "confirmed" | "probable" | "unclear" | "not_found"
    occurrence_type: str = "unknown"  # "same_image" | "edited_variant" | "screenshot_reference" | "unrelated" | "unknown"
    caption: Optional[str] = None
    image_credit: Optional[str] = None
    confidence: float = 0.0
    evidence: list[EvidenceItem] = Field(default_factory=list)


class NodeDecision(BaseModel):
    """Evidence assessment for the validator pipeline."""
    evidence_node_status: str = "contextual_only"  # "direct_evidence" | "contextual_only" | "no_evidence"
    allow_in_external_timeline: bool = False
    allow_cross_platform_relation_candidate: bool = False
    reason: str = ""


class EnrichedMetadata(BaseModel):
    """The full structured output produced by the LLM extractor.

    This replaces the old flat ExtractedMetadata with a rich,
    evidence-tracked schema suitable for content validation pipelines.
    """

    platform_family: str = "news"       # "news" | "social_media" | "forum" | "blog" | "other"
    page_type: str = "news_article"     # "news_article" | "social_post" | "video_page" | "image_gallery" | "other"
    content: ContentInfo = Field(default_factory=ContentInfo)
    metrics: MetricsInfo = Field(default_factory=MetricsInfo)
    provenance: ProvenanceInfo = Field(default_factory=ProvenanceInfo)
    image_occurrence: ImageOccurrence = Field(default_factory=ImageOccurrence)
    node_decision: NodeDecision = Field(default_factory=NodeDecision)
    field_evidence: dict[str, FieldEvidence] = Field(default_factory=dict)

    # ── Backward-compatible flat accessors ──────────────────────────

    @property
    def title(self) -> str:
        return self.content.title

    @property
    def author(self) -> str:
        return self.content.author

    @property
    def source_name(self) -> str:
        return self.provenance.source_platform_hint or ""

    @property
    def publish_time(self) -> str:
        return self.content.published_at or self.content.modified_at

    @property
    def view_count(self) -> Optional[int]:
        return self.metrics.view_count

    @property
    def like_count(self) -> Optional[int]:
        return self.metrics.like_count

    @property
    def comment_count(self) -> Optional[int]:
        return self.metrics.comment_count

    @property
    def repost_count(self) -> Optional[int]:
        return self.metrics.repost_count

    @property
    def share_count(self) -> Optional[int]:
        return self.metrics.share_count

    @property
    def summary(self) -> str:
        return self.content.description

    @property
    def tags(self) -> list[str]:
        return []

    @property
    def confidence(self) -> float:
        return self.provenance.confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        self.provenance.confidence = value


# ══════════════════════════════════════════════════════════════════════
# Legacy alias (ExtractedMetadata = EnrichedMetadata)
# ══════════════════════════════════════════════════════════════════════

ExtractedMetadata = EnrichedMetadata


class CrawlResult(BaseModel):
    """The complete result of crawling one URL."""

    source: URLSource
    fetched: Optional[FetchedPage] = None
    parsed: Optional[ParsedPage] = None
    metadata: Optional[EnrichedMetadata] = None
    error: str = ""  # pipeline-level error

    @property
    def succeeded(self) -> bool:
        return (
            not self.error
            and self.fetched is not None
            and self.fetched.status_code == 200
        )
