"""Tests for LLM metadata extractor (enriched schema v2)."""

from __future__ import annotations

import json

from llmscrapy.extractor import LLMExtractor
from llmscrapy.models import EnrichedMetadata, FieldEvidence


class TestLLMExtractor:
    def test_extract_parses_valid_json(self, mock_openai_client):
        extractor = LLMExtractor()
        extractor._client = mock_openai_client
        result = extractor.extract(
            "测试网页内容",
            title_hint="测试标题",
            target_image_url="https://img.example.com/target.jpg",
            source_platform="baijiahao.baidu.com",
        )

        assert isinstance(result, EnrichedMetadata)
        assert result.platform_family == "baidu_media"
        assert result.page_type == "news_article"

        # Content
        assert result.content.title == "测试新闻标题 — 这是一篇关于AI技术的报道"
        assert result.content.author == "张三"
        assert result.content.publisher == "百家号"
        assert result.content.published_at == "2025-12-15 14:30:00"
        assert "人工智能" in result.content.description

        # Metrics
        assert result.metrics.view_count == 125000
        assert result.metrics.like_count == 3421
        assert result.metrics.comment_count == 567
        assert result.metrics.repost_count == 891
        assert result.metrics.share_count is None

        # Provenance
        assert result.provenance.source_platform_hint == "baijiahao"
        assert result.provenance.source_account_hint == "张三"
        assert result.provenance.confidence == 0.92
        assert len(result.provenance.evidence) >= 1

        # Image occurrence
        assert result.image_occurrence.target_or_variant_present == "probable"
        assert result.image_occurrence.occurrence_type == "edited_variant"
        assert result.image_occurrence.confidence == 0.6

        # Node decision
        assert result.node_decision.allow_in_external_timeline is True
        assert len(result.node_decision.reason) > 0

        # Field evidence (new per-field dict structure)
        assert isinstance(result.field_evidence, dict)
        assert "title" in result.field_evidence
        fe_title = result.field_evidence["title"]
        assert isinstance(fe_title, FieldEvidence)
        assert "测试新闻标题" in (fe_title.value or "")
        assert fe_title.source == "visible_text"
        assert len(fe_title.snippet) > 0

        assert "author" in result.field_evidence
        assert result.field_evidence["author"].value == "张三"

        assert "view_count" in result.field_evidence
        assert result.field_evidence["view_count"].source == "visible_text"

        # Backward-compatible property access
        assert result.title == result.content.title
        assert result.author == result.content.author
        assert result.publish_time == result.content.published_at
        assert result.view_count == result.metrics.view_count
        assert result.confidence == result.provenance.confidence

    def test_extract_handles_markdown_code_block(self, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = (
            '```json\n{"platform_family": "blog", "page_type": "blog_post", '
            '"content": {"title": "MD Title", "author": "", "publisher": "", '
            '"description": "Desc from page", "published_at": "", "modified_at": "", '
            '"canonical_url": "", "image_urls": []}, '
            '"metrics": {}, "provenance": {"confidence": 0.8, "evidence": []}, '
            '"image_occurrence": {"target_or_variant_present": "not_found", '
            '"occurrence_type": "unrelated", "confidence": 0.0, "evidence": []}, '
            '"node_decision": {"evidence_node_status": "contextual_only", '
            '"allow_in_external_timeline": false, '
            '"allow_cross_platform_relation_candidate": false, "reason": ""}, '
            '"field_evidence": {}}\n```'
        )

        extractor = LLMExtractor()
        extractor._client = mock_openai_client
        result = extractor.extract("测试内容")

        assert result.content.title == "MD Title"
        assert result.platform_family == "blog"

    def test_extract_handles_invalid_json(self, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "This is not valid JSON at all..."

        extractor = LLMExtractor()
        extractor._client = mock_openai_client
        result = extractor.extract("测试内容")
        assert "Parse error" in (result.content.description or "")

    def test_extract_handles_empty_response(self, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = ""
        extractor = LLMExtractor()
        extractor._client = mock_openai_client
        result = extractor.extract("测试内容")
        assert result.platform_family == "news"  # default

    def test_extract_handles_api_error(self, mock_openai_client):
        mock_openai_client.chat.completions.create.side_effect = RuntimeError(
            "API timeout"
        )
        extractor = LLMExtractor()
        extractor._client = mock_openai_client
        result = extractor.extract("测试内容")
        assert "API timeout" in (result.content.description or "")

    def test_extract_numeric_parsing(self, mock_openai_client):
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = (
            '{"platform_family": "news", "page_type": "news_article", '
            '"content": {"title": "", "author": "", "publisher": "", '
            '"description": "", "published_at": "", "modified_at": "", '
            '"canonical_url": "", "image_urls": []}, '
            '"metrics": {"view_count": "not_a_number", "like_count": null, '
            '"comment_count": 100, "repost_count": null, "share_count": null}, '
            '"provenance": {"confidence": 0.9, "evidence": []}, '
            '"image_occurrence": {"target_or_variant_present": "unclear", '
            '"occurrence_type": "unknown", "confidence": 0.0, "evidence": []}, '
            '"node_decision": {"evidence_node_status": "contextual_only", '
            '"allow_in_external_timeline": false, '
            '"allow_cross_platform_relation_candidate": false, "reason": ""}, '
            '"field_evidence": {}}'
        )

        extractor = LLMExtractor()
        extractor._client = mock_openai_client
        result = extractor.extract("测试内容")

        assert result.metrics.view_count is None  # invalid number
        assert result.metrics.like_count is None
        assert result.metrics.comment_count == 100

    def test_field_evidence_missing_reason(self, mock_openai_client):
        """Field with source='missing' should have a reason."""
        mock_openai_client.chat.completions.create.return_value.choices[
            0
        ].message.content = json.dumps({
            "platform_family": "news",
            "page_type": "news_article",
            "content": {
                "title": "", "description": "", "published_at": "",
                "modified_at": "", "publisher": "", "author": "",
                "canonical_url": "", "image_urls": [],
            },
            "metrics": {
                "view_count": None, "like_count": None, "comment_count": None,
                "repost_count": None, "share_count": None,
            },
            "provenance": {"confidence": 0.0, "evidence": []},
            "image_occurrence": {
                "target_or_variant_present": "unclear",
                "occurrence_type": "unknown",
                "confidence": 0.0, "evidence": [],
            },
            "node_decision": {
                "evidence_node_status": "no_evidence",
                "allow_in_external_timeline": False,
                "allow_cross_platform_relation_candidate": False,
                "reason": "无可用证据",
            },
            "field_evidence": {
                "title": {
                    "value": None, "source": "missing",
                    "snippet": "", "reason": "页面无标题",
                },
                "view_count": {
                    "value": None, "source": "missing",
                    "snippet": "", "reason": "页面未公开阅读量",
                },
            },
        }, ensure_ascii=False)

        extractor = LLMExtractor()
        extractor._client = mock_openai_client
        result = extractor.extract("空页面")

        fe = result.field_evidence
        assert fe["title"].source == "missing"
        assert len(fe["title"].reason) > 0
        assert fe["view_count"].source == "missing"
        assert "阅读" in fe["view_count"].reason


