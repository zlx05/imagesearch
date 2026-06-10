"""Shared test fixtures and mocks for llmscrapy tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Sample data paths
SAMPLE_JSON = Path(__file__).parent / "fixtures" / "sample_urls.json"
SAMPLE_HTML = Path(__file__).parent / "fixtures" / "sample_page.html"


@pytest.fixture
def sample_json_content() -> str:
    """Return a minimal valid JSON for testing."""
    return json.dumps(
        {
            "timestamp": "20260531_173746",
            "summary": {"input": 10, "passed": 3, "rejected": 7, "final": 3},
            "nodes": [
                {
                    "id": "r1",
                    "url": "https://example.com/article1",
                    "title": "",
                    "similarity": 0.85,
                    "source": "example.com",
                    "engine": "test",
                    "image_url": "",
                    "possible_duplicate": False,
                    "reason": "test entry",
                },
                {
                    "id": "r2",
                    "url": "https://example.com/article2",
                    "title": "Test Title",
                    "similarity": 0.72,
                    "source": "example.com",
                    "engine": "test",
                    "image_url": "https://img.example.com/pic.jpg",
                    "possible_duplicate": True,
                    "reason": "duplicate candidate",
                },
                {
                    # Entry without url — should be skipped
                    "id": "r3",
                    "url": "",
                    "title": "No URL",
                    "similarity": 0.50,
                    "source": "example.com",
                    "engine": "test",
                },
            ],
        },
        ensure_ascii=False,
    )


@pytest.fixture
def sample_json_file(tmp_path: Path, sample_json_content: str) -> Path:
    """Write sample JSON to a temp file and return its path."""
    json_file = tmp_path / "test_urls.json"
    json_file.write_text(sample_json_content, encoding="utf-8")
    return json_file


@pytest.fixture
def sample_html() -> str:
    """Return a realistic Chinese news article HTML snippet."""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>测试新闻标题 — 这是一篇关于AI技术的报道</title>
    <meta property="og:title" content="测试新闻标题 — 这是一篇关于AI技术的报道">
    <meta name="author" content="张三">
</head>
<body>
    <header>
        <nav>导航菜单</nav>
    </header>
    <article>
        <h1>测试新闻标题 — 这是一篇关于AI技术的报道</h1>
        <div class="article-info">
            <span class="author">作者：张三</span>
            <span class="time">发布时间：2025-12-15 14:30:00</span>
            <span class="source">来源：百家号</span>
        </div>
        <div class="content">
            <p>近日，人工智能领域又有重大突破。多家科技公司发布了新一代大语言模型，
            在多项基准测试中取得了前所未有的成绩。</p>
            <p>业内专家表示，这些进展将深刻影响各行各业的数字化转型进程。</p>
            <p>与此同时，监管层面也在加快推进AI治理框架的建立。</p>
        </div>
        <div class="stats">
            <span>阅读量：12.5万</span>
            <span>点赞：3421</span>
            <span>评论：567</span>
            <span>转发：891</span>
        </div>
    </article>
    <footer>
        <p>版权所有 © 2025 百家号</p>
    </footer>
    <script>console.log("tracking code")</script>
    <style>.hidden { display: none; }</style>
</body>
</html>"""


@pytest.fixture
def sample_llm_response() -> str:
    """Return a realistic DeepSeek JSON response (enriched schema v2)."""
    return json.dumps(
        {
            "platform_family": "baidu_media",
            "page_type": "news_article",
            "content": {
                "title": "测试新闻标题 — 这是一篇关于AI技术的报道",
                "description": "近日，人工智能领域又有重大突破。多家科技公司发布了新一代大语言模型。",
                "published_at": "2025-12-15 14:30:00",
                "modified_at": "",
                "publisher": "百家号",
                "author": "张三",
                "canonical_url": "https://baijiahao.baidu.com/s?id=1849192621334484334",
                "image_urls": ["https://img.example.com/ai_photo.jpg"],
            },
            "metrics": {
                "view_count": 125000,
                "like_count": 3421,
                "comment_count": 567,
                "repost_count": 891,
                "share_count": None,
            },
            "provenance": {
                "source_text": "来源：百家号",
                "source_url": None,
                "source_platform_hint": "baijiahao",
                "source_account_hint": "张三",
                "confidence": 0.92,
                "evidence": [
                    {
                        "field": "source_platform",
                        "value": "百家号",
                        "snippet": "来源：百家号",
                        "confidence": 0.95,
                    }
                ],
            },
            "image_occurrence": {
                "target_or_variant_present": "probable",
                "occurrence_type": "edited_variant",
                "caption": "AI技术示意图",
                "image_credit": None,
                "confidence": 0.6,
                "evidence": [
                    {
                        "field": "image_match",
                        "value": "ai_photo.jpg",
                        "snippet": "图片alt：AI技术示意图",
                        "confidence": 0.6,
                    }
                ],
            },
            "node_decision": {
                "evidence_node_status": "contextual_only",
                "allow_in_external_timeline": True,
                "allow_cross_platform_relation_candidate": False,
                "reason": "有发布时间和来源，可作为时间线节点",
            },
            "field_evidence": {
                "title": {
                    "value": "测试新闻标题 — 这是一篇关于AI技术的报道",
                    "source": "visible_text",
                    "snippet": "测试新闻标题 — 这是一篇关于AI技术的报道",
                    "reason": "",
                },
                "author": {
                    "value": "张三",
                    "source": "visible_text",
                    "snippet": "作者：张三",
                    "reason": "",
                },
                "view_count": {
                    "value": "125000",
                    "source": "visible_text",
                    "snippet": "阅读量：12.5万",
                    "reason": "",
                },
            },
        },
        ensure_ascii=False,
    )


@pytest.fixture
def mock_openai_client(sample_llm_response: str):
    """Create a mock OpenAI client that returns a controlled response."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = sample_llm_response
    mock_client.chat.completions.create.return_value.choices = [mock_choice]
    return mock_client


@pytest.fixture
def mock_requests_get(sample_html: str):
    """Create a mock for requests.get that returns sample HTML."""
    with patch("requests.Session.get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = sample_html
        mock_response.status_code = 200
        mock_response.apparent_encoding = "utf-8"
        mock_get.return_value = mock_response
        yield mock_get
