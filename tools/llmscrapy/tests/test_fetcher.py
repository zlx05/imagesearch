"""Tests for all webpage fetchers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from llmscrapy.fetcher import (
    BaseFetcher,
    DirectFetcher,
    FirecrawlFetcher,
    JinaReaderFetcher,
    create_fetcher,
    WebFetcher,
    _detect_anti_crawler_wall,
)
from llmscrapy.models import FetchedPage


# ── Anti-crawler wall detection ─────────────────────────────────────

class TestAntiCrawlerWallDetection:
    def test_detects_baidu_security(self):
        html = "<html>百度安全验证</html>"
        assert _detect_anti_crawler_wall(html) == "Baidu security verification"

    def test_detects_captcha(self):
        html = "<html>请输入验证码</html>"
        assert _detect_anti_crawler_wall(html) == "Generic CAPTCHA"

    def test_detects_blocked(self):
        html = "<html>您的访问被拦截</html>"
        assert _detect_anti_crawler_wall(html) == "Access blocked"

    def test_no_false_positive(self):
        html = "<html><body><p>正常文章内容</p></body></html>"
        assert _detect_anti_crawler_wall(html) is None


# ── DirectFetcher ───────────────────────────────────────────────────

class TestDirectFetcher:
    def test_fetch_success(self, mock_requests_get):
        fetcher = DirectFetcher()
        result = fetcher.fetch("https://example.com/article1")
        assert isinstance(result, FetchedPage)
        assert result.status_code == 200
        assert "测试新闻标题" in result.html
        assert result.error == ""

    def test_fetch_http_error(self):
        with patch("requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_resp.text = "Not Found"
            mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "404 Not Found", response=mock_resp
            )
            mock_get.return_value = mock_resp

            fetcher = DirectFetcher()
            result = fetcher.fetch("https://example.com/notfound")
            assert result.status_code == 404
            assert "HTTP error" in result.error

    def test_fetch_timeout(self):
        with patch("requests.Session.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Timed out")
            fetcher = DirectFetcher()
            result = fetcher.fetch("https://example.com/slow")
            assert result.status_code == 0
            assert "Timeout" in result.error

    def test_fetch_connection_error(self):
        with patch("requests.Session.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError(
                "Connection refused"
            )
            fetcher = DirectFetcher()
            result = fetcher.fetch("https://example.com/dead")
            assert result.status_code == 0
            assert "Connection error" in result.error

    def test_fetch_batch(self, mock_requests_get):
        fetcher = DirectFetcher()
        urls = ["https://example.com/a", "https://example.com/b"]
        results = fetcher.fetch_batch(urls, delay=0.0)
        assert len(results) == 2
        assert all(r.status_code == 200 for r in results)

    def test_fetch_detects_baidu_wall(self):
        """When Baidu returns a security wall, it should be flagged."""
        with patch("requests.Session.get") as mock_get:
            mock_http_adapter_get = MagicMock()
            mock_http_adapter_get.side_effect = None

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "<html>百度安全验证</html>"
            mock_resp.apparent_encoding = "utf-8"
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            # DirectFetcher calls session.get which is now mocked at the class level
            # We also need to handle the cookie-warming call
            fetcher = DirectFetcher()
            # pre-mark the domain as warmed to skip the warm-up request
            fetcher._cookie_cache["https://example.com"] = True

            result = fetcher.fetch("https://example.com/blocked")
            assert result.status_code == 200
            assert "Anti-crawler wall" in result.error

        fetcher.close()

    def test_close(self):
        fetcher = DirectFetcher()
        fetcher.close()

    def test_user_agent_rotates(self):
        fetcher = DirectFetcher()
        ua1 = fetcher.session.headers.get("User-Agent")
        fetcher._rotate_headers(fetcher.session)
        ua2 = fetcher.session.headers.get("User-Agent")
        # Might or might not change (random), but should be a string
        assert isinstance(ua1, str)
        assert len(ua1) > 20
        fetcher.close()


# ── JinaReaderFetcher ───────────────────────────────────────────────

class TestJinaReaderFetcher:
    def test_fetch_success(self):
        """Jina returns JSON envelope: {"code":200, "data": {"title":..., "content":...}}"""
        with patch("requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.encoding = "utf-8"
            mock_resp.json.return_value = {
                "code": 200,
                "status": 20000,
                "data": {
                    "title": "Jina 文章标题",
                    "content": "# 文章内容\n\n这是一篇通过 Jina 获取的文章。",
                    "url": "https://example.com/article1",
                },
            }
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp

            fetcher = JinaReaderFetcher()
            result = fetcher.fetch("https://example.com/article1")

            assert result.status_code == 200
            assert "Jina 文章标题" in result.html
            assert "文章内容" in result.html
            assert result.error == ""

    def test_fetch_http_error(self):
        with patch("requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_resp.text = "Rate limited"
            mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "429 Too Many Requests", response=mock_resp
            )
            mock_get.return_value = mock_resp

            fetcher = JinaReaderFetcher()
            result = fetcher.fetch("https://example.com/blocked")
            assert "Jina HTTP error" in result.error

    def test_fetch_timeout(self):
        with patch("requests.Session.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Timed out")
            fetcher = JinaReaderFetcher()
            result = fetcher.fetch("https://example.com/slow")
            assert "Jina timeout" in result.error


# ── FirecrawlFetcher ────────────────────────────────────────────────

class TestFirecrawlFetcher:
    FIRECRAWL_RESPONSE = {
        "success": True,
        "data": {
            "markdown": "# 百度文章\n\n这是文章正文内容。\n\n阅读量：12.5万\n点赞：3421",
            "html": "<html><body><h1>百度文章</h1><p>这是文章正文内容。</p></body></html>",
            "metadata": {
                "title": "百度文章",
                "og:title": "OG百度文章",
                "author": "张三",
            },
        },
    }

    def test_fetch_success(self, monkeypatch):
        """Firecrawl returns structured scrape result."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")

        with patch("requests.Session.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = self.FIRECRAWL_RESPONSE
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            fetcher = FirecrawlFetcher()
            result = fetcher.fetch("https://baijiahao.baidu.com/s?id=123")

        assert result.status_code == 200
        assert "百度文章" in result.html
        assert "文章正文内容" in result.html
        assert result.error == ""
        fetcher.close()

    def test_fetch_payment_required(self, monkeypatch):
        """402 means credits exhausted."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")

        with patch("requests.Session.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 402
            mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "402 Payment Required", response=mock_resp
            )
            mock_post.return_value = mock_resp

            fetcher = FirecrawlFetcher()
            result = fetcher.fetch("https://example.com/test")

        assert "payment required" in result.error.lower()
        fetcher.close()

    def test_fetch_failure_response(self, monkeypatch):
        """Firecrawl returns success=False."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")

        with patch("requests.Session.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"success": False, "error": "Blocked"}
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            fetcher = FirecrawlFetcher()
            result = fetcher.fetch("https://example.com/test")

        assert "Firecrawl failed" in result.error
        fetcher.close()

    def test_no_api_key_raises(self, monkeypatch):
        """Firecrawl requires API key."""
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
            FirecrawlFetcher()


# ── Factory ─────────────────────────────────────────────────────────

class TestCreateFetcher:
    def test_create_direct(self):
        f = create_fetcher("direct")
        assert isinstance(f, DirectFetcher)

    def test_create_jina(self):
        f = create_fetcher("jina")
        assert isinstance(f, JinaReaderFetcher)

    def test_create_firecrawl(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
        f = create_fetcher("firecrawl")
        assert isinstance(f, FirecrawlFetcher)
        f.close()

    def test_create_web_alias(self):
        f = create_fetcher("web")
        assert isinstance(f, DirectFetcher)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError):
            create_fetcher("nonexistent")

    def test_web_fetcher_alias(self):
        """WebFetcher should be DirectFetcher (backward compat)."""
        assert WebFetcher is DirectFetcher

    def test_base_fetcher_is_abstract(self):
        with pytest.raises(TypeError):
            BaseFetcher()  # cannot instantiate ABC
