"""Tests for HTML parser."""

from __future__ import annotations

from llmscrapy.parser import HTMLParser


class TestHTMLParser:
    def test_parse_extracts_title(self, sample_html):
        parser = HTMLParser()
        result = parser.parse(sample_html, url="https://example.com/test")
        assert "测试新闻标题" in result.title
        assert result.url == "https://example.com/test"

    def test_parse_removes_scripts(self, sample_html):
        parser = HTMLParser()
        result = parser.parse(sample_html)
        assert "console.log" not in result.text
        assert "tracking code" not in result.text

    def test_parse_removes_styles(self, sample_html):
        parser = HTMLParser()
        result = parser.parse(sample_html)
        assert "display: none" not in result.text

    def test_parse_removes_nav(self, sample_html):
        parser = HTMLParser()
        result = parser.parse(sample_html)
        assert "导航菜单" not in result.text

    def test_parse_extracts_body_content(self, sample_html):
        parser = HTMLParser()
        result = parser.parse(sample_html)
        assert "人工智能" in result.text
        assert "数字化转型" in result.text
        assert "阅读量" in result.text

    def test_parse_text_length_tracked(self, sample_html):
        parser = HTMLParser()
        result = parser.parse(sample_html)
        assert result.text_length == len(result.text)
        assert result.text_length > 0

    def test_parse_truncates_long_text(self):
        parser = HTMLParser()
        parser.config.max_text_length = 100
        long_html = "<html><body>" + ("<p>内容</p>" * 200) + "</body></html>"
        result = parser.parse(long_html)
        assert result.text_length <= 100

    def test_parse_empty_html(self):
        parser = HTMLParser()
        result = parser.parse("<html><body></body></html>")
        assert result.text_length == 0
        assert result.title == ""

    def test_parse_fallback_title_from_og(self):
        html = """<html><head>
        <meta property="og:title" content="OG标题">
        </head><body><p>内容</p></body></html>"""
        parser = HTMLParser()
        result = parser.parse(html)
        assert result.title == "OG标题"

    def test_parse_removes_footer(self, sample_html):
        parser = HTMLParser()
        result = parser.parse(sample_html)
        assert "版权所有" not in result.text
