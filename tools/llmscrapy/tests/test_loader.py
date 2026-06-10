"""Tests for URL loader."""

from __future__ import annotations

import pytest

from llmscrapy.loader import discover_json_files, load_urls_from_json
from llmscrapy.models import URLSource


class TestLoadUrlsFromJson:
    def test_loads_all_urls(self, sample_json_file):
        sources = load_urls_from_json(sample_json_file)
        # 3 nodes, but one has empty url -> 2 valid sources
        assert len(sources) == 2
        assert all(isinstance(s, URLSource) for s in sources)

    def test_respects_max_urls(self, sample_json_file):
        sources = load_urls_from_json(sample_json_file, max_urls=1)
        assert len(sources) == 1

    def test_url_fields_parsed_correctly(self, sample_json_file):
        sources = load_urls_from_json(sample_json_file)
        first = sources[0]
        assert first.id == "r1"
        assert first.url == "https://example.com/article1"
        assert first.source == "example.com"
        assert first.similarity == 0.85

    def test_possible_duplicate_flag(self, sample_json_file):
        sources = load_urls_from_json(sample_json_file)
        assert sources[0].possible_duplicate is False
        assert sources[1].possible_duplicate is True

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_urls_from_json(tmp_path / "nonexistent.json")

    def test_max_urls_zero_returns_all(self, sample_json_file):
        sources = load_urls_from_json(sample_json_file, max_urls=0)
        assert len(sources) == 2

    def test_max_urls_none_returns_all(self, sample_json_file):
        sources = load_urls_from_json(sample_json_file, max_urls=None)
        assert len(sources) == 2


class TestDiscoverJsonFiles:
    def test_discovers_json_files(self, tmp_path):
        (tmp_path / "a.json").touch()
        (tmp_path / "b.json").touch()
        (tmp_path / "c.txt").touch()
        files = discover_json_files(tmp_path)
        assert len(files) == 2
        assert all(f.suffix == ".json" for f in files)

    def test_empty_directory(self, tmp_path):
        files = discover_json_files(tmp_path)
        assert files == []
