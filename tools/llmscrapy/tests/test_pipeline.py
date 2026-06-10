"""Integration tests for the full pipeline."""

from __future__ import annotations

from unittest.mock import patch

from llmscrapy.models import CrawlResult, URLSource
from llmscrapy.pipeline import Pipeline


class TestPipeline:
    def test_run_url_successful_flow(
        self, sample_json_file, mock_requests_get, mock_openai_client
    ):
        """Full pipeline run with mocked external dependencies."""
        from llmscrapy.loader import load_urls_from_json

        sources = load_urls_from_json(sample_json_file, max_urls=1)
        assert len(sources) == 1

        pipeline = Pipeline()
        pipeline.extractor._client = mock_openai_client

        result = pipeline.run_url(sources[0])

        assert isinstance(result, CrawlResult)
        assert result.succeeded
        assert result.fetched.status_code == 200
        assert "人工智能" in result.parsed.text
        assert result.metadata.title != ""
        assert result.metadata.author == "张三"

        pipeline.close()

    def test_run_batch(self, sample_json_file, mock_requests_get, mock_openai_client):
        from llmscrapy.loader import load_urls_from_json

        sources = load_urls_from_json(sample_json_file, max_urls=2)
        pipeline = Pipeline()
        pipeline.extractor._client = mock_openai_client

        results = pipeline.run_batch(sources, delay=0.0)
        assert len(results) == 2
        assert all(isinstance(r, CrawlResult) for r in results)

        pipeline.close()

    def test_save_results(self, sample_json_file, mock_requests_get, mock_openai_client, tmp_path):
        from llmscrapy.loader import load_urls_from_json

        sources = load_urls_from_json(sample_json_file, max_urls=1)
        pipeline = Pipeline()
        pipeline.extractor._client = mock_openai_client
        results = pipeline.run_batch(sources, delay=0.0)

        output = tmp_path / "output.json"
        saved = pipeline.save_results(results, output_path=output)

        assert saved.exists()
        assert saved == output

        import json
        with open(saved, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["total"] == 1
        assert data["succeeded"] == 1
        assert len(data["results"]) == 1

        pipeline.close()

    def test_run_from_json(
        self, sample_json_file, mock_requests_get, mock_openai_client, tmp_path
    ):
        pipeline = Pipeline()
        pipeline.extractor._client = mock_openai_client

        output = tmp_path / "out.json"
        results = pipeline.run_from_json(
            sample_json_file, max_urls=1, output_path=output, delay=0.0
        )
        assert len(results) == 1
        assert results[0].succeeded
        assert output.exists()

        pipeline.close()

    def test_pipeline_handles_fetch_error(self, mock_openai_client):
        """Pipeline should survive fetch failures gracefully."""
        source = URLSource(
            id="err1", url="https://example.com/will-fail", source="test"
        )

        with patch("requests.Session.get") as mock_get:
            mock_get.side_effect = Exception("Network down")

            pipeline = Pipeline()
            pipeline.extractor._client = mock_openai_client

            result = pipeline.run_url(source)

        assert not result.succeeded
        assert "Fetch failed" in result.error

        pipeline.close()
