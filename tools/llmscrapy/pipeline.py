"""Pipeline orchestrating the full crawl→parse→extract workflow."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import CrawlerConfig, LLMConfig
from .extractor import LLMExtractor
from .fetcher import BaseFetcher, create_fetcher
from .loader import discover_json_files, load_urls_from_json
from .models import CrawlResult, URLSource
from .parser import HTMLParser

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the end-to-end crawling and extraction pipeline."""

    def __init__(
        self,
        crawler_config: CrawlerConfig | None = None,
        llm_config: LLMConfig | None = None,
        fetcher: BaseFetcher | None = None,
    ):
        self.crawler_config = crawler_config or CrawlerConfig()
        self.llm_config = llm_config or LLMConfig()
        self.fetcher = fetcher or create_fetcher(
            self.crawler_config.fetcher_type, self.crawler_config
        )
        self.parser = HTMLParser(self.crawler_config)
        self.extractor = LLMExtractor(self.llm_config)

    # ── Metrics enrichment ───────────────────────────────────────

    def _enrich_metrics(self, html: str, url: str, mode: str) -> str:
        """Inject interaction metrics as meta tags into HTML.

        Modes:
        - "none": no enrichment (fastest)
        - "baidu": Baidu API only (fast, baidu domains only)
        - "playwright": Playwright DOM scan (universal, slower)
        - "both": Baidu first, Playwright fallback for non-baidu (default)
        """
        if mode == "none":
            return html

        enriched = html

        # ── Phase 1: Baidu API (fast, precise) ─────────────────
        if mode in ("baidu", "both"):
            try:
                from .stats_enricher import fetch_baidu_stats_sync, is_baidu_domain
                if is_baidu_domain(url):
                    stats = fetch_baidu_stats_sync(url)
                    if stats:
                        logger.info(
                            "  Baidu stats: likes=%s comments=%s",
                            stats.get("like_count"),
                            stats.get("comment_count"),
                        )
                        enriched = self._inject_meta(enriched, "baidu", stats)
            except ImportError:
                logger.debug("Playwright not available for Baidu stats")
            except Exception as e:
                logger.warning("Baidu stats enrichment failed: %s", e)

        # ── Phase 2: Playwright DOM scan (universal, slower) ───
        if mode in ("playwright", "both"):
            # In "both" mode, skip Playwright if we already got Baidu stats
            # and the URL is a Baidu domain (already covered)
            if mode == "both":
                try:
                    from .stats_enricher import is_baidu_domain
                    if is_baidu_domain(url):
                        return enriched  # Baidu API already covered
                except ImportError:
                    pass

            try:
                from .playwright_fetcher import _extract_metrics_from_text
                import asyncio
                from playwright.async_api import async_playwright

                async def _pw_scan() -> dict | None:
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(
                            headless=True,
                            args=["--disable-blink-features=AutomationControlled",
                                  "--no-sandbox", "--disable-gpu"],
                        )
                        ctx = await browser.new_context(
                            viewport={"width": 1280, "height": 900},
                            locale="zh-CN",
                            user_agent=(
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36"
                            ),
                        )
                        page = await ctx.new_page()
                        await page.add_init_script(
                            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
                            "window.chrome={runtime:{}};"
                        )
                        if "baidu.com" in url:
                            await page.goto("https://www.baidu.com/",
                                wait_until="networkidle", timeout=10000)
                            await asyncio.sleep(0.3)
                        await page.goto(url, wait_until="networkidle", timeout=20000)
                        await asyncio.sleep(2)
                        for _ in range(2):
                            await page.evaluate(
                                "window.scrollTo(0, document.body.scrollHeight)"
                            )
                            await asyncio.sleep(0.8)
                        text = await page.inner_text("body")
                        await browser.close()
                        return _extract_metrics_from_text(text)

                pw_stats = asyncio.run(_pw_scan())
                if pw_stats:
                    logger.info(
                        "  Playwright enrich: %s", pw_stats
                    )
                    enriched = self._inject_meta(enriched, "pw", pw_stats)
            except ImportError:
                logger.debug("Playwright not installed, skipping DOM enrichment")
            except Exception as e:
                logger.warning("Playwright enrichment failed: %s", e)

        return enriched

    @staticmethod
    def _inject_meta(html: str, prefix: str, data: dict) -> str:
        """Inject key-value pairs as <meta> tags into HTML <head>."""
        meta = "".join(
            f'<meta name="{prefix}:{k}" content="{v}">\n'
            for k, v in data.items()
        )
        if "</head>" in html:
            return html.replace("</head>", f"{meta}</head>", 1)
        return f"<html><head>{meta}</head><body>{html}</body></html>"

    # ── Core pipeline ────────────────────────────────────────────

    def run_url(self, source: URLSource) -> CrawlResult:
        """Run the full pipeline on a single URL.

        Args:
            source: URLSource object with url and metadata.

        Returns:
            CrawlResult with all intermediate and final data.
        """
        # Step 1: Fetch
        fetched = self.fetcher.fetch(source.url)
        if fetched.error:
            return CrawlResult(
                source=source,
                fetched=fetched,
                parsed=None,
                metadata=None,
                error=f"Fetch failed: {fetched.error}",
            )

        # ── Step 1.5: Metrics enrichment (optional) ─────────────
        html = self._enrich_metrics(
            fetched.html, source.url, self.crawler_config.enrich_mode
        )

        # Step 2: Parse
        parsed = self.parser.parse(html, url=source.url)

        # Step 3: Extract via LLM
        metadata = self.extractor.extract(
            text=parsed.text,
            title_hint=parsed.title,
            target_image_url=source.image_url,
            source_platform=source.source,
            head_metadata=parsed.head_metadata,
        )

        return CrawlResult(
            source=source,
            fetched=fetched,
            parsed=parsed,
            metadata=metadata,
            error="",
        )

    def run_batch(
        self,
        sources: List[URLSource],
        delay: float = 0.0,
        workers: int | None = None,
    ) -> List[CrawlResult]:
        """Run the pipeline on a batch of URLs.

        Args:
            sources: List of URLSource objects.
            delay: Seconds between requests (ignored when workers>1).
            workers: Parallel workers. Default: config.max_workers. 1 = sequential.

        Returns:
            List of CrawlResult objects (preserves input order).
        """
        n_workers = workers if workers is not None else self.crawler_config.max_workers

        if n_workers <= 1:
            return self._run_sequential(sources, delay)

        logger.info("Processing %d URLs with %d parallel workers", len(sources), n_workers)
        results_map: dict[int, CrawlResult] = {}

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._run_one, source, idx, len(sources)): idx
                for idx, source in enumerate(sources)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results_map[idx] = future.result()
                except Exception as e:
                    logger.error("Worker failed for %s: %s", sources[idx].url, e)
                    results_map[idx] = CrawlResult(source=sources[idx], error=f"Worker error: {e}")

        return [results_map[i] for i in range(len(sources))]

    def _run_one(self, source: URLSource, idx: int, total: int) -> CrawlResult:
        """Process a single URL (called from worker threads)."""
        logger.info("[%d/%d] %s", idx + 1, total, source.url)
        try:
            result = self.run_url(source)
            if result.succeeded and result.metadata:
                logger.info("  ✓ [%d/%d] %s | author=%s",
                            idx + 1, total,
                            result.metadata.title[:30] if result.metadata.title else "?",
                            result.metadata.author or "?")
            else:
                logger.warning("  ✗ [%d/%d] %s", idx + 1, total, result.error or "?")
            return result
        except Exception as e:
            logger.error("[%d/%d] Pipeline error: %s", idx + 1, total, e)
            return CrawlResult(source=source, error=f"Pipeline error: {e}")

    def _run_sequential(self, sources: List[URLSource], delay: float) -> List[CrawlResult]:
        """Fallback: process URLs one at a time."""
        import time as _time
        results: List[CrawlResult] = []
        for i, source in enumerate(sources):
            logger.info("[%d/%d] Processing: %s", i + 1, len(sources), source.url)
            try:
                result = self.run_url(source)
                results.append(result)
                if result.succeeded and result.metadata:
                    logger.info("  ✓ title=%s, author=%s",
                                result.metadata.title,
                                result.metadata.author)
                else:
                    logger.warning("  ✗ %s", result.error or "Unknown error")
            except Exception as e:
                logger.exception("Unexpected error processing %s", source.url)
                results.append(CrawlResult(source=source, error=f"Pipeline error: {e}"))
            if i < len(sources) - 1 and delay > 0:
                _time.sleep(delay)
        return results

    def save_results(
        self,
        results: List[CrawlResult],
        output_path: str | Path | None = None,
    ) -> Path:
        """Save crawl results as JSON.

        Args:
            results: List of CrawlResult objects.
            output_path: Path to save the JSON. Auto-generated if None.

        Returns:
            The Path where results were saved.
        """
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.crawler_config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"crawl_result_{ts}.json"
        else:
            output_path = Path(output_path)

        data = {
            "timestamp": datetime.now().isoformat(),
            "total": len(results),
            "succeeded": sum(1 for r in results if r.succeeded),
            "failed": sum(1 for r in results if not r.succeeded),
            "results": [r.model_dump() for r in results],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info("Results saved to %s", output_path)
        return output_path

    def run_from_json(
        self,
        json_path: str | Path,
        max_urls: int | None = None,
        output_path: str | Path | None = None,
        delay: float = 0.5,
        exclude_sources: list[str] | None = None,
    ) -> List[CrawlResult]:
        """Convenience method: load URLs from JSON and run the full pipeline.

        Args:
            json_path: Path to the JSON file with URL entries.
            max_urls: Limit number of URLs to process.
            output_path: Path to save results.
            delay: Seconds between requests.
            exclude_sources: Source names to skip (substring match).

        Returns:
            List of CrawlResult objects.
        """
        sources = load_urls_from_json(
            json_path, max_urls=max_urls, exclude_sources=exclude_sources
        )
        logger.info("Loaded %d URLs from %s", len(sources), json_path)

        results = self.run_batch(
            sources, delay=delay, workers=self.crawler_config.max_workers
        )
        self.save_results(results, output_path=output_path)

        return results

    def close(self) -> None:
        """Clean up resources."""
        self.fetcher.close()
