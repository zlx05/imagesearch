"""Command-line interface for llmscrapy."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import CrawlerConfig, LLMConfig
from .loader import discover_json_files, load_urls_from_json
from .pipeline import Pipeline


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the llmscrapy CLI.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    parser = argparse.ArgumentParser(
        description="llmscrapy — LLM-powered web scraper for metadata extraction",
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        help="Path to the JSON file containing URLs. If omitted, auto-discovers *.json in the working directory.",
    )
    parser.add_argument(
        "-n", "--max-urls",
        type=int,
        default=None,
        help="Limit the number of URLs to process (useful for dry runs).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file path (auto-generated if not specified).",
    )
    parser.add_argument(
        "-d", "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between requests (default: 0.5).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="DeepSeek model name (overrides DEEPSEEK_MODEL env var).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DeepSeek API key (overrides DEEPSEEK_API_KEY env var).",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: 3, set 1 for sequential).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--fetcher",
        type=str,
        default="direct",
        choices=["direct", "jina", "firecrawl"],
        help=(
            "Fetch method: 'direct' (HTTP + anti-crawler), "
            "'jina' (free Jina Reader API), "
            "'firecrawl' (paid, JS rendering + proxies, bypasses most walls). "
            "Default: direct."
        ),
    )
    parser.add_argument(
        "--enrich",
        type=str,
        default="both",
        choices=["none", "baidu", "playwright", "both"],
        help=(
            "Metrics enrichment mode: 'none' (no enrichment), "
            "'baidu' (Baidu API only, fast), "
            "'playwright' (Playwright DOM scan, universal), "
            "'both' (Baidu + Playwright fallback, default)."
        ),
    )
    parser.add_argument(
        "--skip-sources",
        type=str,
        nargs="*",
        default=None,
        help="Source names to skip (substring match, case-insensitive). E.g. --skip-sources 微博 小红书",
    )

    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # --- Find JSON file ---
    json_file = args.json_file
    if json_file is None:
        json_files = discover_json_files(Path.cwd())
        if not json_files:
            logger.error("No JSON files found in the current directory.")
            return 1
        json_file = str(json_files[0])
        logger.info("Auto-discovered JSON file: %s", json_file)

    if not Path(json_file).exists():
        logger.error("JSON file not found: %s", json_file)
        return 1

    # --- Configure ---
    crawler_cfg = CrawlerConfig()
    crawler_cfg.fetcher_type = args.fetcher
    crawler_cfg.enrich_mode = args.enrich
    if args.workers is not None:
        crawler_cfg.max_workers = args.workers
    llm_cfg = LLMConfig()
    if args.model:
        llm_cfg.model = args.model
    if args.api_key:
        llm_cfg.api_key = args.api_key

    if not llm_cfg.api_key:
        logger.error(
            "DeepSeek API key not set. "
            "Use --api-key or set DEEPSEEK_API_KEY environment variable."
        )
        return 1

    # --- Run ---
    pipeline = Pipeline(crawler_config=crawler_cfg, llm_config=llm_cfg)
    logger.info("Starting pipeline with model=%s", llm_cfg.model)

    try:
        results = pipeline.run_from_json(
            json_path=json_file,
            max_urls=args.max_urls,
            output_path=args.output,
            delay=args.delay,
            exclude_sources=args.skip_sources,
        )
    finally:
        pipeline.close()

    # --- Summary ---
    succeeded = sum(1 for r in results if r.succeeded)
    failed = len(results) - succeeded
    logger.info(
        "Done. %d succeeded, %d failed, %d total.",
        succeeded, failed, len(results),
    )

    if succeeded > 0:
        for r in results:
            if r.succeeded and r.metadata:
                m = r.metadata
                logger.info(
                    "  [%s] %s | author=%s | time=%s | views=%s | likes=%s",
                    r.source.id,
                    m.title[:50] if m.title else "(no title)",
                    m.author or "?",
                    m.publish_time or "?",
                    m.view_count or "-",
                    m.like_count or "-",
                )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
