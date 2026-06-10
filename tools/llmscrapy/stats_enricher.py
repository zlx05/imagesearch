"""Baidu interaction stats enricher — uses Playwright to capture XHR stats API.

For baijiahao.baidu.com / mbd.baidu.com pages, the like/view/comment
counts are loaded via JavaScript XHR and NOT present in Firecrawl output.
This module extracts the article nid from the rendered page, calls the
internal Baidu stats API, and returns the interaction metrics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

logger = logging.getLogger(__name__)


async def _extract_baidu_stats_async(url: str) -> dict | None:
    """Use Playwright to extract nid and fetch Baidu interaction stats.

    Args:
        url: A baijiahao.baidu.com or mbd.baidu.com article URL.

    Returns:
        dict with like_count, comment_count, view_count, etc., or None.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning("playwright not installed, skipping Baidu stats")
        return None

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            ctx = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                locale="zh-CN",
            )
            page = await ctx.new_page()

            # Stealth
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                window.chrome = {runtime: {}};
            """)

            # Collect stats from XHR responses
            stats_data: dict | None = None

            async def handle_response(response):
                nonlocal stats_data
                if stats_data:
                    return
                resp_url = response.url
                if "pclanding" in resp_url and "feedapi" in resp_url:
                    try:
                        body = await response.text()
                        json_match = re.search(r"\{.*\}", body, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group(0))
                            inner = data.get("data", data)
                            like = inner.get("like", {})
                            if isinstance(like, dict):
                                stats_data = {
                                    "like_count": _to_int(
                                        like.get("count", "0")
                                    ),
                                    "comment_count": _to_int(
                                        inner.get("commentNum", "0")
                                    ),
                                    "is_liked": like.get("is_like", "0"),
                                }
                    except Exception:
                        pass

            page.on("response", handle_response)

            # First get BAIDUID cookie
            await page.goto(
                "https://www.baidu.com/",
                wait_until="networkidle",
                timeout=15000,
            )
            await asyncio.sleep(0.5)

            # Navigate to article
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for XHR to complete
            for _ in range(8):
                if stats_data:
                    break
                await asyncio.sleep(0.5)

            await browser.close()
            return stats_data

    except Exception as e:
        logger.debug("Baidu stats Playwright error: %s", e)
        return None


def _to_int(val: str) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def fetch_baidu_stats_sync(url: str) -> dict | None:
    """Synchronous wrapper for Baidu stats extraction.

    Call this after Firecrawl fetches the page content.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running in async context — schedule on a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    lambda: asyncio.run(_extract_baidu_stats_async(url))
                )
                return future.result(timeout=20)
        return loop.run_until_complete(_extract_baidu_stats_async(url))
    except RuntimeError:
        return asyncio.run(_extract_baidu_stats_async(url))


def is_baidu_domain(url: str) -> bool:
    """Check if the URL is a Baidu-owned domain that supports the stats API."""
    return any(
        d in url.lower()
        for d in ["baijiahao.baidu.com", "mbd.baidu.com"]
    )
