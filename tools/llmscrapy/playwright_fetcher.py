"""Playwright-based fetcher — renders full DOM and extracts metrics.

Generic across all platforms: searches the rendered visible text for
universal engagement patterns (likes, comments, shares, views, etc.)
in multiple languages. No platform-specific code.

Usage:
    python run.py --fetcher playwright -n 5
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Optional

from .config import CrawlerConfig
from .fetcher import BaseFetcher
from .models import FetchedPage

logger = logging.getLogger(__name__)

# ── Universal metrics patterns (language-agnostic) ──────────────────
# Each entry: (metric_name, [list of label patterns])
_METRIC_PATTERNS = [
    (
        "like_count",
        [
            # Label BEFORE number (e.g. "点赞 3421", "赞 3.2万")
            r"(?:点赞|赞|顶|喜欢)[：:\s]{0,3}([\d,]+\.?\d*[万kK亿]?)",
            # Label AFTER number, tight coupling (e.g. "3421次点赞", "3.2万人赞过")
            r"([\d,]+\.?\d*[万kK亿]?)(?:次点赞|人赞过|[个次人]赞|人赞)(?![a-z])",
            # English: "Likes 1.5K", "1.5K likes"
            r"(?:likes?|thumbs?.?up|upvotes?)[：:\s]{0,3}([\d,]+\.?\d*[kKmM]?)",
            r"([\d,]+\.?\d*[kKmM]?)\s?(?:likes?|thumbs?.?up)(?![a-z])",
            # Vietnamese / Japanese / Russian
            r"(?:thích|thả tim|いいね|高評価|нравится|лайк\w*)[：:\s]{0,3}([\d,]+\.?\d*[万kKkKnN]?)",
        ],
    ),
    (
        "comment_count",
        [
            r"(?:评论|回复|留言|讨论)[：:\s]{0,3}([\d,]+\.?\d*[万kK亿]?)",
            r"([\d,]+\.?\d*[万kK亿]?)(?:条评论|条回复|条留言|个讨论)(?![a-z])",
            r"(?:comments?|replies?)[：:\s]{0,3}([\d,]+\.?\d*[kKmM]?)",
            r"([\d,]+\.?\d*[kKmM]?)\s?(?:comments?|replies?)(?![a-z])",
            r"(?:bình luận|trả lời|コメント|返信|коммент\w*|отв[еe]т\w*)[：:\s]{0,3}([\d,]+\.?\d*[万kKkKnN]?)",
            # Markdown: "## 评论 2"
            r"#+\s*评论\s*(\d+)",
        ],
    ),
    (
        "repost_count",
        [
            r"(?:转发|转载|转推|转帖)[：:\s]{0,3}([\d,]+\.?\d*[万kK亿]?)",
            r"([\d,]+\.?\d*[万kK亿]?)(?:次转发|次转载|次转推)(?![a-z])",
            r"(?:reposts?|retweets?)[：:\s]{0,3}([\d,]+\.?\d*[kKmM]?)",
            r"([\d,]+\.?\d*[kKmM]?)\s?(?:reposts?|retweets?)(?![a-z])",
            r"(?:chia sẻ|đăng lại|転送|リツイート|репост\w*|ретвит\w*)[：:\s]{0,3}([\d,]+\.?\d*[万kKkKnN]?)",
        ],
    ),
    (
        "view_count",
        [
            r"(?:阅读|查看|浏览|播放|观看)[：:\s]{0,3}([\d,]+\.?\d*[万kK亿]?)",
            r"([\d,]+\.?\d*[万kK亿]?)(?:次阅读|次播放|次浏览|次观看)(?![a-z])",
            r"(?:views?|plays?)[：:\s]{0,3}([\d,]+\.?\d*[kKmM]?)",
            r"([\d,]+\.?\d*[kKmM]?)\s?(?:views?|plays?)(?![a-z])",
            r"(?:lượt xem|lượt đọc|閲覧|再生|視聴|просмотр\w*|зрител\w*)[：:\s]{0,3}([\d,]+\.?\d*[万kKkKnN]?)",
        ],
    ),
    (
        "share_count",
        [
            r"(?:分享|共享)[：:\s]{0,3}([\d,]+\.?\d*[万kK亿]?)",
            r"([\d,]+\.?\d*[万kK亿]?)(?:次分享)(?![a-z])",
            r"(?:shares?)[：:\s]{0,3}([\d,]+\.?\d*[kKmM]?)",
            r"([\d,]+\.?\d*[kKmM]?)\s?(?:shares?)(?![a-z])",
        ],
    ),
    (
        "favorite_count",
        [
            r"(?:收藏|书签)[：:\s]{0,3}([\d,]+\.?\d*[万kK亿]?)",
            r"([\d,]+\.?\d*[万kK亿]?)(?:次收藏|人收藏)(?![a-z])",
            r"(?:favorites?|bookmarks?|saves?)[：:\s]{0,3}([\d,]+\.?\d*[kKmM]?)",
        ],
    ),
]


def _parse_metric_number(raw: str) -> int | None:
    """Convert metric text to integer.
    Handles: '1.2万' → 12000, '3.4k' → 3400, '1,234' → 1234

    Filters:
    - 4-digit numbers 1900-2099 → None (likely years, not metrics)
    - Numbers < 5 without units (万/k/M) → None (likely noise)
    """
    if not raw:
        return None
    raw_stripped = raw.strip().replace(",", "")
    multiplier = 1
    has_unit = False
    if "亿" in raw_stripped:
        multiplier = 100_000_000
        raw_stripped = raw_stripped.replace("亿", "")
        has_unit = True
    elif "万" in raw_stripped:
        multiplier = 10_000
        raw_stripped = raw_stripped.replace("万", "")
        has_unit = True
    elif raw_stripped.lower().endswith("k"):
        multiplier = 1_000
        raw_stripped = raw_stripped[:-1]
        has_unit = True
    elif raw_stripped.lower().endswith("m"):
        multiplier = 1_000_000
        raw_stripped = raw_stripped[:-1]
        has_unit = True
    elif raw_stripped.lower().endswith("n"):
        multiplier = 1_000
        raw_stripped = raw_stripped[:-1]
        has_unit = True
    try:
        val = int(float(raw_stripped) * multiplier)
    except (ValueError, TypeError):
        return None
    # Filter years
    if 1900 <= val <= 2099 and not has_unit:
        return None
    # Filter 0 and 1 as likely noise (unless has unit like 1万)
    if val <= 1 and not has_unit:
        return None
    return val


def _extract_metrics_from_text(text: str) -> dict[str, int]:
    """Scan rendered page text for engagement metrics using universal patterns.

    Returns dict like {'like_count': 1234, 'comment_count': 567, ...}
    Only keys with found values are included.
    """
    results: dict[str, int] = {}

    for metric_name, patterns in _METRIC_PATTERNS:
        best: int | None = None
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw = match.group(1)
                val = _parse_metric_number(raw)
                if val is not None:
                    # Take the highest value found (avoid sidebar counts)
                    if best is None or val > best:
                        best = val
        if best is not None:
            results[metric_name] = best

    return results


# ══════════════════════════════════════════════════════════════════════

class PlaywrightFetcher(BaseFetcher):
    """Fetch webpages via Playwright headless browser.

    Renders the full DOM (including JavaScript), scrolls to trigger
    lazy-loaded content, then extracts engagement metrics from the
    visible text using universal multilingual patterns.

    Slower than API-based fetchers (~5-8s per page), but works on
    ANY platform without per-site customization.
    """

    def __init__(self, config: CrawlerConfig | None = None):
        self.config = config or CrawlerConfig()
        self._closed = False

    def fetch(self, url: str) -> FetchedPage:
        """Fetch via Playwright — renders full DOM, extracts metrics."""
        fetched_at = datetime.now()
        try:
            result = asyncio.run(self._fetch_async(url))
        except RuntimeError:
            # If already in async context
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self._fetch_async(url))
        except ImportError:
            return FetchedPage(
                url=url,
                status_code=0,
                error="Playwright not installed. Run: pip install playwright && python -m playwright install chromium",
                fetched_at=fetched_at,
            )
        except Exception as e:
            return FetchedPage(
                url=url,
                status_code=0,
                error=f"Playwright error: {e}",
                fetched_at=fetched_at,
            )

        if result is None:
            return FetchedPage(
                url=url,
                status_code=0,
                error="Playwright returned no content",
                fetched_at=fetched_at,
            )

        return FetchedPage(
            url=url,
            html=result["html"],
            status_code=result.get("status", 200),
            fetched_at=fetched_at,
        )

    async def _fetch_async(self, url: str) -> dict | None:
        """Async Playwright fetch with stealth + metrics extraction."""
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )
            ctx = await browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                locale="zh-CN",
            )
            page = await ctx.new_page()

            # ── Stealth ─────────────────────────────────────────
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
                Object.defineProperty(navigator, 'languages', {get: () => ['zh-CN','zh','en']});
                window.chrome = {runtime: {}};
            """)

            try:
                # ── Navigate ────────────────────────────────────
                # Warm cookies for Baidu sites
                if "baidu.com" in url:
                    await page.goto(
                        "https://www.baidu.com/",
                        wait_until="networkidle",
                        timeout=15000,
                    )
                    await asyncio.sleep(0.5)

                await page.goto(
                    url,
                    wait_until="networkidle",
                    timeout=30000,
                )

                # ── Wait for dynamic content ────────────────────
                await asyncio.sleep(3)

                # ── Scroll to trigger lazy loading ──────────────
                for _ in range(3):
                    await page.evaluate(
                        "window.scrollTo(0, document.body.scrollHeight)"
                    )
                    await asyncio.sleep(1)

                # ── Get page title ──────────────────────────────
                title = await page.title()

                # ── Get visible text ────────────────────────────
                visible_text = await page.inner_text("body")

                # ── Extract universal metrics ────────────────────
                metrics = _extract_metrics_from_text(visible_text)
                logger.debug(
                    "Playwright metrics from %s: %s",
                    url[:60],
                    metrics,
                )

                # ── Build HTML with metrics as meta tags ────────
                metrics_meta = ""
                for k, v in metrics.items():
                    metrics_meta += (
                        f'<meta name="pw:{k}" content="{v}">\n'
                    )

                html = (
                    f"<html><head>"
                    f"{metrics_meta}"
                    f"<title>{title}</title>"
                    f"</head><body>{visible_text}</body></html>"
                )

                return {"html": html, "status": 200}

            except Exception as e:
                logger.warning("Playwright fetch error for %s: %s", url[:60], e)
                return None
            finally:
                await browser.close()

    def close(self) -> None:
        self._closed = True
