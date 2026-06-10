"""Fetch webpage content — pluggable fetcher architecture.

Supports:
- DirectFetcher: direct HTTP with Cookie jar, Referer chain, anti-crawler headers
- JinaReaderFetcher: uses Jina AI Reader API (free tier, clean markdown output)
- FirecrawlFetcher: paid API with JS rendering + residential proxies, bypasses most walls
- WebFetcher: alias for DirectFetcher (backward compatibility)
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import CrawlerConfig
from .models import FetchedPage

# ── HTTP 代理支持 ────────────────────────────────────────────────────
def _get_proxies() -> dict | None:
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("http_proxy") or os.environ.get("https_proxy")
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}

# ── User-Agent pools (rotated randomly) ──────────────────────────────
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
]

_MOBILE_USER_AGENTS = [
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-S9080) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36",
]


class BaseFetcher(ABC):
    """Abstract interface for webpage fetchers."""

    @abstractmethod
    def fetch(self, url: str) -> FetchedPage:
        """Fetch a single URL and return a FetchedPage."""
        ...

    def fetch_batch(self, urls: list[str], delay: float = 0.5) -> list[FetchedPage]:
        results: list[FetchedPage] = []
        for i, url in enumerate(urls):
            if i > 0 and delay > 0:
                time.sleep(delay)
            results.append(self.fetch(url))
        return results

    def close(self) -> None:
        """Release resources (no-op by default)."""
        pass


# ══════════════════════════════════════════════════════════════════════
# Direct HTTP fetcher with anti-crawler enhancements
# ══════════════════════════════════════════════════════════════════════

class DirectFetcher(BaseFetcher):
    """Fetches webpages directly via HTTP with anti-crawler measures.

    Features:
    - Cookie persistence (visits homepage first for site cookies)
    - Rotating User-Agent pool
    - Realistic browser headers (Sec-*, Referer, etc.)
    - Randomised inter-request delays
    - Auto encoding detection for Chinese sites
    """

    def __init__(self, config: CrawlerConfig | None = None):
        self.config = config or CrawlerConfig()
        self.session = self._build_session()
        self._cookie_cache: dict[str, bool] = {}  # tracks which domains we've warmed

    # ── Session setup ───────────────────────────────────────────────
    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.proxies.update(_get_proxies() or {})

        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        self._rotate_headers(session)
        return session

    def _rotate_headers(
        self, session: requests.Session | None = None, use_mobile: bool = False
    ) -> dict[str, str]:
        """Build a fresh set of browser-like headers."""
        pool = _MOBILE_USER_AGENTS if use_mobile else _USER_AGENTS
        ua = random.choice(pool)
        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7",
            "Accept-Encoding": "gzip, deflate",
            "Cache-Control": "no-cache",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        }
        if session is not None:
            session.headers.update(headers)
        return headers

    # ── Cookie warming ──────────────────────────────────────────────
    def _warm_cookies(self, url: str) -> None:
        """Visit a domain's homepage to acquire cookies before scraping.

        For baidu.com subdomains, we specifically visit www.baidu.com first
        to get a BAIDUID cookie, which is the key to bypassing their
        anti-crawler verification.
        """
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain in self._cookie_cache:
            return

        try:
            # For Baidu properties, the BAIDUID cookie from www.baidu.com
            # is the critical credential for all *.baidu.com subdomains
            netloc_lower = parsed.netloc.lower()
            if "baidu.com" in netloc_lower:
                self.session.get(
                    "https://www.baidu.com/",
                    timeout=10,
                    allow_redirects=True,
                    headers={
                        "User-Agent": random.choice(_MOBILE_USER_AGENTS),
                        "Accept": "text/html,application/xhtml+xml",
                        "Accept-Language": "zh-CN,zh;q=0.9",
                    },
                )
            else:
                self.session.get(
                    domain,
                    timeout=10,
                    allow_redirects=True,
                    headers={"Referer": "https://www.google.com/"},
                )
        except Exception:
            pass  # warming is best-effort
        finally:
            self._cookie_cache[domain] = True

    def _build_referer(self, url: str) -> str:
        """Build a plausible Referer for the request."""
        parsed = urlparse(url)
        # Pretend we came from a search engine or the site homepage
        if random.random() < 0.5:
            return "https://www.baidu.com/s?wd=" + parsed.path
        return f"{parsed.scheme}://{parsed.netloc}/"

    # ── Core fetch logic ────────────────────────────────────────────
    def fetch(self, url: str) -> FetchedPage:
        """Fetch a single URL."""
        fetched_at = datetime.now()

        # Detect Baidu-owned domains → use mobile UA + special cookie flow
        parsed = urlparse(url)
        is_baidu = "baidu.com" in parsed.netloc.lower()

        # Refresh headers per request (rotates UA, mobile for Baidu)
        self._rotate_headers(self.session, use_mobile=is_baidu)

        # Warm cookies on first visit to a domain
        if self.config.warm_cookies:
            self._warm_cookies(url)

        # Small random jitter to avoid looking like a bot
        time.sleep(random.uniform(0, 0.3))

        headers = {
            "Referer": self._build_referer(url),
        }

        try:
            resp = self.session.get(
                url,
                timeout=self.config.request_timeout,
                allow_redirects=True,
                headers=headers,
            )
            resp.raise_for_status()

            if resp.apparent_encoding:
                resp.encoding = resp.apparent_encoding

            html = resp.text

            # Detect anti-crawler wall
            wall_type = _detect_anti_crawler_wall(html)
            if wall_type:
                return FetchedPage(
                    url=url,
                    html=html,
                    status_code=resp.status_code,
                    error=f"Anti-crawler wall detected: {wall_type}",
                    fetched_at=fetched_at,
                )

            return FetchedPage(
                url=url,
                html=html,
                status_code=resp.status_code,
                fetched_at=fetched_at,
            )

        except requests.exceptions.Timeout as e:
            return FetchedPage(url=url, status_code=0, error=f"Timeout: {e}", fetched_at=fetched_at)
        except requests.exceptions.ConnectionError as e:
            return FetchedPage(url=url, status_code=0, error=f"Connection error: {e}", fetched_at=fetched_at)
        except requests.exceptions.HTTPError as e:
            return FetchedPage(
                url=url,
                html=getattr(e.response, "text", ""),
                status_code=getattr(e.response, "status_code", 0),
                error=f"HTTP error: {e}",
                fetched_at=fetched_at,
            )
        except Exception as e:
            return FetchedPage(url=url, status_code=0, error=f"Unexpected error: {e}", fetched_at=fetched_at)

    def close(self) -> None:
        self.session.close()


# ── Anti-crawler wall detection ─────────────────────────────────────

def _detect_anti_crawler_wall(html: str) -> str | None:
    """Check whether the returned HTML is an anti-crawler / CAPTCHA page."""
    checks = [
        ("百度安全验证", "Baidu security verification"),
        ("百度验证", "Baidu CAPTCHA"),
        ("验证码", "Generic CAPTCHA"),
        ("captcha", "CAPTCHA"),
        ("请点击下方验证", "Click verification"),
        ("请完成以下验证", "Verification required"),
        ("您的访问被拦截", "Access blocked"),
        ("安全检测", "Security check"),
        ("人机验证", "Human verification"),
        ("请稍后重试", "Rate limited"),
    ]
    for keyword, label in checks:
        if keyword in html:
            return label
    return None


# ══════════════════════════════════════════════════════════════════════
# Jina AI Reader API fetcher (free tier, no key needed)
# ══════════════════════════════════════════════════════════════════════

class JinaReaderFetcher(BaseFetcher):
    """Fetch webpages via Jina AI Reader API.

    Jina Reader returns clean, LLM-ready text from any URL.
    Base URL: https://r.jina.ai/

    Free tier: ~1000 requests/month, no API key needed.
    For higher volume, set JINA_API_KEY env var.
    """

    JINA_BASE = "https://r.jina.ai/"

    def __init__(self, config: CrawlerConfig | None = None):
        self.config = config or CrawlerConfig()
        self.session = requests.Session()
        self.session.proxies.update(_get_proxies() or {})
        self.session.headers.update({
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept": "application/json",  # JSON mode → structured response
            "X-Return-Format": "markdown", # but content in markdown
        })
        # Optional API key for higher rate limits
        api_key = os.environ.get("JINA_API_KEY", "")
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def fetch(self, url: str) -> FetchedPage:
        """Fetch via Jina Reader API.

        Jina returns JSON: {"code": 200, "data": {"title": "...", "content": "..."}}
        """
        fetched_at = datetime.now()
        jina_url = urljoin(self.JINA_BASE, url)  # https://r.jina.ai/<url>

        try:
            resp = self.session.get(
                jina_url,
                timeout=max(self.config.request_timeout, 60),
            )
            resp.raise_for_status()

            # Force UTF-8 to handle Chinese characters correctly
            resp.encoding = "utf-8"

            data = resp.json()
            # Jina wraps result in "data" envelope
            inner = data.get("data", data)
            title = inner.get("title", "")
            content = inner.get("content", "")

            # Check if Jina also hit an anti-crawler wall
            wall_type = _detect_anti_crawler_wall(title + " " + content)
            if wall_type:
                return FetchedPage(
                    url=url,
                    html=f"<html><body>{content}</body></html>",
                    status_code=resp.status_code,
                    error=f"Anti-crawler wall (via Jina): {wall_type}",
                    fetched_at=fetched_at,
                )

            html = f"<html><head><title>{title}</title></head><body>{content}</body></html>"

            return FetchedPage(
                url=url,
                html=html,
                status_code=resp.status_code,
                fetched_at=fetched_at,
            )

        except requests.exceptions.Timeout as e:
            return FetchedPage(url=url, status_code=0, error=f"Jina timeout: {e}", fetched_at=fetched_at)
        except requests.exceptions.HTTPError as e:
            return FetchedPage(
                url=url,
                html=getattr(e.response, "text", ""),
                status_code=getattr(e.response, "status_code", 0),
                error=f"Jina HTTP error: {e}",
                fetched_at=fetched_at,
            )
        except Exception as e:
            return FetchedPage(url=url, status_code=0, error=f"Jina error: {e}", fetched_at=fetched_at)

    def close(self) -> None:
        self.session.close()


# ══════════════════════════════════════════════════════════════════════
# Firecrawl — paid API with JS rendering + residential proxies
# ══════════════════════════════════════════════════════════════════════

class FirecrawlFetcher(BaseFetcher):
    """Fetch webpages via Firecrawl API (firecrawl.dev).

    Firecrawl renders JavaScript, rotates residential proxies, and
    bypasses most anti-bot walls — including Baidu's security check.

    Pricing: from $19/mo (3000 credits), 1 credit ≈ 1 page scrape.
    Requires FIRECRAWL_API_KEY env var.

    API docs: https://docs.firecrawl.dev
    """

    FIRECRAWL_BASE = "https://api.firecrawl.dev/v1/scrape"

    def __init__(self, config: CrawlerConfig | None = None):
        self.config = config or CrawlerConfig()
        self.api_key = os.environ.get("FIRECRAWL_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "FirecrawlFetcher requires FIRECRAWL_API_KEY environment variable. "
                "Get one at https://firecrawl.dev"
            )
        self.session = requests.Session()
        self.session.proxies.update(_get_proxies() or {})
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def fetch(self, url: str) -> FetchedPage:
        """Scrape via Firecrawl API with JS rendering."""
        fetched_at = datetime.now()

        payload = {
            "url": url,
            "formats": ["html", "markdown"],
            "onlyMainContent": False,  # keep header (author/date) and metrics area
            "waitFor": 1000,  # ms to wait for JS render
        }

        try:
            resp = self.session.post(
                self.FIRECRAWL_BASE,
                json=payload,
                timeout=max(self.config.request_timeout, 90),
                proxies=self.session.proxies or _get_proxies() or None,
            )

            if resp.status_code == 402:
                return FetchedPage(
                    url=url,
                    status_code=402,
                    error="Firecrawl: payment required / credits exhausted",
                    fetched_at=fetched_at,
                )

            resp.raise_for_status()
            data = resp.json()

            if not data.get("success"):
                return FetchedPage(
                    url=url,
                    status_code=resp.status_code,
                    error=f"Firecrawl failed: {data.get('error', 'unknown')}",
                    fetched_at=fetched_at,
                )

            inner = data.get("data", {})
            fc_metadata = inner.get("metadata", {})

            # Prefer HTML (preserves meta tags, JSON-LD, timestamps). Fallback to markdown.
            raw_html = inner.get("html", "") or inner.get("markdown", "") or inner.get("content", "")
            raw_md = inner.get("markdown", "")

            title = fc_metadata.get("title") or fc_metadata.get("og:title") or ""

            # Inject Firecrawl metadata as <meta> tags into the HTML so the
            # parser can pick them up before stripping.
            meta_tags = ""
            if fc_metadata:
                for key, val in fc_metadata.items():
                    if val and isinstance(val, str):
                        safe_val = val.replace('"', '&quot;')
                        meta_tags += f'<meta name="fc:{key}" content="{safe_val}">\n'

            # ── Try to fetch Baidu interaction stats via internal API ──
            # The baijiahao page HTML contains an embedded nid (news ID).
            # We can use it to call mbd.baidu.com's stats endpoint.
            baidu_stats = _fetch_baidu_stats(raw_html, self.session)
            if baidu_stats:
                meta_tags += (
                    f'<meta name="baidu:likeCount" content="{baidu_stats["like_count"]}">\n'
                    f'<meta name="baidu:commentCount" content="{baidu_stats["comment_count"]}">\n'
                    f'<meta name="baidu:isLiked" content="{baidu_stats["is_liked"]}">\n'
                )

            # Build final HTML: use raw HTML if available, inject meta block
            if raw_html.strip().startswith("<"):
                # Already HTML — inject meta into <head>
                html = raw_html.replace(
                    "</head>", f"{meta_tags}</head>", 1
                )
                if "</head>" not in raw_html and "<html" in raw_html.lower():
                    html = raw_html.replace(
                        "<html", f"<html><head>{meta_tags}</head>", 1
                    )
                if meta_tags and "</head>" not in html:
                    html = f"<html><head>{meta_tags}<title>{title}</title></head><body>{raw_html}</body></html>"
            else:
                # Markdown — wrap with meta tags
                html = f"<html><head>{meta_tags}<title>{title}</title></head><body>{raw_md or raw_html}</body></html>"

            # Also check Firecrawl output for anti-crawler walls (rare, but possible)
            wall_type = _detect_anti_crawler_wall(html)
            if wall_type:
                return FetchedPage(
                    url=url,
                    html=html,
                    status_code=resp.status_code,
                    error=f"Anti-crawler wall (via Firecrawl): {wall_type}",
                    fetched_at=fetched_at,
                )

            return FetchedPage(
                url=url,
                html=html,
                status_code=resp.status_code,
                fetched_at=fetched_at,
            )

        except requests.exceptions.Timeout as e:
            return FetchedPage(url=url, status_code=0, error=f"Firecrawl timeout: {e}", fetched_at=fetched_at)
        except requests.exceptions.HTTPError as e:
            return FetchedPage(
                url=url,
                html=getattr(e.response, "text", ""),
                status_code=getattr(e.response, "status_code", 0),
                error=f"Firecrawl HTTP error: {e}",
                fetched_at=fetched_at,
            )
        except Exception as e:
            return FetchedPage(url=url, status_code=0, error=f"Firecrawl error: {e}", fetched_at=fetched_at)

    def close(self) -> None:
        self.session.close()


# ══════════════════════════════════════════════════════════════════════
# Baidu interaction stats API helper
# ══════════════════════════════════════════════════════════════════════

def _fetch_baidu_stats(html: str, session: requests.Session) -> dict | None:
    """Extract nid from baijiahao HTML and call the internal stats API.

    Returns dict with like_count, comment_count, is_liked, or None.
    """
    # Search for embedded nid in script tags or JSON
    nid_match = re.search(r'"nid"\s*:\s*"(\d+)"', html)
    if not nid_match:
        # Try alternative pattern: nid=news_XXXX
        nid_match = re.search(r'nid[=:]"?news_(\d+)', html)
    if not nid_match:
        return None

    nid = nid_match.group(1)
    stats_url = "https://mbd.baidu.com/feedapi/v1/newsserver/api/pclanding"

    try:
        resp = session.get(
            stats_url,
            params={"nid": f"news_{nid}"},
            headers={
                "User-Agent": random.choice(_USER_AGENTS),
                "Referer": "https://baijiahao.baidu.com/",
                "Accept": "application/json, */*",
            },
            timeout=10,
        )
        resp.raise_for_status()

        # Response may be JSON or JSONP
        text = resp.text.strip()
        json_match = re.search(r'\(?(\{.*\})\)?', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            data = json.loads(text)

        inner = data.get("data", data)
        like = inner.get("like", {})
        if isinstance(like, dict):
            return {
                "like_count": like.get("count", "0"),
                "is_liked": like.get("is_like", "0"),
                "comment_count": inner.get("commentNum", "0"),
            }
    except Exception:
        pass

    return None


# ══════════════════════════════════════════════════════════════════════
# Factory / backward compatibility
# ══════════════════════════════════════════════════════════════════════

def create_fetcher(fetcher_type: str = "direct", config: CrawlerConfig | None = None) -> BaseFetcher:
    """Factory: create a fetcher by name.

    Args:
        fetcher_type: "direct" | "jina" | "firecrawl" | "web"
        config: CrawlerConfig (optional).
    """
    cfg = config or CrawlerConfig()
    _map: dict[str, type[BaseFetcher]] = {
        "direct": DirectFetcher,
        "jina": JinaReaderFetcher,
        "firecrawl": FirecrawlFetcher,
        "web": DirectFetcher,   # backward compat alias
    }
    cls = _map.get(fetcher_type.lower())
    if cls is None:
        raise ValueError(
            f"Unknown fetcher type: {fetcher_type}. "
            f"Choose from: {list(_map.keys())}"
        )
    return cls(cfg)


# Backward-compatible alias
WebFetcher = DirectFetcher
