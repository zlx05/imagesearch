"""Parse HTML into clean text for LLM consumption.

Extracts structured metadata from <head> before stripping HTML tags,
so the extractor can use both clean text and structured hints.
"""

from __future__ import annotations

import json
import re

from bs4 import BeautifulSoup

from .config import CrawlerConfig
from .models import ParsedPage


class HTMLParser:
    """Parses HTML and extracts clean, readable text + head metadata."""

    # Tags to remove entirely before text extraction
    REMOVE_TAGS = [
        "script", "style", "nav", "footer", "header",
        "aside", "noscript", "iframe", "form", "button",
        "svg", "canvas", "video", "audio", "object",
        "embed", "input", "textarea", "select",
    ]

    # Meta tag names/properties to collect from <head>
    META_KEYS = [
        # Standard
        "description", "keywords", "author",
        # OpenGraph
        "og:title", "og:description", "og:image", "og:url",
        "og:site_name", "og:type", "og:published_time",
        # Article
        "article:published_time", "article:modified_time",
        "article:author", "article:publisher", "article:tag",
        # Twitter
        "twitter:title", "twitter:description", "twitter:image",
        # Other
        "published_time", "date", "pubdate", "publishdate",
        "dc:date", "dcterms:issued", "dcterms:modified",
        # Firecrawl-injected
        "fc:title", "fc:description", "fc:author",
        "fc:publishedAt", "fc:modifiedAt", "fc:sourceURL",
        "fc:pageStatusCode", "fc:language",
        # Baidu stats API-injected
        "baidu:likeCount", "baidu:commentCount", "baidu:isLiked",
        # Playwright universal metrics-injected
        "pw:like_count", "pw:comment_count", "pw:repost_count",
        "pw:view_count", "pw:share_count", "pw:favorite_count",
    ]

    def __init__(self, config: CrawlerConfig | None = None):
        self.config = config or CrawlerConfig()

    def parse(self, html: str, url: str = "") -> ParsedPage:
        """Parse HTML string into a ParsedPage with clean text + metadata.

        Args:
            html: Raw HTML content.
            url: Source URL (for reference in the output).

        Returns:
            A ParsedPage with title, cleaned text, and head_metadata dict.
        """
        soup = BeautifulSoup(html, "lxml")

        # ── Step 1: Extract metadata from <head> before stripping ──
        head_metadata = self._extract_head_metadata(soup)

        # ── Step 2: Extract title ──
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        if not title:
            # Try og:title from collected metadata
            title = (
                head_metadata.get("og:title", "")
                or head_metadata.get("twitter:title", "")
                or head_metadata.get("fc:title", "")
            ).strip()

        # ── Step 3: Remove unwanted tags ──
        for tag in self.REMOVE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove hidden elements
        for element in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            element.decompose()

        # ── Step 4: Extract body text ──
        body = soup.body
        if body is None:
            body = soup  # fallback to whole document

        text = body.get_text(separator="\n", strip=True)

        # Clean up: collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Truncate to max length, but keep head portion
        max_len = self.config.max_text_length
        if len(text) > max_len:
            text = text[:max_len]

        return ParsedPage(
            url=url,
            title=title,
            text=text,
            text_length=len(text),
            head_metadata=head_metadata,
        )

    # ── Metadata extraction ────────────────────────────────────────

    def _extract_head_metadata(self, soup: BeautifulSoup) -> dict:
        """Extract all useful metadata from <head> before the DOM is destroyed.

        Collects:
        - <meta name="X" content="Y"> and <meta property="X" content="Y">
        - <script type="application/ld+json"> (first one only)
        - <link rel="canonical" href="...">
        - <time datetime="..."> elements
        """
        meta: dict[str, str] = {}

        # ── Meta tags ──────────────────────────────────────────────
        for tag in soup.find_all("meta"):
            key = tag.get("name") or tag.get("property") or ""
            content = tag.get("content", "")
            if key and content and key.lower() in (
                k.lower() for k in self.META_KEYS
            ):
                meta[key] = content.strip()
            # Also catch any meta with a relevant-looking name
            elif key and content:
                kl = key.lower()
                if any(
                    hint in kl
                    for hint in [
                        "publish", "date", "time", "author", "title", "image",
                        "like", "comment", "count", "baidu", "pw:", "view", "share", "repost",
                    ]
                ):
                    meta[key] = content.strip()

        # ── Canonical link ─────────────────────────────────────────
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            meta["canonical_url"] = canonical["href"].strip()

        # ── JSON-LD ────────────────────────────────────────────────
        ld_json = soup.find("script", type="application/ld+json")
        if ld_json and ld_json.string:
            try:
                ld_data = json.loads(ld_json.string)
                if isinstance(ld_data, list):
                    ld_data = ld_data[0] if ld_data else {}
                if isinstance(ld_data, dict):
                    for field in [
                        "datePublished", "dateModified",
                        "author", "publisher", "headline", "description",
                        "image", "url",
                    ]:
                        val = ld_data.get(field)
                        if val:
                            if isinstance(val, dict):
                                val = val.get("name", str(val))
                            meta[f"jsonld:{field}"] = str(val).strip()
            except (json.JSONDecodeError, TypeError):
                pass

        # ── <time> elements ────────────────────────────────────────
        time_el = soup.find("time")
        if time_el and time_el.get("datetime"):
            meta["time:datetime"] = time_el["datetime"].strip()

        return meta
