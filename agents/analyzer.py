from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from difflib import SequenceMatcher
import json
import os
import re
import sys
import time
from html import escape
from dataclasses import dataclass
from datetime import datetime
from math import ceil, log1p
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# 允许开发者直接运行 python agents/analyzer.py 进行本地调试。
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.state import AgentState, append_log

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - beautifulsoup4 is declared in requirements.
    BeautifulSoup = None  # type: ignore

try:
    from dateutil import parser as date_parser
except ImportError:  # pragma: no cover - fallback for minimal environments.
    date_parser = None


PLACEHOLDER_MARKERS = ("xxxx", "your-", "replace-", "sk-xxxx", "fc-xxxx")
MIN_PUBLISH_YEAR = 1990
MAX_FUTURE_PUBLISH_YEARS = 1


def looks_like_placeholder(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    return not normalized or any(marker in normalized for marker in PLACEHOLDER_MARKERS)


def load_example_env_when_needed() -> None:
    """Keep .env.example as documentation only; runtime config is loaded from .env."""
    return


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=False)
    load_example_env_when_needed()
except ImportError:  # pragma: no cover - python-dotenv is optional for deployed envs.
    load_example_env_when_needed()


@dataclass
class AnalyzerConfig:
    tikomni_api_key_env: str = "TIKOMNI_API_KEY"
    tikomni_base_url_env: str = "TIKOMNI_BASE_URL"
    tikomni_auth_header_env: str = "TIKOMNI_AUTH_HEADER"
    tikomni_auth_scheme_env: str = "TIKOMNI_AUTH_SCHEME"
    firecrawl_api_key_env: str = "FIRECRAWL_API_KEY"
    llm_api_key_env: str = "LLM_API_KEY"
    llm_base_url_env: str = "LLM_BASE_URL"
    llm_model_env: str = "LLM_MODEL"
    default_llm_model: str = "deepseek-v4-pro"
    unknown_time_sort_value: str = "9999-12-31 23:59:59"
    key_node_weight_threshold: float = 0.70
    max_key_nodes_per_platform: int = 3
    key_node_platform_ratio: float = 0.15
    topology_visibility_threshold: float = 0.35
    node_weight_time_ratio: float = 0.10
    node_weight_engagement_ratio: float = 0.45
    node_weight_publisher_ratio: float = 0.20
    node_weight_relation_ratio: float = 0.15
    node_weight_confidence_ratio: float = 0.10
    llm_temperature: float = 0.1
    llm_timeout_seconds: int = 30
    llm_relation_timeout_seconds: int = 75
    firecrawl_timeout_ms: int = 15000
    page_fetch_timeout_seconds: int = 8
    page_fetch_max_chars: int = 200000
    enable_tikomni: bool = True
    tikomni_timeout_seconds: int = 20
    enable_firecrawl: bool = True
    enable_llm: bool = True
    enable_llmscrapy: bool = True
    llmscrapy_fetcher: str = "firecrawl"
    llmscrapy_enrich_mode: str = "baidu"
    llmscrapy_max_workers: int = 5
    max_firecrawl_nodes: int = 200
    max_llm_nodes: int = 30
    max_page_fetch_nodes: int = 200
    max_llmscrapy_nodes: int = 200
    max_workers: int = 5
    max_llm_enrichment_nodes: int = 20
    max_llm_relation_nodes: int = 12
    enable_llm_relation_analysis: bool = True
    enable_mock_fallback: bool = False

    def __post_init__(self) -> None:
        self.llm_timeout_seconds = env_int("ANALYZER_LLM_TIMEOUT_SECONDS", self.llm_timeout_seconds, 1)
        self.llm_relation_timeout_seconds = env_int(
            "ANALYZER_LLM_RELATION_TIMEOUT_SECONDS",
            self.llm_relation_timeout_seconds,
            1,
        )
        self.firecrawl_timeout_ms = env_int("ANALYZER_FIRECRAWL_TIMEOUT_MS", self.firecrawl_timeout_ms, 1000)
        self.page_fetch_timeout_seconds = env_int(
            "ANALYZER_PAGE_FETCH_TIMEOUT_SECONDS",
            self.page_fetch_timeout_seconds,
            1,
        )
        self.enable_tikomni = env_flag("ANALYZER_ENABLE_TIKOMNI", self.enable_tikomni)
        self.tikomni_timeout_seconds = env_int(
            "ANALYZER_TIKOMNI_TIMEOUT_SECONDS",
            self.tikomni_timeout_seconds,
            1,
        )
        self.enable_firecrawl = env_flag("ANALYZER_ENABLE_FIRECRAWL", self.enable_firecrawl)
        self.enable_llm = env_flag("ANALYZER_ENABLE_LLM", self.enable_llm)
        self.enable_llmscrapy = env_flag("ANALYZER_ENABLE_LLMSCRAPY", self.enable_llmscrapy)
        self.llmscrapy_fetcher = (
            os.getenv("ANALYZER_LLMSCRAPY_FETCHER", self.llmscrapy_fetcher).strip()
            or self.llmscrapy_fetcher
        )
        self.llmscrapy_enrich_mode = (
            os.getenv("ANALYZER_LLMSCRAPY_ENRICH_MODE", self.llmscrapy_enrich_mode).strip() or "baidu"
        )
        self.llmscrapy_max_workers = env_int(
            "ANALYZER_LLMSCRAPY_MAX_WORKERS",
            self.llmscrapy_max_workers,
            1,
        )
        self.max_firecrawl_nodes = env_int("ANALYZER_MAX_FIRECRAWL_NODES", self.max_firecrawl_nodes, 0)
        self.max_llm_nodes = env_int("ANALYZER_MAX_LLM_NODES", self.max_llm_nodes, 0)
        self.max_page_fetch_nodes = env_int("ANALYZER_MAX_PAGE_FETCH_NODES", self.max_page_fetch_nodes, 0)
        self.max_llmscrapy_nodes = env_int("ANALYZER_MAX_LLMSCRAPY_NODES", self.max_llmscrapy_nodes, 0)
        self.max_workers = env_int("ANALYZER_MAX_WORKERS", self.max_workers, 1)
        self.max_llm_enrichment_nodes = env_int(
            "ANALYZER_MAX_LLM_ENRICHMENT_NODES",
            self.max_llm_enrichment_nodes,
            0,
        )
        self.max_llm_relation_nodes = env_int("ANALYZER_MAX_LLM_RELATION_NODES", self.max_llm_relation_nodes, 0)
        self.enable_llm_relation_analysis = env_flag(
            "ANALYZER_ENABLE_LLM_RELATION_ANALYSIS",
            self.enable_llm_relation_analysis,
        )
        self.enable_mock_fallback = env_flag("ANALYZER_ENABLE_MOCK_FALLBACK", self.enable_mock_fallback)


MOCK_PARSE_RESULT: Dict[str, Dict[str, Any]] = {
    "n1": {
        "published_at": "2024-03-14 09:12:00",
        "location_hint": "南京",
        "propagation_role": "疑似源头",
        "publisher": "摄影师账号",
        "view_count": 12500,
        "repost_count": 860,
        "comment_count": 124,
        "like_count": 2300,
        "follower_count": 18000,
        "following_count": 320,
        "public_relation_hint": "原始发布页包含作者署名",
        "llm_confidence": 0.92,
        "analyzer_reason": "Mock 数据显示该节点发布时间最早且为原始发布。",
    },
    "n2": {
        "published_at": "2024-03-15 18:30:00",
        "location_hint": "南京",
        "propagation_role": "媒体转载",
        "publisher": "地方新闻站",
        "view_count": 90000,
        "repost_count": 1400,
        "comment_count": 368,
        "like_count": 8200,
        "follower_count": 240000,
        "following_count": 75,
        "public_relation_hint": "正文提到来自摄影师原始发布页",
        "llm_confidence": 0.86,
        "analyzer_reason": "Mock 数据显示该节点晚于源头，互动量较高。",
    },
    "n3": {
        "published_at": "2024-03-16 11:05:00",
        "location_hint": "未知",
        "propagation_role": "社交扩散",
        "publisher": "社交平台用户",
        "view_count": 210000,
        "repost_count": 7600,
        "comment_count": 920,
        "like_count": 18400,
        "follower_count": 56000,
        "following_count": 610,
        "public_relation_hint": "公开页面显示转发链路",
        "llm_confidence": 0.81,
        "analyzer_reason": "Mock 数据显示该节点为高互动社交扩散节点。",
    },
}


class TikomniProvider:
    """Structured Tikomni API adapter for platform post metadata."""

    WEIBO_STATUS_DETAIL_PATH = "/api/u1/v1/weibo/app/fetch_status_detail"
    XHS_NOTE_INFO_V7_PATH = "/api/u1/v1/xiaohongshu/web/get_note_info_v7"
    XHS_USER_INFO_PATH = "/api/u1/v1/xiaohongshu/web_v3/fetch_user_info"

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.api_key = os.getenv(config.tikomni_api_key_env, "") or os.getenv("TIKHUB_API_KEY", "")
        self.base_url = os.getenv(config.tikomni_base_url_env, "https://api.tikomni.com").rstrip("/")
        self.auth_header = os.getenv(config.tikomni_auth_header_env, "Authorization")
        self.auth_scheme = os.getenv(config.tikomni_auth_scheme_env, "Bearer")
        self.enabled = bool(config.enable_tikomni) and not looks_like_placeholder(self.api_key)
        self.reason = ""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = Lock()
        self.session = requests.Session()
        _proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or ""
        if _proxy:
            self.session.proxies.update({"http": _proxy, "https": _proxy})
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Image-Source-Tracing-Analyzer/0.1",
            }
        )
        if not config.enable_tikomni:
            self.reason = "Tikomni disabled by ANALYZER_ENABLE_TIKOMNI"
        elif looks_like_placeholder(self.api_key):
            self.reason = "missing or placeholder TIKOMNI_API_KEY"

    def fetch_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        url = str(node.get("url") or node.get("page_url") or node.get("source_url") or "")
        platform = self._infer_platform(node, url)
        if platform == "xiaohongshu":
            note_id = self.extract_xhs_note_id(url) or str(
                node.get("note_id") or node.get("post_id") or node.get("idstr") or ""
            )
            xsec_token = self.extract_xhs_xsec_token(url) or str(node.get("xsec_token") or "")
            if not note_id:
                return {
                    "ok": False,
                    "provider": "tikomni",
                    "platform": "xiaohongshu",
                    "status": "skipped",
                    "error": "missing xiaohongshu note_id",
                    "raw_endpoint": "",
                    "latency_ms": 0,
                    "raw": {},
                }

            cache_key = f"xiaohongshu:{note_id}:{xsec_token}"
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    return {**cached, "cache_hit": True}
                result = self._fetch_xhs_note_detail(note_id, xsec_token)
                self._cache[cache_key] = result
                return {**result, "cache_hit": False}

        if platform != "weibo":
            return {
                "ok": False,
                "provider": "tikomni",
                "platform": platform,
                "status": "skipped",
                "error": f"unsupported platform for current Tikomni analyzer adapter: {platform}",
                "raw_endpoint": "",
                "latency_ms": 0,
                "raw": {},
            }

        status_id = self.extract_weibo_status_id(url) or str(
            node.get("status_id") or node.get("post_id") or node.get("note_id") or ""
        )
        if not status_id:
            return {
                "ok": False,
                "provider": "tikomni",
                "platform": "weibo",
                "status": "skipped",
                "error": "missing weibo status_id",
                "raw_endpoint": "",
                "latency_ms": 0,
                "raw": {},
            }

        cache_key = f"weibo:{status_id}"
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return {**cached, "cache_hit": True}
            result = self._fetch_weibo_status_detail(status_id)
            self._cache[cache_key] = result
            return {**result, "cache_hit": False}

    def _fetch_xhs_note_detail(self, note_id: str, xsec_token: str = "") -> Dict[str, Any]:
        endpoint = self._resolve_endpoint("TIKOMNI_XHS_NOTE_INFO_V7_ENDPOINT", self.XHS_NOTE_INFO_V7_PATH)
        result: Dict[str, Any] = {
            "ok": False,
            "provider": "tikomni",
            "platform": "xiaohongshu",
            "note_id": note_id,
            "status": "failed",
            "error": "",
            "raw_endpoint": endpoint,
            "latency_ms": 0,
            "raw": {},
        }
        if not self.enabled:
            result["status"] = "skipped"
            result["error"] = self.reason
            return result

        headers = self._auth_headers()
        started = time.perf_counter()
        try:
            response = self.session.get(
                endpoint,
                params={"note_id": note_id, "xsec_token": xsec_token or ""},
                headers=headers,
                timeout=self.config.tikomni_timeout_seconds,
            )
            result["latency_ms"] = int((time.perf_counter() - started) * 1000)
            result["http_status"] = response.status_code
            try:
                payload = response.json()
            except ValueError:
                payload = {"text": response.text[:1000]}
            result["raw"] = payload if isinstance(payload, dict) else {"data": payload}
            if response.status_code >= 400:
                result["status"] = "failed"
                result["error"] = f"HTTP {response.status_code}: {response.text[:300]}"
                return result

            code = result["raw"].get("code")
            nested_code = self._nested_get(result["raw"], "data.code")
            if code not in (None, 0, 200, "0", "200") or nested_code not in (None, 0, 200, "0", "200"):
                result["status"] = "failed"
                result["error"] = str(
                    result["raw"].get("message")
                    or result["raw"].get("message_zh")
                    or self._nested_get(result["raw"], "data.msg")
                    or f"Tikomni code={code or nested_code}"
                )
                return result

            note = self._nested_get(result["raw"], "data.data.0.note_list.0")
            if isinstance(note, dict):
                result["raw"]["xhs_note"] = note
                user_id = str(
                    self._nested_get(note, "user.id")
                    or self._nested_get(note, "user.userid")
                    or ""
                )
                if user_id:
                    user_result = self._fetch_xhs_user_info(user_id)
                    if user_result:
                        result["raw"]["xhs_user_profile"] = user_result.get("profile") or {}
                        result["raw"]["xhs_user_raw"] = user_result.get("raw") or {}
                        result["raw"]["xhs_user_endpoint"] = user_result.get("raw_endpoint") or ""
                        result["latency_ms"] = self._to_int(result["latency_ms"]) + self._to_int(
                            user_result.get("latency_ms")
                        )

            result["ok"] = True
            result["status"] = "success"
            return result
        except requests.RequestException as exc:
            result["latency_ms"] = int((time.perf_counter() - started) * 1000)
            result["status"] = "failed"
            result["error"] = str(exc)
            return result

    def _fetch_xhs_user_info(self, user_id: str) -> Dict[str, Any]:
        endpoint = os.getenv("TIKOMNI_XHS_USER_INFO_ENDPOINT") or (
            f"{self.base_url}{self.XHS_USER_INFO_PATH}"
        )
        started = time.perf_counter()
        result: Dict[str, Any] = {
            "status": "failed",
            "error": "",
            "raw_endpoint": endpoint,
            "latency_ms": 0,
            "raw": {},
            "profile": {},
        }
        try:
            response = self.session.get(
                endpoint,
                params={"user_id": user_id},
                headers=self._auth_headers(),
                timeout=self.config.tikomni_timeout_seconds,
            )
            result["latency_ms"] = int((time.perf_counter() - started) * 1000)
            result["http_status"] = response.status_code
            try:
                payload = response.json()
            except ValueError:
                payload = {"text": response.text[:1000]}
            result["raw"] = payload if isinstance(payload, dict) else {"data": payload}
            if response.status_code >= 400:
                result["error"] = f"HTTP {response.status_code}: {response.text[:300]}"
                return result
            profile = self._nested_get(result["raw"], "data.data")
            if isinstance(profile, dict):
                result["profile"] = profile
                result["status"] = "success"
            return result
        except requests.RequestException as exc:
            result["latency_ms"] = int((time.perf_counter() - started) * 1000)
            result["error"] = str(exc)
            return result

    def _resolve_endpoint(self, env_key: str, default_path: str) -> str:
        ep = os.getenv(env_key, "").strip()
        if ep and ep.startswith("http"):
            return ep
        if ep:
            return f"{self.base_url}{ep}"
        return f"{self.base_url}{default_path}"

    def _fetch_weibo_status_detail(self, status_id: str) -> Dict[str, Any]:
        endpoint = self._resolve_endpoint("TIKOMNI_WEIBO_STATUS_DETAIL_ENDPOINT", self.WEIBO_STATUS_DETAIL_PATH)
        result: Dict[str, Any] = {
            "ok": False,
            "provider": "tikomni",
            "platform": "weibo",
            "status_id": status_id,
            "status": "failed",
            "error": "",
            "raw_endpoint": endpoint,
            "latency_ms": 0,
            "raw": {},
        }
        if not self.enabled:
            result["status"] = "skipped"
            result["error"] = self.reason
            return result

        headers = self._auth_headers()
        started = time.perf_counter()
        try:
            response = self.session.get(
                endpoint,
                params={"status_id": status_id},
                headers=headers,
                timeout=self.config.tikomni_timeout_seconds,
            )
            result["latency_ms"] = int((time.perf_counter() - started) * 1000)
            result["http_status"] = response.status_code
            try:
                payload = response.json()
            except ValueError:
                payload = {"text": response.text[:1000]}
            result["raw"] = payload if isinstance(payload, dict) else {"data": payload}
            if response.status_code >= 400:
                result["status"] = "failed"
                result["error"] = f"HTTP {response.status_code}: {response.text[:300]}"
                return result
            code = result["raw"].get("code")
            if code not in (None, 0, 200, "0", "200"):
                result["status"] = "failed"
                result["error"] = str(
                    result["raw"].get("message")
                    or result["raw"].get("message_zh")
                    or f"Tikomni code={code}"
                )
                return result
            result["ok"] = True
            result["status"] = "success"
            return result
        except requests.RequestException as exc:
            result["latency_ms"] = int((time.perf_counter() - started) * 1000)
            result["status"] = "failed"
            result["error"] = str(exc)
            return result

    def _auth_headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        if self.auth_scheme:
            value = f"{self.auth_scheme} {self.api_key}"
        else:
            value = self.api_key
        return {self.auth_header: value}

    @staticmethod
    def _nested_get(data: Any, path: str) -> Any:
        current = data
        for part in path.split("."):
            if isinstance(current, list) and part.isdigit():
                index = int(part)
                current = current[index] if index < len(current) else None
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    @staticmethod
    def _to_int(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        try:
            return int(float(str(value or 0).replace(",", "")))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def extract_weibo_status_id(url: str) -> Optional[str]:
        parsed = urlparse(url or "")
        query = parse_qs(parsed.query)
        for key in ("status_id", "id", "mid"):
            values = query.get(key)
            if values and values[0]:
                return values[0]
        patterns = [
            r"/status/([0-9A-Za-z]+)",
            r"/detail/([0-9A-Za-z]+)",
            r"/(\d{10,})/?$",
        ]
        for pattern in patterns:
            match = re.search(pattern, parsed.path)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def extract_xhs_note_id(url: str) -> Optional[str]:
        parsed = urlparse(url or "")
        query = parse_qs(parsed.query)
        for key in ("note_id", "noteId", "id"):
            values = query.get(key)
            if values and values[0]:
                return values[0]
        patterns = [
            r"/(?:explore|discovery/item|item)/([0-9A-Za-z]+)",
            r"/([0-9A-Fa-f]{24})(?:/)?$",
        ]
        for pattern in patterns:
            match = re.search(pattern, parsed.path)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def extract_xhs_xsec_token(url: str) -> Optional[str]:
        parsed = urlparse(url or "")
        values = parse_qs(parsed.query).get("xsec_token")
        return values[0] if values and values[0] else None

    @staticmethod
    def _infer_platform(node: Dict[str, Any], url: str) -> str:
        raw = str(node.get("platform") or "").strip().lower()
        if raw in {"weibo", "wb", "微博"}:
            return "weibo"
        if raw in {"xiaohongshu", "xhs", "redbook", "小红书"}:
            return "xiaohongshu"
        host = urlparse(url or "").netloc.lower()
        if "weibo." in host or "weibo.cn" in host:
            return "weibo"
        if "xiaohongshu.com" in host or "xhslink.com" in host:
            return "xiaohongshu"
        return "other"


class FirecrawlCrawler:
    """Thin optional wrapper around firecrawl-py.

    The analyzer only scrapes URLs already present in nodes_data. It does not
    search for new URLs or perform reverse image retrieval.
    """

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.api_key = os.getenv(config.firecrawl_api_key_env, "")
        self.client: Any = None
        self.enabled = False
        self.reason = ""

        if not config.enable_firecrawl:
            self.reason = "Firecrawl disabled by ANALYZER_ENABLE_FIRECRAWL"
            return

        if looks_like_placeholder(self.api_key):
            self.reason = "missing or placeholder FIRECRAWL_API_KEY"
            return

        Firecrawl = None
        FirecrawlApp = None
        try:
            from firecrawl import Firecrawl as ImportedFirecrawl  # type: ignore

            Firecrawl = ImportedFirecrawl
        except ImportError:
            pass
        try:
            from firecrawl import FirecrawlApp as ImportedFirecrawlApp  # type: ignore

            FirecrawlApp = ImportedFirecrawlApp
        except ImportError:
            pass
        if Firecrawl is None and FirecrawlApp is None:
            self.reason = "firecrawl-py is not installed"
            return

        try:
            if Firecrawl is not None:
                self.client = Firecrawl(api_key=self.api_key)
            else:
                self.client = FirecrawlApp(api_key=self.api_key)
            self.enabled = True
        except Exception as exc:  # pragma: no cover - depends on SDK version.
            self.reason = f"failed to initialize Firecrawl client: {exc}"

    def scrape(self, url: str) -> Dict[str, Any]:
        if not url:
            return {"crawl_status": "failed", "error": "missing url"}
        if not self.enabled or self.client is None:
            return {"crawl_status": "skipped", "error": self.reason}

        formats = ["markdown", "html", "rawHtml", "links"]
        try:
            if hasattr(self.client, "scrape"):
                result = self.client.scrape(
                    url,
                    formats=formats,
                    timeout=self.config.firecrawl_timeout_ms,
                    only_main_content=False,
                )
            elif hasattr(self.client, "scrape_url"):
                result = self.client.scrape_url(
                    url,
                    params={
                        "formats": formats,
                        "timeout": self.config.firecrawl_timeout_ms,
                        "onlyMainContent": False,
                    },
                )
            else:
                return {"crawl_status": "failed", "error": "unsupported firecrawl client"}
        except Exception as exc:
            return {"crawl_status": "failed", "error": str(exc)}

        normalized = self._to_dict(result)
        nested_data = normalized.get("data")
        if isinstance(nested_data, dict):
            merged = dict(nested_data)
            merged.update({key: value for key, value in normalized.items() if key != "data"})
            normalized = merged
        if "html" not in normalized:
            normalized["html"] = (
                normalized.get("rawHtml")
                or normalized.get("raw_html")
                or normalized.get("html")
                or ""
            )
        if "markdown" not in normalized:
            normalized["markdown"] = normalized.get("content") or normalized.get("text") or ""
        if not isinstance(normalized.get("metadata"), dict):
            normalized["metadata"] = {}
        if normalized.get("html"):
            html_metadata, visible_text = self._extract_html_metadata(str(normalized.get("html") or ""))
            for key, value in html_metadata.items():
                normalized["metadata"].setdefault(key, value)
            if visible_text and not normalized.get("markdown"):
                normalized["markdown"] = visible_text
        normalized["crawl_status"] = "success"
        normalized["crawl_source"] = "firecrawl"
        return normalized

    @staticmethod
    def _to_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return {"raw": str(value)}

    @staticmethod
    def _extract_html_metadata(html: str) -> tuple[Dict[str, Any], str]:
        metadata: Dict[str, Any] = {}
        if not html:
            return metadata, ""
        if BeautifulSoup is None:
            text = re.sub(r"<[^>]+>", " ", html)
            return metadata, re.sub(r"\s+", " ", text).strip()[:12000]

        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()
        for meta in soup.find_all("meta"):
            key = meta.get("property") or meta.get("name") or meta.get("itemprop")
            value = meta.get("content")
            if key and value:
                metadata.setdefault(str(key).strip(), str(value).strip())
        for time_tag in soup.find_all("time"):
            value = time_tag.get("datetime") or time_tag.get("pubdate") or time_tag.get_text(" ", strip=True)
            if value:
                metadata.setdefault("time_tag", str(value).strip())
                break
        time_attrs = ("datetime", "data-time", "data-publish-time", "data-pubtime", "data-created-at")
        time_selector = re.compile("time|date|publish|pub|created|updated", re.I)
        for tag in soup.find_all(attrs={"class": time_selector}) + soup.find_all(attrs={"id": time_selector}):
            value = ""
            for attr_name in time_attrs:
                if tag.get(attr_name):
                    value = str(tag.get(attr_name)).strip()
                    break
            if not value:
                value = tag.get_text(" ", strip=True)
            if value:
                metadata.setdefault("visible_time_hint", value)
                break
        for script in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
            script_text = script.string or script.get_text(" ", strip=True)
            FirecrawlCrawler._merge_json_ld_metadata(metadata, script_text)
        for script in soup.find_all("script"):
            script_text = script.string or script.get_text(" ", strip=True)
            FirecrawlCrawler._merge_inline_time_metadata(metadata, script_text)
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return metadata, text[:12000]

    @staticmethod
    def _merge_json_ld_metadata(metadata: Dict[str, Any], script_text: str) -> None:
        try:
            payload = json.loads(script_text)
        except (TypeError, json.JSONDecodeError):
            return
        for item in FirecrawlCrawler._iter_json_objects(payload):
            for source_key, target_key in {
                "datePublished": "datePublished",
                "dateCreated": "dateCreated",
                "dateModified": "dateModified",
                "uploadDate": "uploadDate",
                "headline": "headline",
                "name": "jsonld_name",
            }.items():
                value = item.get(source_key)
                if value:
                    metadata.setdefault(target_key, value)
            for source_key in ("author", "creator", "publisher"):
                value = FirecrawlCrawler._json_name(item.get(source_key))
                if value:
                    metadata.setdefault(source_key, value)

    @staticmethod
    def _merge_inline_time_metadata(metadata: Dict[str, Any], text: str) -> None:
        if not text:
            return
        patterns = {
            "publish_time": r'["\']?(?:publish[_-]?time|publishTime|published[_-]?at|publishedAt|public[_-]?time|pubTime|pubtime|pubDate|releaseDate|displayTime|postTime)["\']?\s*[:=]\s*["\']?([^"\'},<]{6,60})',
            "create_time": r'["\']?(?:create[_-]?time|createTime|created[_-]?at|createdAt|ctime|addTime)["\']?\s*[:=]\s*["\']?([^"\'},<]{6,60})',
            "update_time": r'["\']?(?:update[_-]?time|updateTime|updated[_-]?at|updatedAt|modifyTime|modifiedTime)["\']?\s*[:=]\s*["\']?([^"\'},<]{6,60})',
            "datePublished": r'["\']datePublished["\']\s*:\s*["\']([^"\']{6,60})',
            "uploadDate": r'["\']uploadDate["\']\s*:\s*["\']([^"\']{6,60})',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.I)
            if match:
                metadata.setdefault(key, match.group(1).strip().strip('"\' '))

    @staticmethod
    def _iter_json_objects(value: Any) -> List[Dict[str, Any]]:
        if isinstance(value, dict):
            items = [value]
            graph = value.get("@graph")
            if isinstance(graph, list):
                items.extend(item for item in graph if isinstance(item, dict))
            return items
        if isinstance(value, list):
            result: List[Dict[str, Any]] = []
            for item in value:
                result.extend(FirecrawlCrawler._iter_json_objects(item))
            return result
        return []

    @staticmethod
    def _json_name(value: Any) -> Optional[str]:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            name = value.get("name") or value.get("@id")
            return str(name) if name else None
        if isinstance(value, list):
            names = [FirecrawlCrawler._json_name(item) for item in value]
            names = [name for name in names if name]
            return ", ".join(names) if names else None
        return None


class DirectPageCrawler:
    """Fallback page fetcher used when Firecrawl is not configured.

    It only reads candidate URLs that the retriever already found. The goal is
    to extract public page metadata and visible date strings for timeline use.
    """

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.session = requests.Session()
        retry = Retry(
            total=0,
            connect=0,
            read=0,
            status=0,
            backoff_factor=0.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
            }
        )

    def scrape(self, url: str) -> Dict[str, Any]:
        if not url:
            return {"crawl_status": "failed", "error": "missing url", "crawl_source": "direct_http"}

        try:
            response = self.session.get(
                url,
                timeout=self.config.page_fetch_timeout_seconds,
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            return {
                "crawl_status": "failed",
                "error": str(exc),
                "crawl_source": "direct_http",
            }

        limited_reason = self._limited_page_reason(response.url, response.text, response.status_code)
        content_type = response.headers.get("content-type", "")
        text = response.text if "text" in content_type or "html" in content_type else ""
        if not text:
            return {
                "crawl_status": "failed",
                "error": f"unsupported content-type: {content_type or 'unknown'}",
                "crawl_source": "direct_http",
                "status_code": response.status_code,
                "final_url": response.url,
            }

        html = text[: self.config.page_fetch_max_chars]
        metadata, visible_text = self._extract_page_data(html)
        metadata.update(
            {
                "final_url": response.url,
                "status_code": response.status_code,
                "content_type": content_type,
            }
        )
        last_modified = self._normalize_http_date(response.headers.get("last-modified", ""))
        if last_modified and not limited_reason:
            metadata["http_last_modified"] = last_modified

        title = metadata.get("title") or ""
        return {
            "crawl_status": "limited" if limited_reason else "success",
            "crawl_source": "direct_http",
            "error": limited_reason,
            "metadata": metadata,
            "html": html,
            "markdown": "\n".join(part for part in [title, visible_text] if part),
            "links": self._extract_links(html, response.url),
            "status_code": response.status_code,
            "final_url": response.url,
        }

    def _limited_page_reason(self, url: str, text: str, status_code: int) -> Optional[str]:
        lowered_url = url.lower()
        sample = (text or "")[:5000].lower()
        if status_code in {401, 403, 429}:
            return f"页面未提供正文内容，HTTP 状态码 {status_code}"
        if "wappass.baidu.com/static/captcha" in lowered_url or "captcha" in lowered_url:
            return "页面跳转到验证页，未提供原始正文内容"
        if "please wait" in sample and "douyin" in lowered_url:
            return "页面需要动态加载，直接 HTML 未提供正文内容"
        return None

    def _extract_page_data(self, html: str) -> tuple[Dict[str, Any], str]:
        metadata: Dict[str, Any] = {}
        if BeautifulSoup is None:
            return self._extract_page_data_by_regex(html)

        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()

        for meta in soup.find_all("meta"):
            key = meta.get("property") or meta.get("name") or meta.get("itemprop")
            value = meta.get("content")
            if key and value and key not in metadata:
                metadata[str(key).strip()] = str(value).strip()

        for time_tag in soup.find_all("time"):
            value = time_tag.get("datetime") or time_tag.get("pubdate") or time_tag.get_text(" ", strip=True)
            if value:
                metadata.setdefault("time_tag", str(value).strip())
                break

        time_attr_names = ("datetime", "data-time", "data-publish-time", "data-pubtime", "data-created-at")
        for tag in soup.find_all(attrs={"class": re.compile("time|date|publish|created|updated", re.I)}):
            value = ""
            for attr_name in time_attr_names:
                if tag.get(attr_name):
                    value = str(tag.get(attr_name)).strip()
                    break
            if not value:
                value = tag.get_text(" ", strip=True)
            if value:
                metadata.setdefault("visible_time_hint", value)
                break

        for script in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
            script_text = script.string or script.get_text(" ", strip=True)
            self._merge_json_ld(metadata, script_text)

        for script in soup.find_all("script"):
            script_text = script.string or script.get_text(" ", strip=True)
            self._merge_inline_time_metadata(metadata, script_text)

        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        visible_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return metadata, visible_text[:50000]

    def _extract_page_data_by_regex(self, html: str) -> tuple[Dict[str, Any], str]:
        metadata: Dict[str, Any] = {}
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        if title_match:
            metadata["title"] = re.sub(r"\s+", " ", title_match.group(1)).strip()
        for match in re.finditer(r"<meta\s+([^>]+)>", html, flags=re.I):
            attrs = match.group(1)
            key_match = re.search(r'(?:property|name|itemprop)=["\']([^"\']+)["\']', attrs, flags=re.I)
            value_match = re.search(r'content=["\']([^"\']+)["\']', attrs, flags=re.I)
            if key_match and value_match:
                metadata.setdefault(key_match.group(1), value_match.group(1))
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        self._merge_inline_time_metadata(metadata, html)
        return metadata, text[:50000]

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        if BeautifulSoup is None:
            return []
        soup = BeautifulSoup(html, "lxml")
        links: List[str] = []
        for tag in soup.find_all("a", href=True):
            href = str(tag.get("href", "")).strip()
            if href and href.startswith(("http://", "https://")):
                links.append(href)
            if len(links) >= 50:
                break
        return links

    def _merge_json_ld(self, metadata: Dict[str, Any], script_text: str) -> None:
        try:
            payload = json.loads(script_text)
        except (TypeError, json.JSONDecodeError):
            return
        for item in self._iter_json_objects(payload):
            for source_key, target_key in {
                "datePublished": "datePublished",
                "dateCreated": "dateCreated",
                "dateModified": "dateModified",
                "uploadDate": "uploadDate",
                "headline": "headline",
                "name": "jsonld_name",
            }.items():
                value = item.get(source_key)
                if value:
                    metadata.setdefault(target_key, value)
            for source_key in ("author", "creator", "publisher"):
                value = self._json_name(item.get(source_key))
                if value:
                    metadata.setdefault(source_key, value)

    def _merge_inline_time_metadata(self, metadata: Dict[str, Any], text: str) -> None:
        if not text:
            return
        patterns = {
            "publish_time": r'["\']?(?:publish[_-]?time|publishTime|published[_-]?at|publishedAt|public[_-]?time|pubTime|pubtime|pubDate|releaseDate|displayTime|postTime)["\']?\s*[:=]\s*["\']?([^"\'},<]{6,40})',
            "create_time": r'["\']?(?:create[_-]?time|createTime|created[_-]?at|createdAt|ctime|addTime)["\']?\s*[:=]\s*["\']?([^"\'},<]{6,40})',
            "update_time": r'["\']?(?:update[_-]?time|updateTime|updated[_-]?at|updatedAt|modifyTime|modifiedTime)["\']?\s*[:=]\s*["\']?([^"\'},<]{6,40})',
            "datePublished": r'["\']datePublished["\']\s*:\s*["\']([^"\']{6,40})',
            "uploadDate": r'["\']uploadDate["\']\s*:\s*["\']([^"\']{6,40})',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.I)
            if match:
                metadata.setdefault(key, match.group(1).strip().strip('"\' '))

    def _iter_json_objects(self, value: Any) -> List[Dict[str, Any]]:
        if isinstance(value, dict):
            items = [value]
            graph = value.get("@graph")
            if isinstance(graph, list):
                items.extend(item for item in graph if isinstance(item, dict))
            return items
        if isinstance(value, list):
            result: List[Dict[str, Any]] = []
            for item in value:
                result.extend(self._iter_json_objects(item))
            return result
        return []

    @staticmethod
    def _json_name(value: Any) -> Optional[str]:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            name = value.get("name") or value.get("@id")
            return str(name) if name else None
        if isinstance(value, list):
            names = [DirectPageCrawler._json_name(item) for item in value]
            names = [name for name in names if name]
            return ", ".join(names) if names else None
        return None

    @staticmethod
    def _normalize_http_date(value: str) -> Optional[str]:
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        return parsed.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


class LlmScrapyCrawler:
    """Adapter for the local llmscrapy pipeline used by external platforms."""

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.enabled = False
        self.reason = ""
        self.pipeline = None
        self.URLSource = None
        if not config.enable_llmscrapy:
            self.reason = "llmscrapy disabled by ANALYZER_ENABLE_LLMSCRAPY"
            return

        tool_root = Path(__file__).resolve().parents[1] / "tools"
        tool_root_str = str(tool_root)
        if tool_root_str not in sys.path:
            sys.path.insert(0, tool_root_str)

        try:
            from llmscrapy.config import CrawlerConfig, LLMConfig  # type: ignore
            from llmscrapy.models import URLSource  # type: ignore
            from llmscrapy.pipeline import Pipeline  # type: ignore
        except Exception as exc:
            self.reason = f"failed to import llmscrapy: {exc}"
            return

        llm_api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv(config.llm_api_key_env, "")
        llm_base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv(config.llm_base_url_env, "")
        llm_model = os.getenv("DEEPSEEK_MODEL") or os.getenv(config.llm_model_env, config.default_llm_model)
        if not llm_api_key:
            self.reason = "missing DEEPSEEK_API_KEY/LLM_API_KEY for llmscrapy"
            return

        try:
            crawler_config = CrawlerConfig()
            crawler_config.fetcher_type = config.llmscrapy_fetcher
            crawler_config.enrich_mode = config.llmscrapy_enrich_mode
            crawler_config.max_workers = max(config.llmscrapy_max_workers, 1)
            crawler_config.request_timeout = max(config.page_fetch_timeout_seconds, 1)

            llm_config = LLMConfig()
            llm_config.api_key = llm_api_key
            if llm_base_url:
                llm_config.base_url = llm_base_url
            if llm_model:
                llm_config.model = llm_model
            llm_config.temperature = 0.0

            self.pipeline = Pipeline(crawler_config=crawler_config, llm_config=llm_config)
            self.URLSource = URLSource
            self.enabled = True
        except Exception as exc:
            self.reason = f"failed to initialize llmscrapy: {exc}"

    def scrape_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self.pipeline is None or self.URLSource is None:
            return {
                "llmscrapy_status": "skipped",
                "llmscrapy_error": self.reason,
                "crawl_data": {},
                "llm_data": {},
            }

        source = self._source_from_node(node)
        if source is None:
            return {
                "llmscrapy_status": "failed",
                "llmscrapy_error": "missing url",
                "crawl_data": {},
                "llm_data": {},
            }

        try:
            result = self.pipeline.run_url(source)
        except Exception as exc:
            return {
                "llmscrapy_status": "failed",
                "llmscrapy_error": str(exc),
                "crawl_data": {},
                "llm_data": {},
            }

        return self._convert_result(result, str(source.url))

    def scrape_nodes(self, indexed_nodes: List[Tuple[int, Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
        results: Dict[int, Dict[str, Any]] = {}
        if not indexed_nodes:
            return results
        if not self.enabled or self.pipeline is None or self.URLSource is None:
            for index, _node in indexed_nodes:
                results[index] = {
                    "llmscrapy_status": "skipped",
                    "llmscrapy_error": self.reason,
                    "crawl_data": {},
                    "llm_data": {},
                }
            return results

        sources = []
        source_indexes: List[int] = []
        for index, node in indexed_nodes:
            source = self._source_from_node(node)
            if source is None:
                results[index] = {
                    "llmscrapy_status": "failed",
                    "llmscrapy_error": "missing url",
                    "crawl_data": {},
                    "llm_data": {},
                }
                continue
            sources.append(source)
            source_indexes.append(index)

        if not sources:
            return results

        try:
            batch_results = self.pipeline.run_batch(
                sources,
                delay=0.0,
                workers=max(self.config.llmscrapy_max_workers, 1),
            )
        except Exception as exc:
            for index in source_indexes:
                results[index] = {
                    "llmscrapy_status": "failed",
                    "llmscrapy_error": str(exc),
                    "crawl_data": {},
                    "llm_data": {},
                }
            return results

        for index, source, result in zip(source_indexes, sources, batch_results):
            results[index] = self._convert_result(result, str(source.url))
        return results

    def _source_from_node(self, node: Dict[str, Any]) -> Optional[Any]:
        if self.URLSource is None:
            return None
        url = str(node.get("url") or "")
        if not url:
            return None
        return self.URLSource(
            id=str(node.get("id") or node.get("upstream_id") or url),
            url=url,
            title=str(node.get("title") or ""),
            source=str(node.get("source") or urlparse(url).netloc),
            engine=str(node.get("engine") or ""),
            image_url=str(node.get("image_url") or node.get("thumbnail_url") or ""),
            possible_duplicate=bool(node.get("possible_duplicate")),
            reason=str(node.get("reason") or node.get("validation_reason") or ""),
            similarity=self._safe_float(node.get("similarity")),
        )

    def _convert_result(self, result: Any, fallback_url: str) -> Dict[str, Any]:
        fetched = getattr(result, "fetched", None)
        parsed = getattr(result, "parsed", None)
        metadata = getattr(result, "metadata", None)
        error = str(getattr(result, "error", "") or "")
        succeeded = bool(getattr(result, "succeeded", False))
        status = "success" if succeeded else "failed"

        fetched_url = str(getattr(fetched, "url", "") or fallback_url)
        html = str(getattr(fetched, "html", "") or "")
        parsed_text = str(getattr(parsed, "text", "") or "")
        head_metadata = self._model_to_dict(getattr(parsed, "head_metadata", {}) or {})
        parsed_title = str(getattr(parsed, "title", "") or "")
        status_code = int(getattr(fetched, "status_code", 0) or 0) if fetched is not None else 0
        fetched_error = str(getattr(fetched, "error", "") or "")
        llm_data = self._model_to_dict(metadata)

        crawl_data = {
            "crawl_status": status,
            "crawl_source": "llmscrapy",
            "metadata": head_metadata,
            "markdown": parsed_text,
            "content": parsed_text,
            "html": html,
            "raw_html": html,
            "final_url": fetched_url,
            "title": parsed_title,
            "status_code": status_code,
            "error": error or fetched_error,
            "llmscrapy_result": self._model_to_dict(result),
        }
        return {
            "llmscrapy_status": status,
            "llmscrapy_error": error or fetched_error,
            "crawl_data": crawl_data,
            "llm_data": llm_data,
        }

    @staticmethod
    def _model_to_dict(value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [LlmScrapyCrawler._model_to_dict(item) for item in value]
        if isinstance(value, tuple):
            return [LlmScrapyCrawler._model_to_dict(item) for item in value]
        if isinstance(value, dict):
            return {str(key): LlmScrapyCrawler._model_to_dict(item) for key, item in value.items()}
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        return value

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0


class AnalyzerLLMClient:
    """OpenAI-compatible LLM client used only for structured page extraction."""

    SYSTEM_PROMPT = """You are the external web evidence analyzer for an image provenance system.

The input node is not a trusted platform API record. It may be a news article, aggregator page,
Telegram channel post, forum post, blog post, image-host page, or dynamic-platform landing page.

Tasks:
1. Classify the page as platform_family and page_type.
2. Extract metadata for the page itself: title, published_at, modified_at, publisher, author,
   canonical_url, and image_urls.
3. Decide whether the target image or a plausible edited variant is actually present on the page.
4. Extract provenance evidence: image credit, source text, source URL, via/original/citation hints,
   source platform hint, and source account hint.
5. Decide whether this node may enter the external evidence timeline and whether it may be used as
   a cross-platform relation candidate.
6. Extract optional metrics only when the page clearly shows them for the current page content.

Rules:
- Do not use comment times, related-article times, crawl time, current time, or footer copyright years
  as published_at.
- Do not treat a search-result title as page author or publication-time evidence.
- Do not treat topical relevance as proof that the target image appears.
- Do not invent views, reposts, comments, likes, followers, or share counts.
- For every important non-null field, include a short raw evidence snippet and confidence when possible.
- Return strict JSON only. No markdown."""

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.api_key = os.getenv(config.llm_api_key_env, "")
        self.base_url = os.getenv(config.llm_base_url_env, "")
        self.model = os.getenv(config.llm_model_env, config.default_llm_model)
        self.client: Any = None
        self.enabled = False
        self.reason = ""

        if not config.enable_llm:
            self.reason = "LLM disabled by ANALYZER_ENABLE_LLM"
            return

        if looks_like_placeholder(self.api_key):
            self.reason = "missing or placeholder LLM_API_KEY"
            return

        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            self.reason = "openai package is not installed"
            return

        try:
            client_kwargs: Dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": config.llm_timeout_seconds,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**client_kwargs)
            self.enabled = True
        except Exception as exc:  # pragma: no cover - depends on SDK config.
            self.reason = f"failed to initialize LLM client: {exc}"

    def extract(self, node: Dict[str, Any], crawl_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.client is None:
            return None

        user_prompt = self.build_user_prompt(node, crawl_data)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.config.llm_temperature,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            parsed = self._parse_json(content)
            if parsed is None:
                return {
                    "_llm_error": "failed to parse JSON response",
                    "_llm_raw_preview": content[:500],
                }
            return parsed
        except Exception as exc:
            return {"_llm_error": str(exc)}

    def synthesize_topology(self, evidence_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Ask the LLM to decide source strategy and evidence-backed topology edges."""
        if not self.enabled or self.client is None:
            return None

        system_prompt = """你是图片传播溯源 Analyzer 的关系研判 Agent。
你会拿到 API、爬虫、OCR、validator 和规则初判形成的证据池。你的任务不是重新做图片检索，而是基于证据灵活判断源头策略、跨平台关系、边关系和风险。

硬规则：
1. 只有 weibo 和 xiaohongshu 可参与源头竞争；platform=other 只能作为外围证据，不能成为源头。
2. other 节点默认不建立主传播边；除非它提供明确来源证据，也只写入 external_evidence_nodes 或 risk_summary。
3. 如果微博与小红书之间存在明确转载/来源关系，例如正文写转载自、source_url、对方平台账号、水印账号匹配，则输出 single_source。
4. 如果微博与小红书之间没有明确转载/来源关系，则分别给出 platform_sources，不要强行找全局单源头。
5. 边关系必须有 evidence，不能只凭发布时间强连；same_platform_temporal 只能在“同事件/同主体/OCR相似/账号线索”同时存在时作为弱边。
6. OCR 文本、OCR 中的 @账号/平台号/水印文字/海报主体是重要关系证据；如果 OCR 中 @ 账号匹配发布者，应优先作为水印/署名线索。
7. 对非主流平台，如果 API/爬虫效果差，不要硬猜边；请输出 agent_actions，说明下一步应调用 llmscrapy、截图 OCR、站内搜索或人工复核。
8. 输出严格 JSON，不要 markdown。"""
        schema = {
            "source_decision": {
                "global_source_mode": "single_source | per_platform_sources | unknown",
                "global_source_node_id": "",
                "platform_sources": {"weibo": "", "xiaohongshu": ""},
                "confidence": 0.0,
                "reasoning": [],
            },
            "edges": [
                {
                    "source": "",
                    "target": "",
                    "edge_type": "REPOST | CROSS_PLATFORM | watermark_account_match | ocr_account_match | duplicate_cluster",
                    "edge_weight": 0.0,
                    "evidence": [],
                }
            ],
            "cross_platform_relations": [
                {
                    "source": "",
                    "target": "",
                    "relation_type": "",
                    "confidence": 0.0,
                    "evidence": [],
                }
            ],
            "external_evidence_nodes": [],
            "agent_actions": [
                {
                    "node_id": "",
                    "action": "llmscrapy_pipeline | screenshot_ocr | platform_api | manual_review",
                    "reason": "",
                    "priority": "high | medium | low",
                }
            ],
            "risk_summary": [],
        }
        user_prompt = (
            "请根据证据池输出传播拓扑研判 JSON。必须遵守硬规则，尤其是 other 不参与源头竞争。\n"
            f"输出 schema 示例：{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"证据池：\n{json.dumps(evidence_payload, ensure_ascii=False, separators=(',', ':'))[:30000]}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=min(self.config.llm_temperature + 0.1, 0.3),
                timeout=self.config.llm_relation_timeout_seconds,
                max_tokens=1600,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            return self._parse_json(content)
        except Exception as exc:
            return {
                "status": "failed",
                "error": str(exc),
            }

    @staticmethod
    def build_user_prompt(node: Dict[str, Any], crawl_data: Dict[str, Any]) -> str:
        metadata = crawl_data.get("metadata", {})
        markdown = crawl_data.get("markdown") or crawl_data.get("content") or ""
        html_excerpt = crawl_data.get("html") or crawl_data.get("rawHtml") or crawl_data.get("raw_html") or ""
        links = crawl_data.get("links", [])
        evidence_snippets = AnalyzerLLMClient._build_evidence_snippets(
            "\n".join(str(item or "") for item in [metadata, markdown, html_excerpt])
        )
        if len(markdown) > 12000:
            markdown = markdown[:12000]
        if len(html_excerpt) > 8000:
            html_excerpt = html_excerpt[:8000]

        schema = {
            "page_classification": {
                "platform_family": "baidu_media | news | netease | forum | blog | telegram | douyin | dynamic_platform | unknown",
                "page_type": "news_article | forum_post | blog_post | channel_post | aggregator | feed_page | dynamic_platform | unknown",
                "confidence": 0.0,
                "evidence": [],
            },
            "content": {
                "title": None,
                "published_at": None,
                "modified_at": None,
                "publisher": None,
                "author": None,
                "canonical_url": None,
                "image_urls": [],
            },
            "image_occurrence": {
                "target_or_variant_present": "confirmed | probable | unclear | not_found",
                "occurrence_type": "same_image | edited_variant | screenshot_reference | unrelated | unknown",
                "caption": None,
                "image_credit": None,
                "evidence": [],
                "confidence": 0.0,
            },
            "provenance": {
                "content_role": "original_claim | repost_with_source | citation_or_embed | discussion_share | aggregator | unknown",
                "source_text": None,
                "source_url": None,
                "source_platform_hint": None,
                "source_account_hint": None,
                "confidence": 0.0,
                "evidence": [],
            },
            "optional_metrics": {
                "view_count": None,
                "comment_count": None,
                "like_count": None,
                "share_count": None,
                "repost_count": None,
            },
            "field_evidence": {},
            "node_decision": {
                "evidence_node_status": "confirmed_image_occurrence | possible_variant | contextual_only | rejected_after_page_review | inaccessible",
                "allow_in_external_timeline": False,
                "allow_cross_platform_relation_candidate": False,
                "reason": "",
            },
            "confidence": 0.0,
            "reason": "",
        }

        payload = {
            "node_id": node.get("id"),
            "upstream_id": node.get("upstream_id"),
            "url": node.get("url"),
            "canonical_url_hint": node.get("canonical_url"),
            "search_engine": node.get("engine"),
            "source_type": node.get("source_type"),
            "platform_family_hint": node.get("platform_family"),
            "page_type_hint": node.get("page_type_hint") or node.get("page_type"),
            "existing_title_hint": node.get("title"),
            "existing_similarity_readonly": node.get("similarity"),
            "possible_duplicate_readonly": node.get("possible_duplicate"),
            "validator_review_required": node.get("validator_review_required"),
            "validation_reason": node.get("validation_reason") or node.get("reason"),
            "validation_signals": node.get("validation_signals"),
            "image_variant": node.get("image_variant"),
            "suspected_tampering": node.get("suspected_tampering"),
            "watermark_detected": node.get("watermark_detected"),
            "watermark_text": node.get("watermark_text"),
            "watermark_platforms": node.get("watermark_platforms"),
            "watermark_accounts": node.get("watermark_accounts"),
            "crawl_status": crawl_data.get("crawl_status"),
            "crawl_source": crawl_data.get("crawl_source"),
            "crawl_error": crawl_data.get("error") or crawl_data.get("firecrawl_error"),
            "metadata": metadata,
            "markdown": markdown,
            "html_excerpt": html_excerpt,
            "evidence_snippets": evidence_snippets,
            "links": links[:50] if isinstance(links, list) else links,
            "required_schema": schema,
        }
        return (
            "Analyze this external web evidence node and return one strict JSON object matching required_schema. "
            "Use only the supplied crawl data. Preserve uncertainty; do not infer image occurrence from topic alone.\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _build_evidence_snippets(text: str) -> List[str]:
        if not text:
            return []
        compact = re.sub(r"\s+", " ", text)
        keywords = [
            "作者", "发布者", "账号", "博主", "UP主", "来源", "出处", "publisher", "author", "byline",
            "发布时间", "发布于", "发表于", "datePublished", "published", "pubdate", "created_at",
            "浏览", "阅读", "观看", "views", "view_count",
            "转发", "分享", "repost", "share_count", "shares",
            "评论", "comments", "comment_count",
            "点赞", "赞", "喜欢", "likes", "like_count",
            "收藏", "favorites", "collect",
            "粉丝", "followers", "following",
        ]
        snippets: List[str] = []
        seen: set[str] = set()
        lowered = compact.lower()
        for keyword in keywords:
            search_keyword = keyword.lower()
            start = 0
            while len(snippets) < 24:
                index = lowered.find(search_keyword, start)
                if index == -1:
                    break
                left = max(0, index - 220)
                right = min(len(compact), index + 360)
                snippet = compact[left:right].strip()
                fingerprint = snippet[:120]
                if snippet and fingerprint not in seen:
                    snippets.append(snippet)
                    seen.add(fingerprint)
                start = index + len(search_keyword)
            if len(snippets) >= 24:
                break
        return snippets

    @staticmethod
    def _parse_json(content: str) -> Optional[Dict[str, Any]]:
        content = content.strip()
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                parsed = json.loads(content[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None




class TimeSpaceAnalyzerAgent:
    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        tikomni_provider: Optional[TikomniProvider] = None,
        crawler: Optional[Any] = None,
        llm_client: Optional[AnalyzerLLMClient] = None,
    ) -> None:
        self.config = config or AnalyzerConfig()
        self.tikomni_provider = tikomni_provider or TikomniProvider(self.config)
        _ = crawler  # kept for backward-compatible construction; external crawl uses llmscrapy now.
        self.llmscrapy_crawler = LlmScrapyCrawler(self.config)
        self.llm_client = llm_client or AnalyzerLLMClient(self.config)

    def normalize_node_record(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize real upstream node fields into the Analyzer post protocol.

        This method intentionally reads from the node first. API crawlers, page
        parsing, LLM extraction, and mock data are fallbacks; they must not
        overwrite verified upstream fields.
        """
        existing = node.get("normalized_record")
        if isinstance(existing, dict) and existing.get("post") and existing.get("user"):
            return existing

        raw = self._first_mapping(node, ["raw_api_data", "api_data", "raw", "provider_raw"])
        status_obj = self._extract_status_object(node, raw)
        source_obj = self._first_mapping(status_obj, ["retweeted_status", "original_status", "source_status"])
        user_obj = self._first_mapping(status_obj, ["user", "author", "user_info", "author_info"])
        if not user_obj:
            user_obj = self._first_mapping(node, ["user", "author_info", "user_info"])
        xhs_user_profile = raw.get("xhs_user_profile") if isinstance(raw.get("xhs_user_profile"), dict) else {}
        xhs_basic_info = (
            xhs_user_profile.get("basicInfo", {})
            if isinstance(xhs_user_profile.get("basicInfo"), dict)
            else {}
        )
        xhs_followers = self._xhs_interaction_count(xhs_user_profile, "fans")
        xhs_following = self._xhs_interaction_count(xhs_user_profile, "follows")
        xhs_interactions = self._xhs_interaction_count(xhs_user_profile, "interaction")
        xhs_tags = self._xhs_profile_tags(xhs_user_profile)

        platform = self._normalize_platform(
            self._first_non_empty(node, ["platform", "source_platform", "source"]),
            str(node.get("url") or ""),
        )
        provider = str(
            self._first_non_empty(node, ["provider", "api_provider", "data_provider", "source_provider"])
            or "upper_data"
        )
        source_url = str(node.get("url") or node.get("source_url") or "")
        publish_raw = (
            self._first_non_empty(node, ["publish_time", "published_at", "created_at", "time"])
            or self._first_non_empty(status_obj, ["created_at", "publish_time", "published_at", "created_time", "time", "timestamp"])
        )
        publish_time = self.normalize_time(publish_raw)

        post_id = (
            self._first_non_empty(node, ["post_id", "note_id", "noteId", "mblogid", "mid", "idstr", "item_id"])
            or self._first_non_empty(status_obj, ["post_id", "note_id", "noteId", "idstr", "mid", "id", "mblogid"])
        )
        node_title = self._first_non_empty(node, ["title", "display_title", "name"])
        status_title = self._first_non_empty(status_obj, ["title", "display_title", "name"])
        node_text = self._first_non_empty(
            node,
            ["text", "content", "desc", "description", "snippet", "candidate_content_text"],
        )
        status_text = self._first_non_empty(status_obj, ["text", "content", "desc", "description", "longText.content"])
        if provider == "tikomni":
            title = status_title or node_title
            text = status_text or node_text
        else:
            title = node_title or status_title
            text = node_text or status_text
        image_urls = self._coerce_list(
            self._first_non_empty(node, ["image_urls", "images", "pic_urls", "pics", "image_url"])
            or self._first_non_empty(status_obj, ["image_urls", "images", "images_list", "pic_infos", "pics", "pic_urls", "share_info.image"])
        )
        image_urls = self._extract_image_urls(
            image_urls
            or self._first_non_empty(status_obj, ["images_list", "pic_infos", "share_info.image"])
        )

        username = self._first_non_empty(
            node,
            ["username", "nickname", "publisher", "author", "user_name"],
        ) or self._first_non_empty(user_obj, ["username", "screen_name", "nickname", "name"]) or self._first_non_empty(
            xhs_basic_info, ["nickname"]
        )
        user_id = self._first_non_empty(node, ["user_id", "uid", "author_id"]) or self._first_non_empty(
            user_obj, ["user_id", "uid", "idstr", "id", "userid", "userId"]
        )
        verified = self._boolish(
            self._first_non_empty(node, ["verified", "is_verified"])
            if self._first_non_empty(node, ["verified", "is_verified"]) is not None
            else self._first_non_empty(user_obj, ["verified", "is_verified", "red_official_verified", "show_red_official_verify_icon"])
        )

        metrics = {
            "view_count": self._to_int(
                self._first_non_empty(node, ["view_count", "read_count"])
                or self._first_non_empty(status_obj, ["view_count", "read_count", "views"])
            ),
            "like_count": self._to_int(
                self._first_non_empty(node, ["like_count", "liked_count", "attitudes_count"])
                or self._first_non_empty(status_obj, ["like_count", "liked_count", "attitudes_count", "likes"])
            ),
            "comment_count": self._to_int(
                self._first_non_empty(node, ["comment_count", "comments_count"])
                or self._first_non_empty(status_obj, ["comment_count", "comments_count", "comments"])
            ),
            "share_count": self._to_int(
                self._first_non_empty(node, ["share_count", "shares", "shared_count"])
                or self._first_non_empty(status_obj, ["share_count", "shared_count", "shares"])
            ),
            "repost_count": self._to_int(
                self._first_non_empty(node, ["repost_count", "reposts_count"])
                or self._first_non_empty(status_obj, ["repost_count", "reposts_count", "reposts", "shared_count"])
            ),
            "collect_count": self._to_int(
                self._first_non_empty(node, ["collect_count", "collected_count", "favorite_count"])
                or self._first_non_empty(status_obj, ["collect_count", "collected_count", "favorite_count"])
            ),
        }

        original_post_id = (
            self._first_non_empty(node, ["original_post_id", "retweeted_id", "retweeted_mid"])
            or self._first_non_empty(source_obj, ["idstr", "mid", "id", "post_id"])
        )
        is_repost = self._boolish(node.get("is_repost")) or bool(original_post_id) or bool(source_obj)
        source_candidates = self._coerce_list(node.get("source_candidates"))
        if source_obj:
            source_candidates.append(
                {
                    "post_id": str(self._first_non_empty(source_obj, ["idstr", "mid", "id"]) or ""),
                    "text": str(self._first_non_empty(source_obj, ["text", "content", "longText.content"]) or ""),
                    "publish_time": self.normalize_time(self._first_non_empty(source_obj, ["created_at", "publish_time"])),
                    "user": self._first_mapping(source_obj, ["user", "author", "user_info"]),
                    "evidence": "embedded_reposted_status",
                }
            )

        provider_status = self._first_mapping(node, ["provider_status", "api_status"])
        status = str(provider_status.get("status") or ("success" if any([post_id, publish_time, text, username]) else "skipped"))
        missing_fields = self._normalized_missing_fields(post_id, text or title, publish_time, username)

        return {
            "ok": status not in {"failed", "error"} and not missing_fields[:3],
            "provider": provider,
            "platform": platform,
            "source_url": source_url,
            "canonical_url": str(
                self._first_non_empty(node, ["canonical_url", "final_url"]) or source_url
            ),
            "post": {
                "post_id": str(post_id or ""),
                "title": str(title or ""),
                "text": str(text or ""),
                "publish_time": publish_time or "",
                "publish_time_raw": str(publish_raw or ""),
                "image_urls": image_urls,
                "source_text": str(self._first_non_empty(node, ["source_text"]) or self._first_non_empty(status_obj, ["source", "source_text"]) or ""),
                "source_url": str(self._first_non_empty(node, ["source_url"]) or self._first_non_empty(status_obj, ["source_url", "original_url", "share_info.link"]) or ""),
                "original_post_id": str(original_post_id or ""),
                "is_repost": bool(is_repost),
            },
            "user": {
                "user_id": str(user_id or ""),
                "username": str(username or ""),
                "nickname": str(
                    self._first_non_empty(node, ["nickname"])
                    or self._first_non_empty(user_obj, ["nickname", "screen_name", "name"])
                    or username
                    or ""
                ),
                "homepage_url": str(self._first_non_empty(node, ["homepage_url", "profile_url"]) or self._first_non_empty(user_obj, ["homepage_url", "profile_url", "url"]) or xhs_user_profile.get("share_link") or ""),
                "verified": verified,
                "verified_type": str(self._first_non_empty(node, ["verified_type"]) or self._first_non_empty(user_obj, ["verified_type", "verify_type", "red_official_verify_type"]) or ""),
                "verified_reason": str(
                    self._first_non_empty(node, ["verified_reason"])
                    or self._first_non_empty(user_obj, ["verified_reason", "verify_reason"])
                    or xhs_user_profile.get("red_official_verify_base_info")
                    or ", ".join(xhs_tags)
                    or ""
                ),
                "followers_count": self._to_int(self._first_non_empty(node, ["followers_count", "follower_count", "fans_count", "follower_count"]) or self._first_non_empty(user_obj, ["followers_count", "fans_count", "follower_count"]) or xhs_followers),
                "following_count": self._to_int(self._first_non_empty(node, ["following_count", "friends_count", "follow_count"]) or self._first_non_empty(user_obj, ["following_count", "friends_count", "follow_count"]) or xhs_following),
                "statuses_count": self._to_int(self._first_non_empty(node, ["statuses_count", "post_count", "notes_count"]) or self._first_non_empty(user_obj, ["statuses_count", "post_count", "notes_count"])),
                "description": str(self._first_non_empty(node, ["user_description", "description", "bio"]) or self._first_non_empty(user_obj, ["description", "desc", "bio"]) or xhs_basic_info.get("desc") or ""),
                "avatar_url": str(self._first_non_empty(node, ["avatar_url", "avatar"]) or self._first_non_empty(user_obj, ["avatar_url", "avatar", "avatar_hd", "profile_image_url", "image"]) or xhs_basic_info.get("images") or xhs_basic_info.get("imageb") or ""),
                "red_id": str(self._first_non_empty(user_obj, ["red_id", "redId"]) or xhs_basic_info.get("redId") or ""),
                "interaction_count": xhs_interactions,
            },
            "metrics": metrics,
            "relations": {
                "mentioned_accounts": self._coerce_list(node.get("mentioned_accounts") or self._first_non_empty(status_obj, ["mentioned_accounts", "mentions", "at_users"])),
                "linked_urls": self._coerce_list(node.get("linked_urls") or self._first_non_empty(status_obj, ["linked_urls", "urls", "url_struct"])),
                "repost_chain_sample": self._coerce_list(node.get("repost_chain_sample") or node.get("reposts")),
                "source_candidates": source_candidates,
            },
            "provider_status": {
                "status": status,
                "latency_ms": self._to_int(provider_status.get("latency_ms")),
                "missing_fields": missing_fields,
                "error": str(provider_status.get("error") or ""),
                "raw_endpoint": str(provider_status.get("raw_endpoint") or provider_status.get("endpoint") or ""),
            },
            "raw": raw,
        }

    def enrich_node_with_tikomni(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch Tikomni structured metadata and attach a normalized record when available."""
        provider_result = self.tikomni_provider.fetch_node(node)
        status = str(provider_result.get("status") or "unknown")

        enriched = {
            **node,
            "tikomni_status": status,
            "tikomni_error": provider_result.get("error") or "",
            "tikomni_latency_ms": provider_result.get("latency_ms") or 0,
            "tikomni_cache_hit": bool(provider_result.get("cache_hit")),
            "api_provider": "tikomni",
            "api_status": status,
            "api_latency_ms": provider_result.get("latency_ms") or 0,
        }
        if status != "success":
            return enriched

        raw = provider_result.get("raw") if isinstance(provider_result.get("raw"), dict) else {}
        temp_node = {
            **node,
            "platform": provider_result.get("platform") or node.get("platform") or "weibo",
            "provider": "tikomni",
            "api_provider": "tikomni",
            "raw_api_data": raw,
            "provider_status": {
                "status": "success",
                "latency_ms": provider_result.get("latency_ms") or 0,
                "missing_fields": [],
                "error": "",
                "raw_endpoint": provider_result.get("raw_endpoint") or "",
            },
        }
        temp_node.pop("normalized_record", None)
        normalized_record = self.normalize_node_record(temp_node)
        provider_status = normalized_record.get("provider_status")
        if isinstance(provider_status, dict):
            provider_status["missing_fields"] = self._normalized_missing_fields(
                self._nested_get(normalized_record, "post.post_id"),
                self._nested_get(normalized_record, "post.text")
                or self._nested_get(normalized_record, "post.title"),
                self._nested_get(normalized_record, "post.publish_time"),
                self._nested_get(normalized_record, "user.username")
                or self._nested_get(normalized_record, "user.nickname"),
            )
        return {
            **enriched,
            "normalized_record": normalized_record,
            "provider": "tikomni",
            "api_provider": "tikomni",
            "api_status": self._nested_get(normalized_record, "provider_status.status"),
            "api_latency_ms": self._nested_get(normalized_record, "provider_status.latency_ms"),
            "api_missing_fields": self._nested_get(normalized_record, "provider_status.missing_fields") or [],
        }

    def _analysis_from_normalized_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        post = record.get("post", {}) if isinstance(record.get("post"), dict) else {}
        user = record.get("user", {}) if isinstance(record.get("user"), dict) else {}
        metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
        return {
            "platform": record.get("platform") or "other",
            "published_at": post.get("publish_time"),
            "publisher": user.get("username") or user.get("nickname"),
            "author": user.get("username") or user.get("nickname"),
            "title": post.get("title"),
            "description": post.get("text"),
            "view_count": metrics.get("view_count"),
            "repost_count": metrics.get("repost_count"),
            "comment_count": metrics.get("comment_count"),
            "like_count": metrics.get("like_count"),
            "collect_count": metrics.get("collect_count"),
            "follower_count": user.get("followers_count"),
            "following_count": user.get("following_count"),
            "public_relation_hint": post.get("source_text") or post.get("source_url"),
            "date_source": str(record.get("provider") or "upper_data") if post.get("publish_time") else "",
            "time_evidence": {
                "upper_data": str(post.get("publish_time") or ""),
                "provider": str(record.get("provider") or ""),
                "raw": str(post.get("publish_time_raw") or ""),
            },
            "propagation_role": "转载/转发" if post.get("is_repost") else "候选发布节点",
            "llm_confidence": 0.0,
        }

    def _extract_status_object(self, node: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
        existing = self._first_mapping(node, ["post", "status", "note", "item"])
        if existing:
            return existing
        candidates = [
            "xhs_note",
            "status_detail.data.detailInfo.status",
            "data.detailInfo.status",
            "detailInfo.status",
            "data.data.0.note_list.0",
            "data.status",
            "data.note",
            "data.items.0",
            "result",
            "data",
        ]
        for path in candidates:
            value = self._nested_get(raw, path)
            if isinstance(value, dict):
                return value
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _normalize_platform(platform: Any, url: str = "") -> str:
        raw = str(platform or "").strip().lower()
        aliases = {
            "wb": "weibo",
            "微博": "weibo",
            "xhs": "xiaohongshu",
            "redbook": "xiaohongshu",
            "小红书": "xiaohongshu",
        }
        if raw in aliases:
            return aliases[raw]
        host = urlparse(url or "").netloc.lower()
        if "weibo." in host or "weibo.cn" in host:
            return "weibo"
        if "xiaohongshu.com" in host or "xhslink.com" in host:
            return "xiaohongshu"
        if raw in {"other", "unknown", "未知"}:
            return "other"
        return "other"

    def _first_non_empty(self, data: Dict[str, Any], keys: List[str]) -> Any:
        for key in keys:
            value = self._nested_get(data, key)
            if value not in (None, "", [], {}):
                return value
        return None

    def _first_mapping(self, data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        for key in keys:
            value = self._nested_get(data, key)
            if isinstance(value, dict):
                return value
        return {}

    @staticmethod
    def _nested_get(data: Any, path: str) -> Any:
        current = data
        for part in path.split("."):
            if isinstance(current, list) and part.isdigit():
                index = int(part)
                current = current[index] if index < len(current) else None
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    @staticmethod
    def _coerce_list(value: Any) -> List[Any]:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _extract_image_urls(self, value: Any) -> List[str]:
        urls: List[str] = []
        if not value:
            return urls
        if isinstance(value, str):
            return [value] if value.startswith("http") else []
        if isinstance(value, list):
            for item in value:
                urls.extend(self._extract_image_urls(item))
        elif isinstance(value, dict):
            direct = self._first_non_empty(
                value,
                [
                    "url",
                    "src",
                    "url_size_large",
                    "original",
                    "original.url",
                    "largest.url",
                    "large.url",
                    "bmiddle.url",
                    "thumbnail.url",
                    "url_multi_level.high",
                    "url_multi_level.medium",
                    "share_info.image",
                ],
            )
            if isinstance(direct, str) and direct.startswith("http"):
                urls.append(direct)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    urls.extend(self._extract_image_urls(child))
        return list(dict.fromkeys(urls))

    @staticmethod
    def _xhs_interaction_count(profile: Dict[str, Any], interaction_type: str) -> int:
        interactions = profile.get("interactions")
        if not isinstance(interactions, list):
            return 0
        for item in interactions:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "") == interaction_type:
                value = item.get("count") or item.get("i18nCount")
                return TimeSpaceAnalyzerAgent._to_int(value)
        return 0

    @staticmethod
    def _xhs_profile_tags(profile: Dict[str, Any]) -> List[str]:
        tags = profile.get("tags")
        if not isinstance(tags, list):
            return []
        result: List[str] = []
        for item in tags:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name:
                result.append(name)
        return result

    @staticmethod
    def _boolish(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "verified"}

    @staticmethod
    def _normalized_missing_fields(post_id: Any, text_or_title: Any, publish_time: Any, username: Any) -> List[str]:
        missing = []
        if not post_id:
            missing.append("post.post_id")
        if not text_or_title:
            missing.append("post.text or post.title")
        if not publish_time:
            missing.append("post.publish_time")
        if not username:
            missing.append("user.username or user.nickname")
        return missing

    def parse(self, state: AgentState) -> AgentState:
        logs = append_log(
            state,
            "parse_node: using structured API for mainstream platforms; llmscrapy pipeline for other platforms; explicit evidence required for topology edges.",
        )
        started_at = time.perf_counter()

        input_nodes = self.preprocess_input_nodes(state.get("nodes_data") or state.get("nodes") or [])
        self._append_progress_log(
            logs,
            f"{self._progress_bar(1, 6)} [stage 1/6] input loaded: {len(input_nodes)} candidate posts",
        )

        analyzed_nodes = self._analyze_nodes(input_nodes, logs=logs)

        self._append_progress_log(logs, f"{self._progress_bar(3, 6)} [stage 3/6] sorting posts by publish time")
        analyzed_nodes.sort(key=self._sort_key)
        self._append_progress_log(logs, f"{self._progress_bar(4, 6)} [stage 4/6] assigning sources, key nodes and post-to-post edges")
        self._assign_topology(analyzed_nodes)
        self._assign_matrix_candidates(analyzed_nodes)
        self._assign_duplicate_clusters(analyzed_nodes)
        self._append_progress_log(logs, f"{self._progress_bar(5, 6)} [stage 5/6] building topology payload")
        topology_data = self.build_topology_data(analyzed_nodes)
        analysis_summary = self.build_analysis_summary(analyzed_nodes, topology_data)
        mermaid_graph = self.build_mermaid_graph(analyzed_nodes)
        elapsed_seconds = round(time.perf_counter() - started_at, 2)
        topology_data["runtime"] = {
            "elapsed_seconds": elapsed_seconds,
            "elapsed_human": self._format_elapsed(elapsed_seconds),
            "node_count": len(analyzed_nodes),
            "edge_count": len(topology_data.get("edges") or []),
        }
        self._append_progress_log(
            logs,
            f"{self._progress_bar(6, 6)} [stage 6/6] analyzer finished in {self._format_elapsed(elapsed_seconds)}; "
            f"nodes={len(analyzed_nodes)}, edges={len(topology_data.get('edges') or [])}",
        )

        return {
            "nodes_data": analyzed_nodes,
            "mermaid_graph": mermaid_graph,
            "topology_data": topology_data,
            "analysis_summary": analysis_summary,
            "execution_logs": logs,
        }

    @staticmethod
    def _progress_bar(done: int, total: int, width: int = 18) -> str:
        total = max(total, 1)
        done = min(max(done, 0), total)
        filled = int(round(width * done / total))
        return f"[{'#' * filled}{'-' * (width - filled)}] {done / total:>6.1%}"

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        seconds = max(float(seconds or 0), 0.0)
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes, rem = divmod(seconds, 60)
        if minutes < 60:
            return f"{int(minutes)}m {rem:.1f}s"
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours)}h {int(minutes)}m {rem:.1f}s"

    @staticmethod
    def _append_progress_log(logs: List[str], message: str) -> None:
        log_line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        print(log_line)
        logs.append(log_line)

    def preprocess_input_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make node ids unique in-place by position semantics without reordering nodes."""
        seen: Dict[str, int] = {}
        prepared: List[Dict[str, Any]] = []
        for index, original in enumerate(nodes or []):
            node = dict(original)
            raw_id = str(node.get("id") or f"node_{index + 1:04d}")
            count = seen.get(raw_id, 0)
            seen[raw_id] = count + 1
            node["upstream_id"] = str(node.get("upstream_id") or raw_id)
            node["id"] = raw_id if count == 0 else f"{raw_id}__dup{count}"
            node["input_order"] = self._to_int(node.get("input_order")) if node.get("input_order") is not None else index

            url = str(node.get("url") or node.get("page_url") or node.get("source_url") or "")
            canonical_url = str(node.get("canonical_url") or "").strip() or self._canonicalize_url(url)
            if canonical_url:
                node["canonical_url"] = canonical_url

            classification = self.classify_external_page(node)
            node.setdefault("platform_family", classification.get("platform_family", "unknown"))
            node.setdefault("page_type_hint", classification.get("page_type", "unknown"))
            node.setdefault("page_type", node.get("page_type_hint"))
            node["validator_review_required"] = self._validator_review_required(node)
            prepared.append(node)
        return prepared

    def classify_external_page(self, node: Dict[str, Any]) -> Dict[str, str]:
        url = str(node.get("url") or node.get("canonical_url") or node.get("source_url") or "")
        platform = self._normalize_platform(node.get("platform"), url)
        if platform == "weibo":
            return {"platform_family": "weibo", "page_type": "social_post"}
        if platform == "xiaohongshu":
            return {"platform_family": "xiaohongshu", "page_type": "social_post"}

        host = urlparse(url or "").netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        rules = [
            (("baijiahao.baidu.com",), "baidu_media", "news_article"),
            (("mbd.baidu.com", "baidu.com"), "baidu_media", "aggregator"),
            (("163.com",), "netease", "news_article"),
            (("sohu.com", "guancha.cn", "inf.news", "thepaper.cn", "ifeng.com", "qq.com", "sina.com.cn"), "news", "news_article"),
            (("t.me", "telegram.me"), "telegram", "channel_post"),
            (("voz.vn", "reddit.com", "forum", "bbs."), "forum", "forum_post"),
            (("livejournal.com", "blog.jp", "ameblo.jp", "fc2.com", "blogspot.", "wordpress."), "blog", "blog_post"),
            (("douyin.com", "iesdouyin.com"), "douyin", "dynamic_platform"),
        ]
        for markers, family, page_type in rules:
            if any(marker in host for marker in markers):
                return {"platform_family": family, "page_type": page_type}
        return {"platform_family": "unknown", "page_type": "unknown"}

    @staticmethod
    def _canonicalize_url(url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return url.strip()
        return parsed._replace(fragment="", query="").geturl().rstrip("/")

    def _validator_review_required(self, node: Dict[str, Any]) -> bool:
        if self._boolish(node.get("validator_review_required")):
            return True
        if self._boolish(node.get("possible_duplicate")):
            return True
        if self._boolish(node.get("suspected_tampering")):
            return True
        if node.get("validation_signals") or node.get("image_variant") or node.get("tampering_signals"):
            return True
        reason = str(node.get("reason") or node.get("validation_reason") or "").lower()
        return any(marker in reason for marker in ("tamper", "variant", "duplicate", "review", "suspected"))

    def _analyze_nodes(self, input_nodes: List[Dict[str, Any]], logs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if not input_nodes:
            return []

        nodes = [dict(node) for node in input_nodes]
        total = len(nodes)
        llmscrapy_batch = [
            (index, node)
            for index, node in enumerate(nodes)
            if index < self.config.max_llmscrapy_nodes
            and self._normalize_platform(node.get("platform"), str(node.get("url") or "")) not in {"weibo", "xiaohongshu"}
            and str(node.get("url") or "")
        ]
        if logs is not None:
            self._append_progress_log(
                logs,
                f"{self._progress_bar(0, total)} [stage 2/6] crawler/API enrichment started; "
                f"analyzer_workers={min(max(self.config.max_workers, 1), total)} "
                f"llmscrapy_workers={self.config.llmscrapy_max_workers}",
            )
        if llmscrapy_batch:
            if logs is not None:
                self._append_progress_log(
                    logs,
                    f"{self._progress_bar(0, total)} llmscrapy batch started "
                    f"({len(llmscrapy_batch)} external nodes, workers={self.config.llmscrapy_max_workers})",
                )
            llmscrapy_results = self.llmscrapy_crawler.scrape_nodes(llmscrapy_batch)
            for index, result in llmscrapy_results.items():
                nodes[index]["_llmscrapy_prefetch_result"] = result
            if logs is not None:
                success_count = sum(
                    1
                    for result in llmscrapy_results.values()
                    if str(result.get("llmscrapy_status") or "") == "success"
                )
                self._append_progress_log(
                    logs,
                    f"{self._progress_bar(success_count, max(len(llmscrapy_batch), 1))} "
                    f"llmscrapy batch finished; success={success_count}/{len(llmscrapy_batch)}",
                )

        max_workers = min(max(self.config.max_workers, 1), total)
        if max_workers <= 1:
            analyzed_nodes = []
            for index, node in enumerate(nodes):
                analyzed = self._analyze_node_safely(node=node, index=index)
                analyzed_nodes.append(analyzed)
                if logs is not None:
                    self._append_progress_log(
                        logs,
                        f"{self._progress_bar(index + 1, total)} crawler/API progress "
                        f"({index + 1}/{total}) node={analyzed.get('id')}",
                    )
            return analyzed_nodes

        analyzed_nodes: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._analyze_node_safely, node, index): index
                for index, node in enumerate(nodes)
            }
            for future in as_completed(future_to_index):
                analyzed = future.result()
                analyzed_nodes.append(analyzed)
                if logs is not None:
                    done = len(analyzed_nodes)
                    self._append_progress_log(
                        logs,
                        f"{self._progress_bar(done, total)} crawler/API progress ({done}/{total}) node={analyzed.get('id')}",
                    )
        return analyzed_nodes

    def _analyze_node_safely(self, node: Dict[str, Any], index: int) -> Dict[str, Any]:
        try:
            return self.analyze_node(node, index)
        except Exception as exc:
            return {
                **node,
                "published_at": node.get("published_at"),
                "publisher": node.get("publisher") or node.get("author"),
                "view_count": self._to_int(node.get("view_count")),
                "repost_count": self._to_int(node.get("repost_count")),
                "comment_count": self._to_int(node.get("comment_count")),
                "like_count": self._to_int(node.get("like_count")),
                "follower_count": self._to_int(node.get("follower_count")),
                "following_count": self._to_int(node.get("following_count")),
                "propagation_role": node.get("propagation_role", "未知"),
                "crawl_status": "failed",
                "crawl_source": "analyzer",
                "llm_status": "skipped",
                "llm_reason": f"skipped because analyzer failed: {exc}",
                "llm_used": False,
                "date_source": node.get("date_source", "missing"),
                "time_evidence": node.get("time_evidence", {}),
                "llm_confidence": self._clamp_float(node.get("llm_confidence")),
                "node_weight": 0.0,
                "source_score": 0.0,
                "parent_id": None,
                "edge_weight_from_parent": 0.0,
                "edge_method_from_parent": "",
                "edge_score_components_from_parent": {},
                "is_suspected_source": False,
                "is_key_node": False,
                "is_topology_visible": True,
                "topology_omit_reason": None,
                "analyzer_reason": f"Analyzer failed for this node: {exc}",
                "input_order": index,
            }

    def analyze_node(self, node: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
        node = self.enrich_node_with_tikomni(node)
        prefetched_llmscrapy_result = (
            node.get("_llmscrapy_prefetch_result")
            if isinstance(node.get("_llmscrapy_prefetch_result"), dict)
            else None
        )
        if "_llmscrapy_prefetch_result" in node:
            node = {key: value for key, value in node.items() if key != "_llmscrapy_prefetch_result"}
        url = str(node.get("url", ""))
        domain = urlparse(url).netloc if url else ""
        normalized_record = self.normalize_node_record(node)
        node_platform = self._normalize_platform(
            normalized_record.get("platform") or node.get("platform"),
            url,
        )
        is_mainstream_platform = node_platform in {"weibo", "xiaohongshu"}
        page_classification = self.classify_external_page(node)
        platform_family = str(node.get("platform_family") or page_classification.get("platform_family") or node_platform)
        page_type = str(node.get("page_type") or node.get("page_type_hint") or page_classification.get("page_type") or "unknown")
        validator_review_required = self._validator_review_required(node)
        upper_rule_data = self._analysis_from_normalized_record(normalized_record)
        provider = str(normalized_record.get("provider") or node.get("api_provider") or node.get("provider") or "")
        has_verified_structured_fields = (
            is_mainstream_platform
            and provider in {"tikomni", "platform_adapter"}
            and bool(upper_rule_data.get("published_at") or upper_rule_data.get("publisher"))
        )
        llmscrapy_status = "skipped"
        llmscrapy_error = ""
        llmscrapy_llm_data: Dict[str, Any] = {}

        if has_verified_structured_fields:
            crawl_data = {
                "crawl_status": "skipped",
                "crawl_source": "verified_structured_data",
                "metadata": {},
                "markdown": "",
                "html": "",
                "final_url": url,
                "error": "skipped because verified platform structured fields are available",
            }
        elif is_mainstream_platform:
            crawl_data = {
                "crawl_status": "skipped",
                "crawl_source": "structured_api_only",
                "metadata": {},
                "markdown": "",
                "html": "",
                "final_url": url,
                "error": "skipped because mainstream platforms use structured API/upstream fields",
            }
        elif prefetched_llmscrapy_result:
            llmscrapy_result = prefetched_llmscrapy_result
            llmscrapy_status = str(llmscrapy_result.get("llmscrapy_status") or "unknown")
            llmscrapy_error = str(llmscrapy_result.get("llmscrapy_error") or "")
            llmscrapy_llm_data = (
                llmscrapy_result.get("llm_data")
                if isinstance(llmscrapy_result.get("llm_data"), dict)
                else {}
            )
            crawl_data = (
                llmscrapy_result.get("crawl_data")
                if isinstance(llmscrapy_result.get("crawl_data"), dict)
                else {}
            )
            if not crawl_data:
                llmscrapy_error = llmscrapy_error or "llmscrapy returned no crawl data"
                crawl_data = {
                    "crawl_status": "failed",
                    "crawl_source": "llmscrapy",
                    "metadata": {},
                    "markdown": "",
                    "html": "",
                    "final_url": url,
                    "error": llmscrapy_error,
                }
        elif not self.llmscrapy_crawler.enabled:
            llmscrapy_error = self.llmscrapy_crawler.reason
            crawl_data = {
                "crawl_status": "skipped",
                "crawl_source": "llmscrapy",
                "metadata": {},
                "markdown": "",
                "html": "",
                "final_url": url,
                "error": llmscrapy_error,
            }
        elif index >= self.config.max_llmscrapy_nodes:
            llmscrapy_error = "skipped by ANALYZER_MAX_LLMSCRAPY_NODES"
            crawl_data = {
                "crawl_status": "skipped",
                "crawl_source": "llmscrapy",
                "metadata": {},
                "markdown": "",
                "html": "",
                "final_url": url,
                "error": llmscrapy_error,
            }
        else:
            llmscrapy_result = self.llmscrapy_crawler.scrape_node(node)
            llmscrapy_status = str(llmscrapy_result.get("llmscrapy_status") or "unknown")
            llmscrapy_error = str(llmscrapy_result.get("llmscrapy_error") or "")
            llmscrapy_llm_data = (
                llmscrapy_result.get("llm_data")
                if isinstance(llmscrapy_result.get("llm_data"), dict)
                else {}
            )
            crawl_data = (
                llmscrapy_result.get("crawl_data")
                if isinstance(llmscrapy_result.get("crawl_data"), dict)
                else {}
            )
            if not crawl_data:
                llmscrapy_error = llmscrapy_error or "llmscrapy returned no crawl data"
                crawl_data = {
                    "crawl_status": "failed",
                    "crawl_source": "llmscrapy",
                    "metadata": {},
                    "markdown": "",
                    "html": "",
                    "final_url": url,
                    "error": llmscrapy_error,
                }

        firecrawl_status = "skipped"
        firecrawl_error = "replaced by llmscrapy pipeline for external platforms" if not is_mainstream_platform else ""
        crawl_status = str(crawl_data.get("crawl_status", "unknown"))
        crawl_source = str(crawl_data.get("crawl_source") or ("llmscrapy" if not is_mainstream_platform else "none"))
        mock_data = (
            dict(MOCK_PARSE_RESULT.get(str(node.get("id")), {}))
            if self.config.enable_mock_fallback and not has_verified_structured_fields
            else {}
        )
        rule_data = self.extract_by_rules(crawl_data) if crawl_status in {"success", "limited", "skipped"} else {}
        rule_data = self._merge_analysis(rule_data, upper_rule_data, {"platform_family": platform_family, "page_type": page_type})
        llm_data = llmscrapy_llm_data or {}
        llm_data = self._flatten_llm_external_analysis(llm_data)
        if not is_mainstream_platform and crawl_source == "llmscrapy":
            self._remove_rule_derived_external_metrics(rule_data)
        if is_mainstream_platform:
            llm_status = "skipped"
            llm_reason = "skipped because mainstream platforms use structured API/upstream fields"
        elif llmscrapy_status == "success" and llm_data:
            if llm_data.get("published_at"):
                llm_status = "success"
                llm_reason = "extracted by llmscrapy"
            else:
                llm_status = "success_partial"
                llm_reason = str(llm_data.get("reason") or "llmscrapy returned structured data without publish time")
        elif llmscrapy_status == "success":
            llm_status = "success_partial"
            llm_reason = "llmscrapy crawled the page but returned no structured metadata"
        elif llmscrapy_status == "skipped":
            llm_status = "skipped"
            llm_reason = llmscrapy_error or "llmscrapy skipped"
        else:
            llm_status = "failed"
            llm_reason = llmscrapy_error or "llmscrapy failed"

        merged = self._merge_analysis(mock_data, node, rule_data, llm_data)
        published_at = self.normalize_time(merged.get("published_at"))
        if published_at:
            merged["published_at"] = published_at
            if not merged.get("date_source"):
                merged["date_source"] = "llm" if llm_data.get("published_at") else "page_metadata"
            if not merged.get("time_evidence"):
                merged["time_evidence"] = {
                    "search_result": "",
                    "page_metadata": "",
                    "time_tag": "",
                    "visible_text": published_at if merged.get("date_source") == "llm" else "",
                    "url_pattern": "",
                    "http_last_modified": "",
                    "llm": published_at if merged.get("date_source") == "llm" else "",
                }

        node_decision = (
            {}
            if is_mainstream_platform
            else self.qualify_external_node(node, crawl_data, rule_data, llm_data)
        )

        if crawl_status in {"success", "limited"}:
            reason = (
                merged.get("reason")
                or merged.get("analyzer_reason")
                or f"{crawl_source} 抓取成功并完成解析。"
            )
        elif mock_data:
            reason = merged.get("analyzer_reason") or "llmscrapy 不可用，使用 Mock 数据回退。"
        else:
            reason = crawl_data.get("error") or "未能获取可靠网页解析结果。"

        external_llmscrapy_metrics = not is_mainstream_platform and crawl_source == "llmscrapy"
        enriched = {
            **node,
            "id": node.get("id")
            or self._nested_get(normalized_record, "post.post_id")
            or node.get("note_id")
            or node.get("url")
            or f"node_{index + 1:04d}",
            "input_order": self._to_int(node.get("input_order")) if node.get("input_order") is not None else index,
            "domain": domain,
            "canonical_url": merged.get("canonical_url") or node.get("canonical_url") or url,
            "platform_family": merged.get("platform_family") or platform_family,
            "page_type": merged.get("page_type") or page_type,
            "page_type_hint": node.get("page_type_hint") or page_type,
            "validator_review_required": validator_review_required,
            "normalized_record": normalized_record,
            "platform": normalized_record.get("platform") or merged.get("platform") or node.get("platform") or "other",
            "api_provider": normalized_record.get("provider"),
            "api_status": self._nested_get(normalized_record, "provider_status.status"),
            "api_latency_ms": self._nested_get(normalized_record, "provider_status.latency_ms"),
            "api_missing_fields": self._nested_get(normalized_record, "provider_status.missing_fields") or [],
            "title": merged.get("title") or node.get("title") or self._nested_get(normalized_record, "post.title"),
            "description": (
                merged.get("description")
                or node.get("description")
                or node.get("candidate_content_text")
                or self._nested_get(normalized_record, "post.text")
            ),
            "published_at": merged.get("published_at"),
            "publisher": merged.get("publisher") or merged.get("author"),
            "view_count": self._count_for_node_output(merged, "view_count", external_llmscrapy_metrics),
            "repost_count": self._count_for_node_output(merged, "repost_count", external_llmscrapy_metrics),
            "comment_count": self._count_for_node_output(merged, "comment_count", external_llmscrapy_metrics),
            "like_count": self._count_for_node_output(merged, "like_count", external_llmscrapy_metrics),
            "collect_count": self._count_for_node_output(merged, "collect_count", external_llmscrapy_metrics),
            "follower_count": self._to_int(merged.get("follower_count")),
            "following_count": self._to_int(merged.get("following_count")),
            "image_urls": self._coerce_list(
                merged.get("image_urls")
                or self._nested_get(normalized_record, "post.image_urls")
                or node.get("image_urls")
                or []
            ),
            "thumbnail_url": (
                node.get("thumbnail_url")
                or node.get("image_url")
                or self._first_non_empty(normalized_record, ["post.image_urls.0"])
                or ""
            ),
            "public_relation_hint": merged.get("public_relation_hint"),
            "source_text": merged.get("source_text") or node.get("source_text"),
            "source_url": merged.get("source_url") or node.get("source_url"),
            "image_caption": merged.get("image_caption") or node.get("image_caption"),
            "image_credit": merged.get("image_credit") or node.get("image_credit"),
            "llm_page_analysis": merged.get("llm_page_analysis") or {},
            "image_occurrence": merged.get("image_occurrence") or {},
            "provenance": merged.get("provenance") or {},
            "field_evidence": merged.get("field_evidence") or {},
            "node_decision": node_decision or merged.get("node_decision") or {},
            "evidence_node_status": (node_decision or {}).get("evidence_node_status") or merged.get("evidence_node_status"),
            "allow_in_external_timeline": bool((node_decision or {}).get("allow_in_external_timeline")),
            "allow_cross_platform_relation_candidate": bool(
                (node_decision or {}).get("allow_cross_platform_relation_candidate")
            ),
            "node_decision_reason": (node_decision or {}).get("reason") or merged.get("node_decision_reason") or "",
            "location_hint": merged.get("location_hint"),
            "propagation_role": merged.get("propagation_role", "未知"),
            "crawl_status": crawl_status,
            "crawl_source": crawl_source,
            "firecrawl_status": firecrawl_status,
            "firecrawl_error": firecrawl_error,
            "llmscrapy_status": llmscrapy_status,
            "llmscrapy_error": llmscrapy_error,
            "llmscrapy_used": bool(llmscrapy_llm_data),
            "llm_status": llm_status,
            "llm_reason": llm_reason,
            "llm_used": bool(llmscrapy_llm_data and llm_data),
            "date_source": merged.get("date_source", node.get("date_source")),
            "time_evidence": merged.get(
                "time_evidence",
                node.get(
                    "time_evidence",
                    {
                        "search_result": "",
                        "page_metadata": "",
                        "time_tag": "",
                        "visible_text": "",
                        "url_pattern": "",
                        "http_last_modified": "",
                    },
                ),
            ),
            "llm_confidence": self._clamp_float(
                merged.get("confidence", merged.get("llm_confidence", 0.0))
            ),
            "node_weight": 0.0,
            "source_score": 0.0,
            "parent_id": None,
            "edge_weight_from_parent": 0.0,
            "edge_method_from_parent": "",
            "edge_score_components_from_parent": {},
            "is_suspected_source": False,
            "is_key_node": False,
            "is_topology_visible": True,
            "topology_omit_reason": None,
            "analyzer_reason": str(reason),
        }
        enriched["tamper_analysis"] = self.analyze_tampering(enriched)
        enriched["influence_analysis"] = self.analyze_influence(enriched)
        enriched["is_key_node"] = bool(enriched["influence_analysis"].get("is_key_node"))
        enriched["is_big_v"] = bool(enriched["influence_analysis"].get("is_big_v"))
        enriched["is_matrix_account_candidate"] = False
        return enriched

    def qualify_external_node(
        self,
        node: Dict[str, Any],
        crawl_data: Dict[str, Any],
        rule_data: Dict[str, Any],
        llm_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        llm_decision = llm_data.get("node_decision") if isinstance(llm_data.get("node_decision"), dict) else {}
        if llm_decision:
            status = str(llm_decision.get("evidence_node_status") or "contextual_only")
            return {
                "evidence_node_status": status,
                "allow_in_external_timeline": self._boolish(llm_decision.get("allow_in_external_timeline")),
                "allow_cross_platform_relation_candidate": self._boolish(
                    llm_decision.get("allow_cross_platform_relation_candidate")
                ),
                "reason": str(llm_decision.get("reason") or "LLM node decision"),
            }

        crawl_status = str(crawl_data.get("crawl_status") or "")
        if crawl_status not in {"success", "limited", "skipped"}:
            return {
                "evidence_node_status": "inaccessible",
                "allow_in_external_timeline": False,
                "allow_cross_platform_relation_candidate": False,
                "reason": str(crawl_data.get("error") or crawl_data.get("firecrawl_error") or "page inaccessible"),
            }

        occurrence = llm_data.get("image_occurrence") if isinstance(llm_data.get("image_occurrence"), dict) else {}
        present = str(occurrence.get("target_or_variant_present") or "").lower()
        occurrence_type = str(occurrence.get("occurrence_type") or "").lower()
        has_image_evidence = bool(
            rule_data.get("image_urls")
            or node.get("image_url")
            or node.get("image_urls")
            or occurrence.get("evidence")
        )
        has_source_evidence = bool(
            rule_data.get("source_url")
            or rule_data.get("source_text")
            or rule_data.get("public_relation_hint")
            or llm_data.get("source_url")
            or llm_data.get("source_text")
        )

        if present == "confirmed" or occurrence_type in {"same_image", "edited_variant", "screenshot_reference"}:
            status = "confirmed_image_occurrence"
            allow_timeline = True
        elif present == "probable" or node.get("image_variant") or self._boolish(node.get("suspected_tampering")):
            status = "possible_variant"
            allow_timeline = bool(rule_data.get("published_at") or has_image_evidence)
        elif has_image_evidence and self._clamp_float(node.get("similarity")) >= 0.75:
            status = "possible_variant"
            allow_timeline = bool(rule_data.get("published_at"))
        elif present == "not_found":
            status = "rejected_after_page_review"
            allow_timeline = False
        else:
            status = "contextual_only"
            allow_timeline = False

        return {
            "evidence_node_status": status,
            "allow_in_external_timeline": bool(allow_timeline),
            "allow_cross_platform_relation_candidate": bool(allow_timeline and has_source_evidence),
            "reason": (
                f"status={status}; image_evidence={has_image_evidence}; "
                f"source_evidence={has_source_evidence}; crawl_status={crawl_status}"
            ),
        }

    def analyze_tampering(self, node: Dict[str, Any]) -> Dict[str, Any]:
        evidence: List[str] = []
        tamper_types: List[str] = []
        score = 0.0

        validator_reason = str(node.get("reason") or node.get("validation_reason") or "")
        if "疑似篡改候选" in validator_reason:
            if "validator_reason_suspected_tampering" not in tamper_types:
                tamper_types.append("validator_reason_suspected_tampering")
            score += 0.35
            evidence.append(f"validator_reason={validator_reason}")

        if node.get("suspected_tampering"):
            score += 0.45
            evidence.append(str(node.get("tampering_reason") or "validator marked suspected_tampering=true"))

        for signal in self._coerce_list(node.get("tampering_signals")):
            signal_text = str(signal)
            if signal_text and signal_text not in tamper_types:
                tamper_types.append(signal_text)
            score += 0.08

        variant = str(node.get("image_variant") or "")
        variant_markers = {
            "watermark_added_or_changed": ("水印", "watermark"),
            "content_text_changed": ("文字", "文本", "ocr", "text"),
            "edited_variant": ("裁剪", "尺寸", "颜色", "调色", "滤镜", "黑白", "编辑", "variant"),
            "montage_or_recomposition": ("拼接", "重组", "结构"),
            "uncertain_visual_variant": ("视觉相似度不足", "变化", "差异"),
        }
        for tamper_type, markers in variant_markers.items():
            if any(marker.lower() in variant.lower() for marker in markers):
                if tamper_type not in tamper_types:
                    tamper_types.append(tamper_type)
                score += 0.08
        if variant:
            evidence.append(f"image_variant={variant}")

        platform = self._normalize_platform(node.get("platform"), str(node.get("url") or ""))
        watermark_platforms = [str(item).lower() for item in self._coerce_list(node.get("watermark_platforms"))]
        if watermark_platforms and platform not in watermark_platforms:
            tamper_types.append("cross_platform_repost_with_watermark")
            tamper_types.append("watermark_mismatch")
            score += 0.18
            evidence.append(f"watermark_platforms={watermark_platforms} current_platform={platform}")

        overlap = self._clamp_float(node.get("ocr_content_overlap"))
        if 0 < overlap < 0.45:
            if "content_text_changed" not in tamper_types:
                tamper_types.append("content_text_changed")
            score += 0.12
            evidence.append(f"low ocr_content_overlap={overlap}")

        if node.get("watermark_detected"):
            evidence.append(f"watermark_text={node.get('watermark_text') or []}")

        score = min(score, 1.0)
        return {
            "is_tampered": score >= 0.35 or bool(node.get("suspected_tampering")),
            "tamper_score": round(score, 2),
            "tamper_types": list(dict.fromkeys(tamper_types)),
            "evidence": evidence,
            "reason": str(node.get("tampering_reason") or "; ".join(evidence) or "no tampering evidence"),
        }

    def analyze_influence(self, node: Dict[str, Any]) -> Dict[str, Any]:
        record = node.get("normalized_record") if isinstance(node.get("normalized_record"), dict) else {}
        user = record.get("user", {}) if isinstance(record.get("user"), dict) else {}
        verified = bool(user.get("verified") or node.get("verified"))
        verified_reason = str(user.get("verified_reason") or node.get("verified_reason") or "")
        followers = self._to_int(user.get("followers_count") or node.get("followers_count") or node.get("follower_count"))
        engagement = self._engagement_total(node) + self._to_int(node.get("collect_count"))
        evidence: List[str] = []

        follower_score = min(log1p(followers) / 13.0, 1.0)
        engagement_score = min(log1p(engagement) / 11.0, 1.0)
        verified_score = 1.0 if verified or verified_reason else 0.0
        media_score = 1.0 if any(k in verified_reason for k in ("媒体", "新闻", "官方", "机构", "政务")) else 0.0
        influence_score = min(
            follower_score * 0.45 + engagement_score * 0.30 + verified_score * 0.18 + media_score * 0.07,
            1.0,
        )

        if followers:
            evidence.append(f"followers_count={followers}")
        if engagement:
            evidence.append(f"engagement_total={engagement}")
        if verified or verified_reason:
            evidence.append(f"verified={verified} reason={verified_reason}")

        is_big_v = verified or followers >= 100000 or media_score >= 1.0
        is_key = (
            influence_score >= 0.78
            or engagement >= 5000
            or (is_big_v and (followers >= 100000 or engagement >= 500))
        )
        return {
            "is_key_node": bool(is_key),
            "is_big_v": bool(is_big_v),
            "is_matrix_account_candidate": False,
            "influence_score": round(influence_score, 2),
            "evidence": evidence,
        }

    def _extract_publisher(self, metadata: Dict[str, Any], text: str, url: str = "") -> Optional[str]:
        value = self._first_value(
            metadata,
            [
                "author",
                "article:author",
                "byline",
                "dc.creator",
                "dc:creator",
                "creator",
                "publisher",
                "article:publisher",
                "source",
                "site",
                "siteName",
                "site_name",
                "og:site_name",
                "application-name",
                "jsonld_name",
            ],
        )
        if value:
            return value
        return self._extract_publisher_from_text(text)

    @staticmethod
    def needs_metadata_enrichment(data: Dict[str, Any]) -> bool:
        return not data or not data.get("published_at") or not (data.get("publisher") or data.get("author"))

    @staticmethod
    def needs_provenance_analysis(data: Dict[str, Any]) -> bool:
        return not data or not (data.get("source_url") or data.get("source_text") or data.get("public_relation_hint"))

    def needs_metric_enrichment(self, data: Dict[str, Any], page_type: str) -> bool:
        if page_type not in {"social_post", "channel_post", "forum_post"}:
            return False
        count_fields = ("view_count", "repost_count", "comment_count", "like_count", "share_count")
        return not any(self._to_int(data.get(field)) > 0 for field in count_fields)

    def _needs_external_rich_analysis(
        self,
        data: Dict[str, Any],
        page_type: str,
        validator_review_required: bool,
    ) -> bool:
        return (
            self.needs_metadata_enrichment(data)
            or self.needs_provenance_analysis(data)
            or validator_review_required
            or self.needs_metric_enrichment(data, page_type)
        )

    def _needs_external_llm_analysis(
        self,
        data: Dict[str, Any],
        page_type: str,
        validator_review_required: bool,
    ) -> bool:
        if page_type == "dynamic_platform":
            return validator_review_required or self.needs_metadata_enrichment(data)
        return (
            self.needs_metadata_enrichment(data)
            or self.needs_provenance_analysis(data)
            or validator_review_required
        )

    @staticmethod
    def _extract_publisher_from_text(text: str) -> Optional[str]:
        if not text:
            return None
        patterns = [
            r"(?:作者|发布者|来源|来源于|出处|账号|博主|UP主)\s*[:：]\s*([^\s｜|,，。]{2,40})",
            r"(?:By|Author|Source|Publisher)\s*[:：]\s*([A-Za-z0-9_\- .]{2,60})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if not match:
                continue
            value = match.group(1).strip(" -_｜|,，。:：")
            if value and not re.search(r"^\d{2,4}[-/.年月]", value):
                return value[:60]
        return None

    def extract_open_web_evidence(self, crawl_data: Dict[str, Any]) -> Dict[str, Any]:
        metadata = crawl_data.get("metadata") if isinstance(crawl_data.get("metadata"), dict) else {}
        html = str(crawl_data.get("html") or crawl_data.get("rawHtml") or crawl_data.get("raw_html") or "")
        markdown = str(crawl_data.get("markdown") or crawl_data.get("content") or "")
        final_url = str(crawl_data.get("final_url") or "")
        evidence: Dict[str, Any] = {
            "title": (
                metadata.get("title")
                or metadata.get("og:title")
                or metadata.get("og_title")
                or metadata.get("twitter:title")
                or metadata.get("twitter_title")
            ),
            "description": (
                metadata.get("description")
                or metadata.get("og:description")
                or metadata.get("og_description")
                or metadata.get("twitter:description")
                or metadata.get("twitter_description")
            ),
            "canonical_url": metadata.get("canonical_url") or metadata.get("og:url") or final_url,
            "image_urls": [],
            "image_caption": "",
            "image_credit": "",
            "source_url": "",
            "source_text": "",
            "source_url_candidates": [],
            "source_text_candidates": [],
            "language_hint": metadata.get("language") or metadata.get("lang") or "",
        }

        image_candidates = self._coerce_list(
            metadata.get("og:image")
            or metadata.get("ogImage")
            or metadata.get("twitter:image")
            or metadata.get("image")
        )
        if BeautifulSoup is not None and html:
            soup = BeautifulSoup(html, "html.parser")
            canonical = soup.find("link", attrs={"rel": lambda value: value and "canonical" in self._coerce_list(value)})
            if canonical and canonical.get("href"):
                evidence["canonical_url"] = str(canonical.get("href"))
            for meta in soup.find_all("meta"):
                key = str(meta.get("property") or meta.get("name") or "").strip().lower()
                content = str(meta.get("content") or "").strip()
                if not content:
                    continue
                if key in {"og:url", "twitter:url"} and not evidence.get("canonical_url"):
                    evidence["canonical_url"] = content
                elif key in {"title", "og:title", "twitter:title"} and not evidence.get("title"):
                    evidence["title"] = content
                elif key in {"description", "og:description", "twitter:description"} and not evidence.get("description"):
                    evidence["description"] = content
                elif key in {"og:image", "twitter:image", "image"}:
                    image_candidates.append(content)
                elif key in {"author", "article:author", "publisher"}:
                    evidence.setdefault("author", content)
                    evidence.setdefault("publisher", content)
            if not evidence.get("title"):
                heading = soup.find("h1")
                if heading:
                    evidence["title"] = heading.get_text(" ", strip=True)[:200]
            for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
                raw_json = script.string or script.get_text(" ", strip=True)
                for item in self._iter_jsonld_objects(raw_json):
                    item_type = item.get("@type")
                    type_values = {str(value).lower() for value in self._coerce_list(item_type)}
                    if not type_values.intersection({"article", "newsarticle", "blogposting", "discussionforumposting"}):
                        continue
                    if item.get("datePublished") and not evidence.get("published_at"):
                        evidence["published_at"] = self.normalize_time(item.get("datePublished"))
                    if item.get("dateModified") and not evidence.get("modified_at"):
                        evidence["modified_at"] = self.normalize_time(item.get("dateModified"))
                    if not evidence.get("title"):
                        evidence["title"] = item.get("headline") or item.get("name")
                    if not evidence.get("description"):
                        evidence["description"] = item.get("description")
                    author = item.get("author")
                    if isinstance(author, dict):
                        author = author.get("name")
                    if author:
                        evidence.setdefault("author", str(author))
                    publisher = item.get("publisher")
                    if isinstance(publisher, dict):
                        publisher = publisher.get("name")
                    if publisher:
                        evidence.setdefault("publisher", str(publisher))
                    image_candidates.extend(self._extract_image_urls(item.get("image")))
            for figure in soup.find_all("figure")[:8]:
                caption = figure.find("figcaption")
                caption_text = caption.get_text(" ", strip=True) if caption else ""
                if caption_text and not evidence.get("image_caption"):
                    evidence["image_caption"] = caption_text[:300]
                img = figure.find("img")
                if img and img.get("src"):
                    image_candidates.append(str(img.get("src")))
            for img in soup.find_all("img")[:40]:
                if img.get("src"):
                    image_candidates.append(str(img.get("src")))
                alt = str(img.get("alt") or "").strip()
                if alt and not evidence.get("image_caption"):
                    evidence["image_caption"] = alt[:300]
            for link in soup.find_all("a")[:120]:
                text = link.get_text(" ", strip=True)
                href = str(link.get("href") or "").strip()
                text_haystack = text.lower()
                parsed_href = urlparse(href)
                href_haystack = f"{parsed_href.netloc} {parsed_href.path}".lower()
                relation_markers = ("source", "via", "original", "credit", "出处", "来源", "原文", "转载")
                if href and (
                    any(marker in text_haystack for marker in relation_markers)
                    or any(marker in href_haystack for marker in ("source", "via", "original", "credit"))
                ):
                    evidence["source_url_candidates"].append(href)
                    if text:
                        evidence["source_text_candidates"].append(text)

        combined_text = f"{markdown}\n{html}"
        relation_hint = self._extract_relation_hint(combined_text)
        if relation_hint:
            evidence["source_text_candidates"].append(relation_hint)
        if not evidence.get("image_credit"):
            credit_match = re.search(
                r"(?:source|via|credit|original|from)\s*[:：]\s*([^\n\r<]{2,160})",
                combined_text,
                flags=re.I,
            )
            if credit_match:
                evidence["image_credit"] = credit_match.group(1).strip()

        image_urls = []
        for image_url in self._extract_image_urls(image_candidates):
            if image_url not in image_urls:
                image_urls.append(image_url)
        source_urls = [str(item) for item in self._coerce_list(evidence.get("source_url_candidates")) if str(item)]
        source_texts = [str(item) for item in self._coerce_list(evidence.get("source_text_candidates")) if str(item)]
        evidence["image_urls"] = image_urls[:20]
        evidence["source_url_candidates"] = source_urls[:10]
        evidence["source_text_candidates"] = source_texts[:10]
        evidence["source_url"] = source_urls[0] if source_urls else ""
        evidence["source_text"] = source_texts[0] if source_texts else ""
        evidence["canonical_url"] = self._canonicalize_url(str(evidence.get("canonical_url") or final_url))
        return {key: value for key, value in evidence.items() if value not in (None, "", [], {})}

    def _iter_jsonld_objects(self, raw_json: str) -> List[Dict[str, Any]]:
        if not raw_json:
            return []
        try:
            parsed = json.loads(raw_json)
        except (TypeError, ValueError):
            return []
        pending = self._coerce_list(parsed)
        objects: List[Dict[str, Any]] = []
        while pending:
            current = pending.pop(0)
            if isinstance(current, dict):
                objects.append(current)
                graph = current.get("@graph")
                if graph:
                    pending.extend(self._coerce_list(graph))
            elif isinstance(current, list):
                pending.extend(current)
        return objects

    def extract_by_rules(self, crawl_data: Dict[str, Any]) -> Dict[str, Any]:
        metadata = crawl_data.get("metadata") if isinstance(crawl_data.get("metadata"), dict) else {}
        markdown = str(crawl_data.get("markdown") or crawl_data.get("content") or "")
        html = str(crawl_data.get("html") or crawl_data.get("rawHtml") or crawl_data.get("raw_html") or "")
        metadata_text = json.dumps(metadata, ensure_ascii=False)
        text = f"{metadata_text}\n{markdown}\n{html}"
        open_web_evidence = self.extract_open_web_evidence(crawl_data)

        published_at, date_source, time_evidence = self._extract_time_with_evidence(
            metadata=metadata,
            text=text,
            url=str(crawl_data.get("final_url") or ""),
        )
        if not published_at and open_web_evidence.get("published_at"):
            published_at = self.normalize_time(open_web_evidence.get("published_at"))
            date_source = "json_ld"
            time_evidence["page_metadata"] = published_at or ""
        return {
            "title": open_web_evidence.get("title") or metadata.get("title"),
            "description": open_web_evidence.get("description") or metadata.get("description"),
            "published_at": published_at,
            "date_source": date_source,
            "time_evidence": time_evidence,
            "modified_at": open_web_evidence.get("modified_at"),
            "canonical_url": open_web_evidence.get("canonical_url"),
            "image_urls": open_web_evidence.get("image_urls") or [],
            "image_caption": open_web_evidence.get("image_caption"),
            "image_credit": open_web_evidence.get("image_credit"),
            "source_url": open_web_evidence.get("source_url"),
            "source_text": open_web_evidence.get("source_text"),
            "source_url_candidates": open_web_evidence.get("source_url_candidates") or [],
            "source_text_candidates": open_web_evidence.get("source_text_candidates") or [],
            "language_hint": open_web_evidence.get("language_hint"),
            "publisher": open_web_evidence.get("publisher")
            or self._extract_publisher(metadata, text, str(crawl_data.get("final_url") or "")),
            "author": open_web_evidence.get("author"),
            "view_count": self._extract_count(
                text,
                [
                    "浏览",
                    "阅读",
                    "观看",
                    "播放",
                    "views",
                    "view",
                    "viewCount",
                    "view_count",
                    "readCount",
                    "read_count",
                    "playCount",
                    "play_count",
                ],
            ),
            "repost_count": self._extract_count(
                text,
                [
                    "转发",
                    "分享",
                    "reposts",
                    "repost",
                    "repostCount",
                    "repost_count",
                    "shares",
                    "share",
                    "shareCount",
                    "share_count",
                    "forwardCount",
                    "forward_count",
                ],
            ),
            "comment_count": self._extract_count(
                text,
                [
                    "评论",
                    "comments",
                    "comment",
                    "commentCount",
                    "comment_count",
                    "comments_count",
                    "replyCount",
                    "reply_count",
                ],
            ),
            "like_count": self._extract_count(
                text,
                [
                    "点赞",
                    "赞",
                    "喜欢",
                    "likes",
                    "like",
                    "likeCount",
                    "like_count",
                    "diggCount",
                    "digg_count",
                    "favoriteCount",
                    "favorite_count",
                    "collectCount",
                    "collect_count",
                ],
            ),
            "follower_count": self._extract_count(
                text,
                ["粉丝", "followers", "follower", "followerCount", "follower_count", "fansCount", "fans_count"],
            ),
            "following_count": self._extract_count(text, ["关注", "following"]),
            "public_relation_hint": open_web_evidence.get("source_text") or self._extract_relation_hint(text),
            "location_hint": None,
            "propagation_role": self._infer_role_from_text(text),
            "confidence": 0.55 if published_at else 0.25,
            "reason": "规则解析网页 metadata/正文得到部分时空线索。",
        }

    def normalize_time(self, value: Any) -> Optional[str]:
        if value in (None, "", "null"):
            return None
        if isinstance(value, datetime):
            return self._format_publish_time(value)

        raw_value = str(value).strip().strip('"\'`[]{}()')
        raw_value = re.sub(
            r"^(?:published?|publish(?:ed)?_?time|pub(?:lish)?date|created?|updated?|date|time)\s*[:=：]\s*",
            "",
            raw_value,
            flags=re.I,
        )
        try:
            parsed_http = parsedate_to_datetime(raw_value)
            if parsed_http:
                return self._format_publish_time(parsed_http.replace(tzinfo=None))
        except (TypeError, ValueError):
            pass

        numeric_match = re.fullmatch(r"\d{10}|\d{13}", raw_value)
        if numeric_match:
            try:
                timestamp = int(raw_value)
                if len(raw_value) == 13:
                    timestamp = timestamp // 1000
                if 946684800 <= timestamp <= 4102444800:
                    return self._format_publish_time(datetime.fromtimestamp(timestamp))
            except (OverflowError, OSError, ValueError):
                return None

        compact_date = re.fullmatch(r"(20\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])", raw_value)
        if compact_date:
            try:
                return self._format_publish_time(
                    datetime(
                        int(compact_date.group(1)),
                        int(compact_date.group(2)),
                        int(compact_date.group(3)),
                    )
                )
            except ValueError:
                return None

        short_cn_year = re.match(
            r"^(\d{2})年(\d{1,2})月(\d{1,2})日(?:\s*(\d{1,2}):(\d{2})(?::(\d{2}))?)?$",
            raw_value,
        )
        if short_cn_year:
            year = 2000 + int(short_cn_year.group(1))
            month = int(short_cn_year.group(2))
            day = int(short_cn_year.group(3))
            hour = int(short_cn_year.group(4) or 0)
            minute = int(short_cn_year.group(5) or 0)
            second = int(short_cn_year.group(6) or 0)
            try:
                return self._format_publish_time(datetime(year, month, day, hour, minute, second))
            except ValueError:
                return None
        raw_value = (
            raw_value.replace("年", "-")
            .replace("月", "-")
            .replace("日", " ")
            .replace(".", "-")
            .replace("T", " ")
            .replace("Z", "")
        )
        raw_value = re.sub(r"([+-]\d{2}:?\d{2})$", "", raw_value).strip()
        raw_value = re.sub(r"\s+", " ", raw_value)

        known_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
            "%Y.%m.%d %H:%M:%S",
            "%Y.%m.%d %H:%M",
            "%Y.%m.%d",
        ]
        for fmt in known_formats:
            try:
                parsed = datetime.strptime(raw_value, fmt)
                return self._format_publish_time(parsed)
            except ValueError:
                continue

        if date_parser is not None:
            try:
                parsed = date_parser.parse(raw_value)
                if parsed:
                    return self._format_publish_time(parsed.replace(tzinfo=None))
            except (ValueError, OverflowError, TypeError):
                return None
        return None

    @staticmethod
    def _format_publish_time(parsed: datetime) -> Optional[str]:
        max_year = datetime.now().year + MAX_FUTURE_PUBLISH_YEARS
        if parsed.year < MIN_PUBLISH_YEAR or parsed.year > max_year:
            return None
        return parsed.strftime("%Y-%m-%d %H:%M:%S")

    def calculate_node_weight(self, node: Dict[str, Any], index: int, total: int) -> float:
        """Calculate propagation influence, not source likelihood.

        Time is intentionally low-weight here. Source detection is handled by
        source_score/is_suspected_source so a late viral repost can still be a
        key propagation node.
        """
        time_score = 1.0 - (index / max(total - 1, 1))
        engagement_raw = (
            self._to_int(node.get("view_count")) * 0.04
            + self._to_int(node.get("repost_count")) * 2.0
            + self._to_int(node.get("comment_count")) * 1.0
            + self._to_int(node.get("like_count")) * 0.35
            + self._to_int(node.get("collect_count")) * 0.45
        )
        engagement_score = min(log1p(engagement_raw) / 14.5, 1.0)
        publisher_raw = self._to_int(node.get("follower_count")) * 0.7 + self._to_int(
            node.get("following_count")
        ) * 0.05
        publisher_score = min(log1p(publisher_raw) / 15.0, 1.0)
        relation_score = 1.0 if node.get("public_relation_hint") else 0.0
        confidence = self._clamp_float(node.get("llm_confidence"))

        weight = (
            time_score * self.config.node_weight_time_ratio
            + engagement_score * self.config.node_weight_engagement_ratio
            + publisher_score * self.config.node_weight_publisher_ratio
            + relation_score * self.config.node_weight_relation_ratio
            + confidence * self.config.node_weight_confidence_ratio
        )
        return round(min(max(weight, 0.0), 1.0), 2)

    def calculate_source_score(self, node: Dict[str, Any], index: int, total: int) -> float:
        """Calculate likelihood of being the source separately from influence."""
        if not node.get("published_at"):
            return 0.0

        time_order_score = 1.0 - (index / max(total - 1, 1))
        time_reliability_score = self._time_reliability(node)
        original_claim_score = self._original_claim_score(node)
        watermark_origin_match_score = self._watermark_origin_match_score(node)
        publisher_credibility_score = self._clamp_float(
            (node.get("influence_analysis") or {}).get("influence_score")
        )
        repost_penalty = self._repost_penalty(node)
        watermark_mismatch_penalty = self._watermark_mismatch_penalty(node)
        domain_penalty = self._source_domain_penalty(node)

        score = (
            time_order_score * 0.30
            + time_reliability_score * 0.20
            + original_claim_score * 0.20
            + watermark_origin_match_score * 0.15
            + publisher_credibility_score * 0.10
            - repost_penalty * 0.20
            - watermark_mismatch_penalty * 0.15
            - domain_penalty
        )
        return round(min(max(score, 0.0), 1.0), 2)

    def _original_claim_score(self, node: Dict[str, Any]) -> float:
        record = node.get("normalized_record") if isinstance(node.get("normalized_record"), dict) else {}
        post = record.get("post", {}) if isinstance(record.get("post"), dict) else {}
        relations = record.get("relations", {}) if isinstance(record.get("relations"), dict) else {}
        text = " ".join(
            str(value or "")
            for value in (
                node.get("title"),
                node.get("description"),
                node.get("public_relation_hint"),
                post.get("text"),
                post.get("source_text"),
                post.get("source_url"),
                relations.get("source_candidates"),
            )
        ).lower()
        if any(marker in text for marker in ("原创", "首发", "作者发布", "original", "first publish", "source")):
            return 1.0
        if any(marker in text for marker in ("来源", "转载", "转发", "引用", "搬运", "via", "from")):
            return 0.15
        return 0.45

    def _watermark_origin_match_score(self, node: Dict[str, Any]) -> float:
        platform = self._normalize_platform(node.get("platform"), str(node.get("url") or ""))
        publisher = str(node.get("publisher") or node.get("author") or "").lower()
        watermark_platforms = [
            self._normalize_platform(item)
            for item in self._coerce_list(node.get("watermark_platforms"))
        ]
        if watermark_platforms and platform not in watermark_platforms:
            return 0.0

        account_values: List[str] = []
        accounts = node.get("watermark_accounts")
        if isinstance(accounts, dict):
            for value in accounts.values():
                account_values.extend(str(item).lower() for item in self._coerce_list(value))
        else:
            account_values.extend(str(item).lower() for item in self._coerce_list(accounts))
        if account_values:
            if publisher and any(publisher in account or account in publisher for account in account_values):
                return 1.0
            return 0.25
        if watermark_platforms:
            return 0.7
        return 0.45

    def _repost_penalty(self, node: Dict[str, Any]) -> float:
        record = node.get("normalized_record") if isinstance(node.get("normalized_record"), dict) else {}
        post = record.get("post", {}) if isinstance(record.get("post"), dict) else {}
        if post.get("is_repost") or post.get("original_post_id"):
            return 1.0
        hint = " ".join(
            str(node.get(key) or "")
            for key in ("public_relation_hint", "title", "description", "propagation_role")
        ).lower()
        if any(marker in hint for marker in ("转载", "转发", "引用", "搬运", "via", "from", "repost")):
            return 0.8
        return 0.0

    def _watermark_mismatch_penalty(self, node: Dict[str, Any]) -> float:
        platform = self._normalize_platform(node.get("platform"), str(node.get("url") or ""))
        watermark_platforms = [
            self._normalize_platform(item)
            for item in self._coerce_list(node.get("watermark_platforms"))
        ]
        if watermark_platforms and platform not in watermark_platforms:
            return 1.0

        publisher = str(node.get("publisher") or node.get("author") or "").lower()
        accounts = node.get("watermark_accounts")
        account_values: List[str] = []
        if isinstance(accounts, dict):
            for value in accounts.values():
                account_values.extend(str(item).lower() for item in self._coerce_list(value))
        else:
            account_values.extend(str(item).lower() for item in self._coerce_list(accounts))
        if account_values and publisher and not any(publisher in account or account in publisher for account in account_values):
            return 0.6
        return 0.0

    @staticmethod
    def _text_similarity(left: Any, right: Any) -> float:
        left_text = re.sub(r"\s+", " ", str(left or "")).strip().lower()
        right_text = re.sub(r"\s+", " ", str(right or "")).strip().lower()
        if not left_text or not right_text:
            return 0.0
        return SequenceMatcher(None, left_text[:2000], right_text[:2000]).ratio()

    def _time_reliability(self, node: Dict[str, Any]) -> float:
        date_source = str(node.get("date_source") or "")
        weights = {
            "page_metadata": 1.0,
            "tikomni": 1.0,
            "time_tag": 0.95,
            "upper_data": 0.95,
            "llm": 0.85,
            "visible_text": 0.75,
            "search_result": 0.65,
            "url_pattern": 0.45,
            "http_last_modified": 0.25,
        }
        return weights.get(date_source, 0.35)

    @staticmethod
    def _source_domain_penalty(node: Dict[str, Any]) -> float:
        domain = str(node.get("domain") or "").lower()
        aggregation_domains = (
            "huaban.com",
            "pinterest.",
            "pinimg.com",
            "baidu.com",
            "bing.com",
            "google.",
        )
        if any(marker in domain for marker in aggregation_domains):
            return 0.18
        return 0.0

    def build_mermaid_graph(self, nodes: List[Dict[str, Any]]) -> str:
        lines = ["graph TD"]
        if not nodes:
            return "\n".join(lines)

        visible_nodes = [node for node in nodes if node.get("is_topology_visible", True)]
        visible_ids = {str(node.get("id")) for node in visible_nodes}

        for node in visible_nodes:
            node_id = self._safe_mermaid_id(node.get("id"))
            label = (
                f"{node.get('published_at') or '未知时间'}<br/>"
                f"{node.get('domain') or '未知域名'}<br/>"
                f"{node.get('propagation_role') or '未知'}<br/>"
                f"weight={node.get('node_weight', 0.0):.2f}"
            )
            lines.append(f'    {node_id}["{self._escape_mermaid_label(label)}"]')

        for node in visible_nodes:
            parent_id = self._nearest_visible_parent_id(nodes, node, visible_ids)
            if not parent_id:
                continue
            current_id = self._safe_mermaid_id(node.get("id"))
            edge_weight = self._clamp_float(node.get("edge_weight_from_parent"))
            lines.append(
                f"    {self._safe_mermaid_id(parent_id)} -->|edge={edge_weight:.2f}| {current_id}"
            )

        for node in visible_nodes:
            node_id = self._safe_mermaid_id(node.get("id"))
            if node.get("is_suspected_source"):
                lines.append(f"    class {node_id} sourceNode")
            elif node.get("is_key_node"):
                lines.append(f"    class {node_id} keyNode")

        lines.append("    classDef sourceNode fill:#fff3b0,stroke:#d9a300,stroke-width:2px")
        lines.append("    classDef keyNode fill:#e0f2fe,stroke:#0284c7,stroke-width:2px")
        return "\n".join(lines)

    def build_topology_data(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        topology_nodes: List[Dict[str, Any]] = []
        topology_edges: List[Dict[str, Any]] = []
        platform_timelines: Dict[str, List[str]] = {}
        external_timelines: Dict[str, List[str]] = {}
        platform_rank_by_id: Dict[str, int] = {}
        grouped_by_platform: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for node in nodes:
            platform = self._normalize_platform(node.get("platform"), str(node.get("url") or ""))
            rank_key = platform if platform in {"weibo", "xiaohongshu"} else f"external:{node.get('platform_family') or 'unknown'}"
            grouped_by_platform[rank_key].append(node)
        for platform_nodes in grouped_by_platform.values():
            ranked = sorted(
                platform_nodes,
                key=lambda item: self._clamp_float(item.get("node_weight")),
                reverse=True,
            )
            for rank, node in enumerate(ranked, start=1):
                platform_rank_by_id[str(node.get("id") or "")] = rank

        for node in nodes:
            node_id = str(node.get("id") or "")
            platform = self._normalize_platform(node.get("platform"), str(node.get("url") or ""))
            platform_family = str(node.get("platform_family") or platform or "unknown")
            page_type = str(node.get("page_type") or node.get("page_type_hint") or "unknown")
            tamper_analysis = node.get("tamper_analysis") or {}
            is_tampered = bool(tamper_analysis.get("is_tampered"))
            badges = []
            if node.get("is_suspected_source"):
                badges.append("源头候选")
            if node.get("is_key_node"):
                badges.append("关键节点")
            if is_tampered:
                badges.append("疑似篡改")
            if node.get("is_big_v"):
                badges.append("大V/认证")
            if node.get("is_matrix_account_candidate"):
                badges.append("矩阵账号候选")
            duplicate_analysis = node.get("duplicate_analysis") if isinstance(node.get("duplicate_analysis"), dict) else {}
            if duplicate_analysis.get("is_possible_duplicate"):
                badges.append("疑似重复")
            node_type = (
                "source"
                if node.get("is_suspected_source")
                else "tampered"
                if is_tampered
                else "key"
                if node.get("is_key_node")
                else "normal"
            )
            if self._is_mainstream_platform(node):
                platform_timelines.setdefault(platform, []).append(node_id)
            elif self._boolish(node.get("allow_in_external_timeline")):
                external_timelines.setdefault(platform_family, []).append(node_id)
            topology_nodes.append(
                {
                    "id": node_id,
                    "url": node.get("url"),
                    "canonical_url": node.get("canonical_url"),
                    "image_url": node.get("image_url") or node.get("thumbnail_url"),
                    "thumbnail_url": node.get("thumbnail_url") or node.get("image_url"),
                    "platform": platform,
                    "platform_family": platform_family,
                    "page_type": page_type,
                    "platform_rank": platform_rank_by_id.get(node_id, 0),
                    "input_order": self._to_int(node.get("input_order")),
                    "label": self._build_node_label(node),
                    "publish_time": node.get("published_at"),
                    "publisher": node.get("publisher") or node.get("author"),
                    "title": node.get("title"),
                    "description": self._truncate_text(node.get("description"), 500),
                    "source_score": self._clamp_float(node.get("source_score")),
                    "node_weight": self._clamp_float(node.get("node_weight")),
                    "node_type": node_type,
                    "badges": badges,
                    "is_suspected_source": bool(node.get("is_suspected_source")),
                    "is_platform_source": bool(node.get("is_platform_source")),
                    "is_key_node": bool(node.get("is_key_node")),
                    "is_big_v": bool(node.get("is_big_v")),
                    "is_tampered": is_tampered,
                    "is_cross_platform_node": bool(
                        node.get("edge_type_from_parent") == "cross_platform_watermark"
                        or "cross_platform_repost_with_watermark"
                        in self._coerce_list(tamper_analysis.get("tamper_types"))
                    ),
                    "is_topology_visible": bool(node.get("is_topology_visible", True)),
                    "is_matrix_account_candidate": bool(node.get("is_matrix_account_candidate")),
                    "possible_duplicate": bool(node.get("possible_duplicate")),
                    "evidence_node_status": node.get("evidence_node_status"),
                    "allow_in_external_timeline": bool(node.get("allow_in_external_timeline")),
                    "allow_cross_platform_relation_candidate": bool(
                        node.get("allow_cross_platform_relation_candidate")
                    ),
                    "image_urls": node.get("image_urls") or [],
                    "source_url": node.get("source_url"),
                    "source_text": node.get("source_text"),
                    "source_platform_hint": node.get("source_platform_hint"),
                    "source_account_hint": node.get("source_account_hint"),
                    "node_decision_reason": node.get("node_decision_reason"),
                    "field_evidence": node.get("field_evidence") or {},
                    "node_decision": node.get("node_decision") or {},
                    "image_occurrence": node.get("image_occurrence") or {},
                    "provenance": node.get("provenance") or {},
                    "duplicate_analysis": duplicate_analysis,
                    "tamper_analysis": tamper_analysis,
                    "influence_analysis": node.get("influence_analysis") or {},
                    "matrix_account_analysis": node.get("matrix_account_analysis") or {},
                    "metrics": {
                        "view_count": self._nullable_count_for_display(node.get("view_count")),
                        "like_count": self._nullable_count_for_display(node.get("like_count")),
                        "comment_count": self._nullable_count_for_display(node.get("comment_count")),
                        "repost_count": self._nullable_count_for_display(node.get("repost_count")),
                        "collect_count": self._nullable_count_for_display(node.get("collect_count")),
                        "share_count": self._nullable_count_for_display(node.get("share_count")),
                    },
                    "validator": {
                        "similarity": node.get("similarity"),
                        "image_variant": node.get("image_variant"),
                        "reason": node.get("reason"),
                        "validation_reason": node.get("validation_reason"),
                        "candidate_content_text": self._truncate_text(node.get("candidate_content_text"), 500),
                        "ocr_relation_signals": self._extract_ocr_relation_signals(str(node.get("candidate_content_text") or "")),
                        "watermark_detected": node.get("watermark_detected"),
                        "watermark_platforms": node.get("watermark_platforms") or [],
                        "watermark_accounts": node.get("watermark_accounts") or {},
                        "ocr_content_overlap": node.get("ocr_content_overlap"),
                    },
                    "normalized_record": node.get("normalized_record") or {},
                }
            )

        topology_edges = self._build_topology_edges(nodes)

        return {
            "nodes": topology_nodes,
            "edges": topology_edges,
            "platform_timelines": platform_timelines,
            "external_timelines": external_timelines,
            "duplicate_clusters": self._build_duplicate_cluster_summary(nodes),
            "cross_platform_relations": self._build_cross_platform_relations(nodes),
            "source_decision": self._build_source_decision(nodes),
            "external_evidence_nodes": [
                str(node.get("id"))
                for node in nodes
                if not self._is_mainstream_platform(node)
                and self._boolish(node.get("allow_in_external_timeline"))
            ],
            "agent_actions": self._build_agent_actions(nodes),
        }

    def _build_topology_edges(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        node_ids = {str(node.get("id") or "") for node in nodes}
        node_by_id = {str(node.get("id") or ""): node for node in nodes}
        edge_by_pair: Dict[tuple[str, str], Dict[str, Any]] = {}

        def allows_topology_edges(node_id: str) -> bool:
            node = node_by_id.get(node_id) or {}
            if self._is_mainstream_platform(node):
                return True
            return self._boolish(node.get("allow_in_external_timeline"))

        def add_edge(edge: Dict[str, Any]) -> None:
            source = str(edge.get("source") or edge.get("from") or "")
            target = str(edge.get("target") or edge.get("to") or "")
            edge_type = str(edge.get("edge_type") or "inferred")
            if not source or not target or source == target:
                return
            if source not in node_ids or target not in node_ids:
                return
            if edge_type != "duplicate_cluster" and (
                not allows_topology_edges(source) or not allows_topology_edges(target)
            ):
                return
            confidence = self._clamp_float(edge.get("confidence", edge.get("edge_weight")))
            if confidence <= 0:
                return
            normalized = {
                "source": source,
                "target": target,
                "from": source,
                "to": target,
                "edge_type": edge_type,
                "edge_weight": round(confidence, 2),
                "confidence": round(confidence, 2),
                "method": str(edge.get("method") or edge.get("edge_type") or "inferred"),
                "evidence": [str(item) for item in self._coerce_list(edge.get("evidence")) if str(item).strip()][:6],
                "score_components": edge.get("score_components") or {},
            }
            key = (source, target)
            existing = edge_by_pair.get(key)
            if not existing or self._clamp_float(existing.get("confidence")) < confidence:
                edge_by_pair[key] = normalized

        for node in nodes:
            if not node.get("parent_id"):
                continue
            add_edge(
                {
                    "source": str(node.get("parent_id")),
                    "target": str(node.get("id") or ""),
                    "edge_type": node.get("edge_type_from_parent") or "inferred",
                    "edge_weight": self._clamp_float(node.get("edge_weight_from_parent")),
                    "confidence": self._clamp_float(node.get("edge_weight_from_parent")),
                    "method": node.get("edge_method_from_parent") or node.get("edge_type_from_parent") or "inferred",
                    "evidence": node.get("edge_evidence_from_parent") or [],
                    "score_components": node.get("edge_score_components_from_parent") or {},
                }
            )

        for edge in self._build_duplicate_cluster_edges(nodes):
            add_edge(edge)

        edges = list(edge_by_pair.values())
        edges.sort(key=lambda item: self._clamp_float(item.get("confidence")), reverse=True)
        return edges

    def _assign_duplicate_clusters(self, nodes: List[Dict[str, Any]]) -> None:
        ordered = sorted(nodes, key=lambda item: self._to_int(item.get("input_order")))
        base_node: Optional[Dict[str, Any]] = None
        clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        explicit_clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for node in ordered:
            explicit_cluster_id = str(node.get("duplicate_cluster_id") or "").strip()
            if explicit_cluster_id:
                explicit_clusters[explicit_cluster_id].append(node)
                continue
            is_duplicate = self._boolish(node.get("possible_duplicate"))
            if not is_duplicate or base_node is None:
                base_node = node
                if not is_duplicate:
                    clusters[str(node.get("id") or "")].append(node)
                    continue
            base_id = str(base_node.get("id") or node.get("id") or "")
            clusters[base_id].append(node)

        for node in nodes:
            node["duplicate_analysis"] = {
                "is_possible_duplicate": self._boolish(node.get("possible_duplicate")),
                "base_node_id": "",
                "cluster_id": "",
                "center_node_id": "",
                "cluster_role": "single",
                "cluster_member_ids": [],
                "explanation": "not marked as possible_duplicate by validator",
            }

        for base_id, members in clusters.items():
            unique_members = []
            seen: set[str] = set()
            for member in members:
                member_id = str(member.get("id") or "")
                if not member_id or member_id in seen:
                    continue
                seen.add(member_id)
                unique_members.append(member)
            if not unique_members:
                continue
            center = self._select_duplicate_cluster_center(unique_members)
            center_id = str(center.get("id") or "")
            member_ids = [str(member.get("id") or "") for member in unique_members]
            cluster_id = f"dup_{base_id}"
            for member in unique_members:
                member_id = str(member.get("id") or "")
                is_duplicate = self._boolish(member.get("possible_duplicate"))
                role = "center" if member_id == center_id else ("duplicate" if is_duplicate else "base")
                member["duplicate_analysis"] = {
                    "is_possible_duplicate": is_duplicate,
                    "base_node_id": base_id,
                    "cluster_id": cluster_id,
                    "center_node_id": center_id,
                    "cluster_role": role,
                    "cluster_member_ids": member_ids,
                    "explanation": (
                        "possible_duplicate=true nodes are attached to the nearest previous possible_duplicate=false baseline; "
                        "cluster center is the highest influence node, or the earliest node when all influence weights are low."
                    ),
                }

        lookup: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            for key in (node.get("id"), node.get("upstream_id"), node.get("duplicate_anchor_id")):
                if key not in (None, ""):
                    lookup[str(key)] = node

        for explicit_cluster_id, members in explicit_clusters.items():
            unique_members = []
            seen: set[str] = set()
            for member in members:
                member_id = str(member.get("id") or "")
                if not member_id or member_id in seen:
                    continue
                seen.add(member_id)
                unique_members.append(member)
            if not unique_members:
                continue
            anchor_hint = str(unique_members[0].get("duplicate_anchor_id") or "")
            anchor = lookup.get(anchor_hint) if anchor_hint else None
            if anchor not in unique_members:
                anchor = None
            center = anchor or self._select_duplicate_cluster_center(unique_members)
            center_id = str(center.get("id") or "")
            base_id = str((anchor or unique_members[0]).get("id") or "")
            member_ids = [str(member.get("id") or "") for member in unique_members]
            for member in unique_members:
                member_id = str(member.get("id") or "")
                is_duplicate = self._boolish(member.get("possible_duplicate")) or member_id != base_id
                role = "center" if member_id == center_id else ("base" if member_id == base_id else "duplicate")
                member["duplicate_analysis"] = {
                    "is_possible_duplicate": is_duplicate,
                    "base_node_id": base_id,
                    "cluster_id": explicit_cluster_id,
                    "center_node_id": center_id,
                    "cluster_role": role,
                    "cluster_member_ids": member_ids,
                    "explanation": (
                        "validator duplicate_cluster_id is used explicitly; input order is preserved and no nodes are merged."
                    ),
                }

        self._mark_topology_visibility(nodes)

    def _select_duplicate_cluster_center(self, members: List[Dict[str, Any]]) -> Dict[str, Any]:
        weighted = max(members, key=lambda item: self._clamp_float(item.get("node_weight")))
        if self._clamp_float(weighted.get("node_weight")) >= self.config.topology_visibility_threshold:
            return weighted
        timed = [member for member in members if member.get("published_at")]
        if timed:
            return min(timed, key=self._source_selection_key)
        return min(members, key=lambda item: self._to_int(item.get("input_order")))

    def _build_duplicate_cluster_edges(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        node_by_id = {str(node.get("id") or ""): node for node in nodes}
        edges: List[Dict[str, Any]] = []
        emitted: set[tuple[str, str]] = set()
        for node in nodes:
            analysis = node.get("duplicate_analysis") if isinstance(node.get("duplicate_analysis"), dict) else {}
            center_id = str(analysis.get("center_node_id") or "")
            node_id = str(node.get("id") or "")
            member_ids = [
                str(item)
                for item in self._coerce_list(analysis.get("cluster_member_ids"))
                if str(item)
            ]
            if len(member_ids) <= 1:
                continue
            if not center_id or center_id == node_id:
                continue
            center = node_by_id.get(center_id)
            if not center:
                continue
            key = (center_id, node_id)
            if key in emitted:
                continue
            emitted.add(key)
            text_similarity = self._text_similarity(
                self._node_text_for_similarity(center),
                self._node_text_for_similarity(node),
            )
            image_similarity = max(
                self._clamp_float(center.get("similarity")),
                self._clamp_float(node.get("similarity")),
            )
            confidence = min(0.9, 0.55 + image_similarity * 0.25 + text_similarity * 0.12)
            edges.append(
                {
                    "source": center_id,
                    "target": node_id,
                    "from": center_id,
                    "to": node_id,
                    "edge_type": "duplicate_cluster",
                    "edge_weight": round(confidence, 2),
                    "confidence": round(confidence, 2),
                    "method": "possible_duplicate+image+text",
                    "evidence": [
                        f"possible_duplicate baseline={analysis.get('base_node_id')}",
                        f"cluster_center={center_id}",
                        f"image_similarity={image_similarity:.2f}",
                        f"text_similarity={text_similarity:.2f}",
                    ],
                    "score_components": {
                        "image_similarity": round(image_similarity, 3),
                        "text_similarity": round(text_similarity, 3),
                    },
                }
            )
        return edges

    def _build_duplicate_cluster_summary(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        clusters: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            analysis = node.get("duplicate_analysis") if isinstance(node.get("duplicate_analysis"), dict) else {}
            cluster_id = str(analysis.get("cluster_id") or "")
            member_ids = [str(item) for item in self._coerce_list(analysis.get("cluster_member_ids")) if str(item)]
            if not cluster_id or len(member_ids) <= 1:
                continue
            clusters[cluster_id] = {
                "cluster_id": cluster_id,
                "base_node_id": str(analysis.get("base_node_id") or ""),
                "center_node_id": str(analysis.get("center_node_id") or ""),
                "member_node_ids": member_ids,
                "size": len(member_ids),
                "explanation": analysis.get("explanation") or "",
            }
        return list(clusters.values())

    def _build_agent_actions(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for node in nodes:
            platform = self._normalize_platform(node.get("platform"), str(node.get("url") or ""))
            if platform != "other":
                continue
            crawl_status = str(node.get("crawl_status") or node.get("firecrawl_status") or "")
            has_ocr = bool(node.get("candidate_content_text"))
            reason_parts = []
            if crawl_status and crawl_status not in {"success", "skipped"}:
                reason_parts.append(f"crawler status={crawl_status}")
            if has_ocr:
                reason_parts.append("OCR text can be used as peripheral evidence")
            if not reason_parts:
                reason_parts.append("non-mainstream platform requires stronger page evidence before graph linking")
            actions.append({
                "node_id": str(node.get("id") or ""),
                "platform": platform,
                "action": "llmscrapy_pipeline + screenshot_ocr",
                "priority": "medium" if has_ocr else "high",
                "reason": "; ".join(reason_parts),
            })
        return actions

    def _build_source_decision(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        source_nodes = [node for node in nodes if node.get("is_suspected_source") and self._is_mainstream_platform(node)]
        platform_sources = {
            self._normalize_platform(node.get("platform"), str(node.get("url") or "")): str(node.get("id"))
            for node in source_nodes
        }
        if len(source_nodes) == 1:
            reasoning = "single mainstream source selected"
            if source_nodes[0].get("llm_topology_status") == "success" and source_nodes[0].get("llm_topology_reason"):
                reasoning = str(source_nodes[0].get("llm_topology_reason"))
            return {
                "global_source_mode": "single_source",
                "global_source_node_id": str(source_nodes[0].get("id")),
                "platform_sources": platform_sources,
                "confidence": self._clamp_float(source_nodes[0].get("source_score")),
                "reasoning": [reasoning],
            }
        if source_nodes:
            return {
                "global_source_mode": "per_platform_sources",
                "global_source_node_id": "",
                "platform_sources": platform_sources,
                "confidence": max(self._clamp_float(node.get("source_score")) for node in source_nodes),
                "reasoning": ["no explicit cross-platform source relation; source selected per mainstream platform"],
            }
        return {
            "global_source_mode": "unknown",
            "global_source_node_id": "",
            "platform_sources": {},
            "confidence": 0.0,
            "reasoning": ["no eligible mainstream source node"],
        }

    def build_analysis_summary(
        self,
        nodes: List[Dict[str, Any]],
        topology_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        source = next((node for node in nodes if node.get("is_suspected_source")), None)
        if source is None and nodes:
            source = max(nodes, key=lambda item: self._clamp_float(item.get("source_score")))

        tampered_nodes = [
            str(node.get("id"))
            for node in nodes
            if (node.get("tamper_analysis") or {}).get("is_tampered")
        ]
        risk_summary = []
        for node in nodes:
            tamper = node.get("tamper_analysis") or {}
            if tamper.get("is_tampered"):
                risk_summary.append(
                    {
                        "node_id": str(node.get("id")),
                        "risk": "possible_image_tampering_or_context_shift",
                        "reason": tamper.get("reason") or "",
                    }
                )
            if node.get("edge_type_from_parent") == "cross_platform_watermark":
                risk_summary.append(
                    {
                        "node_id": str(node.get("id")),
                        "risk": "cross_platform_watermark_relation",
                        "reason": "; ".join(node.get("edge_evidence_from_parent") or []),
                    }
                )

        return {
            "suspected_source_node_id": str(source.get("id")) if source else "",
            "global_source_mode": topology_data.get("source_decision", {}).get("global_source_mode", "single_source" if source else "unknown"),
            "global_source_node_id": topology_data.get("source_decision", {}).get("global_source_node_id", str(source.get("id")) if source else ""),
            "platform_sources": topology_data.get("source_decision", {}).get("platform_sources", self._fallback_platform_sources(nodes)),
            "source_reasoning": topology_data.get("source_decision", {}).get("reasoning", []),
            "agent_actions": topology_data.get("agent_actions", []),
            "key_node_ids": [str(node.get("id")) for node in nodes if node.get("is_key_node")],
            "tampered_node_ids": tampered_nodes,
            "matrix_account_node_ids": [
                str(node.get("id")) for node in nodes if node.get("is_matrix_account_candidate")
            ],
            "platform_timelines": topology_data.get("platform_timelines", {}),
            "cross_platform_relation_summary": topology_data.get("cross_platform_relations", []),
            "risk_summary": risk_summary,
        }

    def _run_llm_topology_synthesis(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.config.enable_llm_relation_analysis:
            return {"status": "skipped", "reason": "disabled by ANALYZER_ENABLE_LLM_RELATION_ANALYSIS"}
        if not self.llm_client.enabled:
            return {"status": "skipped", "reason": self.llm_client.reason}
        candidates = self._select_nodes_for_llm_relation(nodes)
        if not candidates:
            return {"status": "skipped", "reason": "no nodes selected for relation synthesis"}
        payload = self._build_relation_evidence_payload(candidates)
        result = self.llm_client.synthesize_topology(payload)
        if not isinstance(result, dict):
            return {"status": "failed", "reason": "LLM returned no topology synthesis"}
        result.setdefault("status", "success" if not result.get("error") else "failed")
        result.setdefault("evidence_payload_summary", {
            "node_count": len(candidates),
            "mainstream_node_count": sum(1 for node in candidates if self._is_mainstream_platform(node)),
            "external_node_count": sum(1 for node in candidates if not self._is_mainstream_platform(node)),
        })
        return result

    def _select_nodes_for_llm_relation(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        mainstream = [node for node in nodes if self._is_mainstream_platform(node)]
        external = [
            node for node in nodes
            if not self._is_mainstream_platform(node)
            and (
                node.get("candidate_content_text")
                or node.get("watermark_text")
                or node.get("public_relation_hint")
                or node.get("title")
            )
        ]
        selected = mainstream + external
        limit = max(self.config.max_llm_relation_nodes, 0)
        if limit:
            selected = selected[:limit]
        return selected

    def _build_relation_evidence_payload(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        compact_nodes: List[Dict[str, Any]] = []
        for node in nodes:
            record = node.get("normalized_record") if isinstance(node.get("normalized_record"), dict) else {}
            post = record.get("post", {}) if isinstance(record.get("post"), dict) else {}
            user = record.get("user", {}) if isinstance(record.get("user"), dict) else {}
            relations = record.get("relations", {}) if isinstance(record.get("relations"), dict) else {}
            provider_status = record.get("provider_status", {}) if isinstance(record.get("provider_status"), dict) else {}
            validation_signals = node.get("validation_signals") if isinstance(node.get("validation_signals"), dict) else {}
            ocr_text = (
                node.get("candidate_content_text")
                or validation_signals.get("candidate_content_text")
                or validation_signals.get("candidate_ocr_text")
                or ""
            )
            ocr_relation_signals = self._extract_ocr_relation_signals(str(ocr_text or ""))
            compact_nodes.append(
                {
                    "id": str(node.get("id") or ""),
                    "platform": self._normalize_platform(node.get("platform"), str(node.get("url") or "")),
                    "url": node.get("url"),
                    "publish_time": node.get("published_at"),
                    "publisher": node.get("publisher") or user.get("username") or user.get("nickname"),
                    "user_id": user.get("user_id"),
                    "verified": user.get("verified"),
                    "followers_count": user.get("followers_count"),
                    "title": node.get("title") or post.get("title"),
                    "post_text": self._truncate_text(post.get("text") or node.get("description"), 1000),
                    "ocr_content_text": self._truncate_text(ocr_text, 1200),
                    "ocr_content_overlap": node.get("ocr_content_overlap"),
                    "ocr_relation_signals": ocr_relation_signals,
                    "watermark_detected": node.get("watermark_detected"),
                    "watermark_platforms": node.get("watermark_platforms") or [],
                    "watermark_accounts": node.get("watermark_accounts") or {},
                    "watermark_text": node.get("watermark_text") or [],
                    "image_variant": node.get("image_variant"),
                    "suspected_tampering": node.get("suspected_tampering"),
                    "tampering_signals": node.get("tampering_signals") or [],
                    "tampering_reason": node.get("tampering_reason"),
                    "source_url": post.get("source_url"),
                    "source_text": post.get("source_text"),
                    "original_post_id": post.get("original_post_id"),
                    "is_repost": post.get("is_repost"),
                    "mentioned_accounts": relations.get("mentioned_accounts") or [],
                    "linked_urls": relations.get("linked_urls") or [],
                    "source_candidates": relations.get("source_candidates") or [],
                    "metrics": {
                        "like_count": node.get("like_count"),
                        "comment_count": node.get("comment_count"),
                        "repost_count": node.get("repost_count"),
                        "view_count": node.get("view_count"),
                    },
                    "provider": record.get("provider") or node.get("api_provider"),
                    "provider_status": provider_status,
                    "crawl_status": node.get("crawl_status"),
                    "crawl_source": node.get("crawl_source"),
                    "llmscrapy_status": node.get("llmscrapy_status"),
                    "llmscrapy_error": node.get("llmscrapy_error"),
                    "llm_status": node.get("llm_status"),
                    "rule_candidate_edges": node.get("candidate_edges") or [],
                    "rule_tamper_analysis": node.get("tamper_analysis") or {},
                    "rule_influence_analysis": node.get("influence_analysis") or {},
                }
            )
        pairwise_evidence = self._build_pairwise_relation_evidence(nodes)
        return {
            "task": "image_propagation_source_and_topology_synthesis",
            "mainstream_source_platforms": ["weibo", "xiaohongshu"],
            "external_platform_policy": "platform=other is peripheral evidence only; do not choose it as source and do not build main edges among other nodes",
            "source_strategy_policy": (
                "If weibo and xiaohongshu have explicit repost/source/watermark account relation, choose single_source. "
                "If not, choose per_platform_sources. With only one mainstream platform, select a source inside that platform only."
            ),
            "tool_manifest": {
                "tikomni_fetch": "structured API evidence for weibo/xiaohongshu when available",
                "llmscrapy_pipeline": "enhanced external-page crawl plus LLM field extraction",
                "ocr_validator": "image OCR, watermark and visual variant evidence supplied by validator",
                "llm_deep_read": "LLM can decide whether weak crawler text is useful, but must cite evidence",
            },
            "nodes": compact_nodes,
            "pairwise_evidence": pairwise_evidence,
        }

    def _extract_ocr_relation_signals(self, text: str) -> Dict[str, Any]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        accounts = []
        for match in re.finditer(r"[@＠]\s*([A-Za-z0-9_\-\u4e00-\u9fff]{2,32})", normalized):
            account = match.group(1).strip()
            if account and account not in accounts:
                accounts.append(account)
        platform_mentions = []
        platform_patterns = {
            "weibo": ("微博", "weibo", "weibo.com", "weibo.cn"),
            "xiaohongshu": ("小红书", "xhs", "redbook", "xiaohongshu", "小红书号"),
            "douyin": ("抖音", "douyin"),
            "bilibili": ("bilibili", "哔哩哔哩", "b站"),
        }
        lower_text = normalized.lower()
        for platform, markers in platform_patterns.items():
            if any(marker.lower() in lower_text for marker in markers):
                platform_mentions.append(platform)
        source_phrases = []
        for marker in ("转载", "转自", "来源", "源自", "via", "from", "搬运", "原博", "原作者", "首发"):
            if marker.lower() in lower_text:
                source_phrases.append(marker)
        entity_terms = []
        for match in re.finditer(r"[A-Za-z][A-Za-z0-9 .,&:-]{2,40}|[\u4e00-\u9fff]{2,12}", normalized):
            term = match.group(0).strip(" ,，。:：!！")
            if len(term) >= 2 and term not in entity_terms:
                entity_terms.append(term)
            if len(entity_terms) >= 20:
                break
        return {
            "accounts": accounts,
            "platform_mentions": platform_mentions,
            "source_phrases": source_phrases,
            "entity_terms": entity_terms,
            "has_account_signature": bool(accounts),
        }

    def _build_pairwise_relation_evidence(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        evidence_items: List[Dict[str, Any]] = []
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1:]:
                left_id = str(left.get("id") or "")
                right_id = str(right.get("id") or "")
                if not left_id or not right_id:
                    continue
                left_platform = self._normalize_platform(left.get("platform"), str(left.get("url") or ""))
                right_platform = self._normalize_platform(right.get("platform"), str(right.get("url") or ""))
                signals = []
                text_similarity = self._text_similarity(
                    self._node_text_for_similarity(left),
                    self._node_text_for_similarity(right),
                )
                if text_similarity >= 0.45:
                    signals.append(f"text_or_ocr_similarity={text_similarity:.2f}")
                if self._ocr_mentions_parent(left, right):
                    signals.append("left_ocr_mentions_right_publisher_or_account")
                if self._ocr_mentions_parent(right, left):
                    signals.append("right_ocr_mentions_left_publisher_or_account")
                if self._watermark_mentions_parent(left, right):
                    signals.append("left_watermark_mentions_right")
                if self._watermark_mentions_parent(right, left):
                    signals.append("right_watermark_mentions_left")
                distance = self._publish_time_distance_hours(left, right)
                if distance is not None:
                    signals.append(f"publish_time_distance_hours={distance:.1f}")
                if not signals:
                    continue
                evidence_items.append({
                    "left": left_id,
                    "right": right_id,
                    "left_platform": left_platform,
                    "right_platform": right_platform,
                    "signals": signals,
                    "policy_hint": (
                        "mainstream-mainstream edge candidate"
                        if self._is_mainstream_platform(left) and self._is_mainstream_platform(right)
                        else "external evidence only; do not create main edge if either side is other"
                    ),
                })
        return evidence_items

    def _apply_llm_topology_synthesis(self, nodes: List[Dict[str, Any]], synthesis: Dict[str, Any]) -> None:
        if synthesis.get("status") != "success":
            for node in nodes:
                node["llm_topology_status"] = synthesis.get("status", "failed")
                node["llm_topology_reason"] = synthesis.get("reason") or synthesis.get("error") or ""
            self._enforce_source_and_edge_guards(nodes)
            return

        node_by_id = {str(node.get("id") or ""): node for node in nodes}
        source_decision = synthesis.get("source_decision") if isinstance(synthesis.get("source_decision"), dict) else {}
        edges = synthesis.get("edges") if isinstance(synthesis.get("edges"), list) else []

        for node in nodes:
            node["llm_topology_status"] = "success"
            node["llm_topology_reason"] = "; ".join(str(item) for item in self._coerce_list(source_decision.get("reasoning")))[:500]
            node["is_suspected_source"] = False
            node["is_platform_source"] = False
            node["parent_id"] = None
            node["edge_type_from_parent"] = "isolated"
            node["edge_weight_from_parent"] = 0.0
            node["edge_method_from_parent"] = ""
            node["edge_score_components_from_parent"] = {}
            node["edge_evidence_from_parent"] = []
            node["candidate_edges"] = []

        mode = str(source_decision.get("global_source_mode") or "").strip()
        global_source_id = str(source_decision.get("global_source_node_id") or "")
        platform_sources = source_decision.get("platform_sources") if isinstance(source_decision.get("platform_sources"), dict) else {}

        if mode == "single_source" and self._is_mainstream_node_id(node_by_id, global_source_id):
            source_node = node_by_id[global_source_id]
            source_node["is_suspected_source"] = True
            source_node["is_platform_source"] = True
            source_node["is_key_node"] = True
        else:
            for platform in ("weibo", "xiaohongshu"):
                candidate_id = str(platform_sources.get(platform) or "")
                if not self._is_mainstream_node_id(node_by_id, candidate_id):
                    candidate = self._fallback_platform_source_node(nodes, platform)
                    candidate_id = str(candidate.get("id")) if candidate else ""
                if candidate_id and candidate_id in node_by_id:
                    source_node = node_by_id[candidate_id]
                    source_node["is_suspected_source"] = True
                    source_node["is_platform_source"] = True
                    source_node["is_key_node"] = True
            if not any(node.get("is_suspected_source") for node in nodes):
                candidate = self._fallback_global_mainstream_source_node(nodes)
                if candidate:
                    candidate["is_suspected_source"] = True
                    candidate["is_platform_source"] = True
                    candidate["is_key_node"] = True

        for raw_edge in edges:
            if not isinstance(raw_edge, dict):
                continue
            edge = self._sanitize_llm_edge(raw_edge, node_by_id)
            if not edge:
                continue
            child = node_by_id[str(edge["target"])]
            child["parent_id"] = str(edge["source"])
            child["edge_type_from_parent"] = edge["edge_type"]
            child["edge_weight_from_parent"] = edge["edge_weight"]
            child["edge_method_from_parent"] = edge.get("method", edge["edge_type"])
            child["edge_score_components_from_parent"] = edge.get("score_components", {})
            child["edge_evidence_from_parent"] = edge["evidence"]
            child["candidate_edges"].append(edge)

        self._enforce_source_and_edge_guards(nodes)
        self._mark_topology_visibility(nodes)

    def _sanitize_llm_edge(self, edge: Dict[str, Any], node_by_id: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target or source == target:
            return None
        if source not in node_by_id or target not in node_by_id:
            return None
        source_node = node_by_id[source]
        target_node = node_by_id[target]
        if not self._is_mainstream_platform(source_node) or not self._is_mainstream_platform(target_node):
            return None
        evidence = [str(item) for item in self._coerce_list(edge.get("evidence")) if str(item).strip()]
        edge_type = str(edge.get("edge_type") or "isolated")
        allowed = {
            "explicit_repost",
            "explicit_source_url",
            "watermark_account_match",
            "ocr_account_match",
            "cross_platform_watermark",
            "REPOST",
            "CROSS_PLATFORM",
            "duplicate_cluster",
        }
        if edge_type not in allowed:
            return None
        if not evidence:
            return None
        weight = self._clamp_float(edge.get("edge_weight"))
        return {
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "edge_weight": round(weight or 0.5, 2),
            "evidence": evidence[:5],
        }

    def _merge_cross_platform_relations(self, existing: Any, llm_relations: Any) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for relation in [*(existing if isinstance(existing, list) else []), *(llm_relations if isinstance(llm_relations, list) else [])]:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source") or "")
            target = str(relation.get("target") or "")
            relation_type = str(relation.get("relation_type") or "possible_cross_platform_relation")
            key = (source, target, relation_type)
            if key in seen:
                continue
            seen.add(key)
            merged.append({
                "source": source,
                "target": target,
                "relation_type": relation_type,
                "confidence": self._clamp_float(relation.get("confidence")),
                "evidence": [str(item) for item in self._coerce_list(relation.get("evidence"))],
            })
        merged.sort(key=lambda item: self._clamp_float(item.get("confidence")), reverse=True)
        return merged

    def _merge_llm_summary(self, summary: Dict[str, Any], synthesis: Dict[str, Any]) -> None:
        source_decision = synthesis.get("source_decision") if isinstance(synthesis.get("source_decision"), dict) else {}
        if source_decision:
            summary["global_source_mode"] = source_decision.get("global_source_mode", summary.get("global_source_mode", "unknown"))
            summary["global_source_node_id"] = source_decision.get("global_source_node_id", summary.get("global_source_node_id", ""))
            summary["platform_sources"] = source_decision.get("platform_sources", summary.get("platform_sources", {}))
            summary["source_reasoning"] = source_decision.get("reasoning", summary.get("source_reasoning", []))
        llm_risks = synthesis.get("risk_summary") if isinstance(synthesis.get("risk_summary"), list) else []
        if llm_risks:
            summary["risk_summary"] = [*summary.get("risk_summary", []), *llm_risks]

    def _fallback_platform_sources(self, nodes: List[Dict[str, Any]]) -> Dict[str, str]:
        return {
            platform: str(candidate.get("id"))
            for platform in ("weibo", "xiaohongshu")
            if (candidate := self._fallback_platform_source_node(nodes, platform))
        }

    def _fallback_platform_source_node(self, nodes: List[Dict[str, Any]], platform: str) -> Optional[Dict[str, Any]]:
        candidates = [
            node for node in nodes
            if self._normalize_platform(node.get("platform"), str(node.get("url") or "")) == platform
            and node.get("published_at")
        ]
        if not candidates:
            return None
        return min(candidates, key=self._source_selection_key)

    def _fallback_global_mainstream_source_node(self, nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        candidates = [node for node in nodes if self._is_mainstream_platform(node) and node.get("published_at")]
        if not candidates:
            return None
        return min(candidates, key=self._source_selection_key)

    def _is_mainstream_platform(self, node: Dict[str, Any]) -> bool:
        return self._normalize_platform(node.get("platform"), str(node.get("url") or "")) in {"weibo", "xiaohongshu"}

    def _is_mainstream_node_id(self, node_by_id: Dict[str, Dict[str, Any]], node_id: str) -> bool:
        return bool(node_id and node_id in node_by_id and self._is_mainstream_platform(node_by_id[node_id]))

    def _enforce_source_and_edge_guards(self, nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            if not self._is_mainstream_platform(node):
                node["is_suspected_source"] = False
                node["is_platform_source"] = False
                node["parent_id"] = None
                node["edge_type_from_parent"] = "isolated"
                node["edge_weight_from_parent"] = 0.0
                node["edge_method_from_parent"] = ""
                node["edge_score_components_from_parent"] = {}
                node["edge_evidence_from_parent"] = []
                node["candidate_edges"] = []
        node_by_id = {str(node.get("id") or ""): node for node in nodes}
        for node in nodes:
            parent_id = str(node.get("parent_id") or "")
            if not parent_id:
                continue
            parent = node_by_id.get(parent_id)
            if not parent or not self._is_mainstream_platform(parent) or not self._is_mainstream_platform(node):
                node["parent_id"] = None
                node["edge_type_from_parent"] = "isolated"
                node["edge_weight_from_parent"] = 0.0
                node["edge_method_from_parent"] = ""
                node["edge_score_components_from_parent"] = {}
                node["edge_evidence_from_parent"] = []

    @staticmethod
    def _truncate_text(value: Any, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        return text if len(text) <= limit else f"{text[:limit - 1]}…"

    def _assign_matrix_candidates(self, nodes: List[Dict[str, Any]]) -> None:
        for node in nodes:
            node["matrix_account_analysis"] = {
                "is_matrix_candidate": False,
                "related_node_ids": [],
                "confidence": 0.0,
                "evidence": [],
            }
            node["is_matrix_account_candidate"] = False

        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1:]:
                confidence, evidence = self._matrix_pair_score(left, right)
                if confidence < 0.65:
                    continue
                for node, other in ((left, right), (right, left)):
                    analysis = node["matrix_account_analysis"]
                    analysis["is_matrix_candidate"] = True
                    analysis["confidence"] = max(self._clamp_float(analysis.get("confidence")), confidence)
                    analysis["evidence"] = list(dict.fromkeys([*analysis.get("evidence", []), *evidence]))
                    related = analysis.get("related_node_ids", [])
                    other_id = str(other.get("id"))
                    if other_id not in related:
                        related.append(other_id)
                    analysis["related_node_ids"] = related
                    node["is_matrix_account_candidate"] = True

    def _matrix_pair_score(self, left: Dict[str, Any], right: Dict[str, Any]) -> tuple[float, List[str]]:
        evidence: List[str] = []
        score = 0.0
        left_name = str(left.get("publisher") or left.get("author") or "")
        right_name = str(right.get("publisher") or right.get("author") or "")
        name_similarity = self._text_similarity(left_name, right_name)
        if name_similarity >= 0.82:
            score += 0.35
            evidence.append(f"publisher names are similar ({name_similarity:.2f})")

        left_record = left.get("normalized_record") if isinstance(left.get("normalized_record"), dict) else {}
        right_record = right.get("normalized_record") if isinstance(right.get("normalized_record"), dict) else {}
        left_user = left_record.get("user", {}) if isinstance(left_record.get("user"), dict) else {}
        right_user = right_record.get("user", {}) if isinstance(right_record.get("user"), dict) else {}
        desc_similarity = self._text_similarity(left_user.get("description"), right_user.get("description"))
        if desc_similarity >= 0.78:
            score += 0.15
            evidence.append(f"user descriptions are similar ({desc_similarity:.2f})")

        verify_similarity = self._text_similarity(left_user.get("verified_reason"), right_user.get("verified_reason"))
        if verify_similarity >= 0.78:
            score += 0.20
            evidence.append(f"verified reasons are similar ({verify_similarity:.2f})")

        text_similarity = self._text_similarity(
            self._node_text_for_similarity(left),
            self._node_text_for_similarity(right),
        )
        if text_similarity >= 0.82:
            score += 0.25
            evidence.append(f"post/OCR text is similar ({text_similarity:.2f})")

        if self._watermark_mentions_parent(left, right) or self._watermark_mentions_parent(right, left):
            score += 0.25
            evidence.append("watermark account/platform links the two publishers")

        distance = self._publish_time_distance_hours(left, right)
        if distance is not None and distance <= 6:
            score += 0.12
            evidence.append(f"publish times are close ({distance:.1f}h)")

        return round(min(score, 1.0), 2), evidence

    def _build_cross_platform_relations(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        relations: List[Dict[str, Any]] = []
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1:]:
                left_platform = self._normalize_platform(left.get("platform"), str(left.get("url") or ""))
                right_platform = self._normalize_platform(right.get("platform"), str(right.get("url") or ""))
                if left_platform == right_platform:
                    continue
                evidence: List[str] = []
                confidence = 0.0
                if self._watermark_mentions_parent(left, right):
                    confidence += 0.45
                    evidence.append("left node watermark points to right publisher/platform")
                if self._watermark_mentions_parent(right, left):
                    confidence += 0.45
                    evidence.append("right node watermark points to left publisher/platform")

                text_similarity = self._text_similarity(
                    self._node_text_for_similarity(left),
                    self._node_text_for_similarity(right),
                )
                if text_similarity >= 0.65:
                    confidence += 0.25
                    evidence.append(f"text/OCR similarity={text_similarity:.2f}")

                publisher_similarity = self._text_similarity(
                    left.get("publisher") or left.get("author"),
                    right.get("publisher") or right.get("author"),
                )
                if publisher_similarity >= 0.75:
                    confidence += 0.20
                    evidence.append(f"publisher similarity={publisher_similarity:.2f}")

                if confidence <= 0:
                    continue
                relations.append(
                    {
                        "source": str(left.get("id")),
                        "target": str(right.get("id")),
                        "relation_type": "possible_cross_platform_relation",
                        "confidence": round(min(confidence, 1.0), 2),
                        "evidence": evidence,
                    }
                )
        relations.sort(key=lambda item: self._clamp_float(item.get("confidence")), reverse=True)
        return relations

    def _build_node_label(self, node: Dict[str, Any]) -> str:
        publisher = node.get("publisher") or node.get("author") or "unknown"
        platform = self._normalize_platform(node.get("platform"), str(node.get("url") or ""))
        return f"{platform}:{publisher}"

    def _node_text_for_similarity(self, node: Dict[str, Any]) -> str:
        record = node.get("normalized_record") if isinstance(node.get("normalized_record"), dict) else {}
        post = record.get("post", {}) if isinstance(record.get("post"), dict) else {}
        return " ".join(
            str(value or "")
            for value in (
                node.get("title"),
                node.get("description"),
                node.get("candidate_content_text"),
                post.get("title"),
                post.get("text"),
            )
        )

    def _publish_time_distance_hours(self, left: Dict[str, Any], right: Dict[str, Any]) -> Optional[float]:
        left_dt = self._parse_normalized_datetime(left.get("published_at"))
        right_dt = self._parse_normalized_datetime(right.get("published_at"))
        if not left_dt or not right_dt:
            return None
        return abs((left_dt - right_dt).total_seconds()) / 3600.0

    @staticmethod
    def _parse_normalized_datetime(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    def _assign_topology(self, nodes: List[Dict[str, Any]]) -> None:
        total = len(nodes)
        for index, node in enumerate(nodes):
            node["node_weight"] = self.calculate_node_weight(node, index, total)
            node["source_score"] = self.calculate_source_score(node, index, total)
            node["is_key_node"] = False
            node["is_suspected_source"] = False
            node["is_platform_source"] = False
            node["parent_id"] = None
            node["edge_weight_from_parent"] = 0.0
            node["edge_type_from_parent"] = "isolated"
            node["edge_method_from_parent"] = ""
            node["edge_score_components_from_parent"] = {}
            node["edge_evidence_from_parent"] = []
            node["candidate_edges"] = []

        platform_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for node in nodes:
            platform = self._normalize_platform(node.get("platform"), str(node.get("url") or ""))
            if platform in {"weibo", "xiaohongshu"}:
                platform_groups[platform].append(node)
            elif self._boolish(node.get("allow_in_external_timeline")):
                family = str(node.get("platform_family") or "unknown")
                platform_groups[f"external:{family}"].append(node)
        for platform_nodes in platform_groups.values():
            known_time_nodes = [node for node in platform_nodes if node.get("published_at")]
            if known_time_nodes:
                source_node = min(known_time_nodes, key=self._source_selection_key)
            else:
                source_node = max(
                    platform_nodes,
                    key=lambda item: (
                        self._clamp_float(item.get("source_score")),
                        self._clamp_float(item.get("node_weight")),
                    ),
                )
            source_node["is_suspected_source"] = True
            source_node["is_platform_source"] = True

        self._select_key_nodes_by_platform(nodes)

        for index in range(1, len(nodes)):
            if nodes[index].get("is_suspected_source"):
                continue
            nodes[index]["candidate_edges"] = self._build_candidate_edges(nodes, index)
            parent = self._select_parent(nodes, index)
            if parent is None:
                continue
            edge = self._infer_edge(parent, nodes[index])
            nodes[index]["parent_id"] = parent.get("id")
            nodes[index]["edge_type_from_parent"] = edge.get("edge_type", "isolated")
            nodes[index]["edge_weight_from_parent"] = edge.get("edge_weight", 0.0)
            nodes[index]["edge_method_from_parent"] = edge.get("method", "")
            nodes[index]["edge_score_components_from_parent"] = edge.get("score_components") or {}
            nodes[index]["edge_evidence_from_parent"] = edge.get("evidence", [])

        self._mark_topology_visibility(nodes)

    def _select_key_nodes_by_platform(self, nodes: List[Dict[str, Any]]) -> None:
        platform_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for node in nodes:
            platform = self._normalize_platform(node.get("platform"), str(node.get("url") or ""))
            if platform in {"weibo", "xiaohongshu"}:
                platform_groups[platform].append(node)
            elif self._boolish(node.get("allow_in_external_timeline")):
                family = str(node.get("platform_family") or "unknown")
                platform_groups[f"external:{family}"].append(node)

        for platform_nodes in platform_groups.values():
            if not platform_nodes:
                continue
            max_count = min(
                self.config.max_key_nodes_per_platform,
                max(1, ceil(len(platform_nodes) * self.config.key_node_platform_ratio)),
            )
            ranked = sorted(
                platform_nodes,
                key=lambda item: (
                    self._clamp_float(item.get("node_weight")),
                    self._clamp_float((item.get("influence_analysis") or {}).get("influence_score")),
                    self._engagement_total(item),
                    self._to_int(item.get("follower_count")),
                ),
                reverse=True,
            )
            selected = 0
            for node in ranked:
                if selected >= max_count:
                    break
                if not self._is_key_topology_candidate(node):
                    continue
                node["is_key_node"] = True
                selected += 1

            if selected == 0:
                fallback = ranked[0]
                if (
                    fallback.get("is_suspected_source")
                    or self._engagement_total(fallback) > 0
                    or self._clamp_float(fallback.get("node_weight")) >= self.config.topology_visibility_threshold
                ):
                    fallback["is_key_node"] = True

    def _mark_topology_visibility(self, nodes: List[Dict[str, Any]]) -> None:
        candidates = [
            node
            for node in nodes
            if node.get("is_suspected_source")
            or node.get("is_key_node")
            or node.get("parent_id")
            or (node.get("duplicate_analysis") or {}).get("cluster_member_ids")
            or self._clamp_float(node.get("node_weight")) >= self.config.topology_visibility_threshold
        ]
        selected_ids = {str(node.get("id")) for node in candidates}
        if not selected_ids and nodes:
            selected_ids = {str(nodes[0].get("id"))}

        for node in nodes:
            should_show = str(node.get("id")) in selected_ids
            node["is_topology_visible"] = should_show
            node["topology_omit_reason"] = (
                None
                if should_show
                else "传播效益较弱，未进入拓扑主图，可在报告中简要提及。"
            )

    def _source_selection_key(self, node: Dict[str, Any]) -> tuple[str, float, float, float]:
        """Earliest reliable publish time is the primary source signal."""
        return (
            str(node.get("published_at") or self.config.unknown_time_sort_value),
            -self._time_reliability(node),
            self._source_domain_penalty(node),
            -self._clamp_float(node.get("source_score")),
        )

    def _select_parent(self, nodes: List[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
        current = nodes[index]
        if not self._is_mainstream_platform(current) and not self._boolish(
            current.get("allow_cross_platform_relation_candidate")
        ):
            return None
        candidates = [
            node
            for node in nodes[:index]
            if node.get("id") != current.get("id")
            and (
                self._is_mainstream_platform(node)
                or self._boolish(node.get("allow_cross_platform_relation_candidate"))
            )
        ]
        if not candidates:
            return None

        candidate_edges = self._build_candidate_edges(nodes, index)
        if candidate_edges:
            best_parent_id = candidate_edges[0].get("source")
            matched = next((node for node in candidates if node.get("id") == best_parent_id), None)
            if matched:
                return matched
        return None

    def _calculate_edge_weight(self, parent: Dict[str, Any], child: Dict[str, Any]) -> float:
        inferred = self._infer_edge(parent, child)
        inferred_weight = self._clamp_float(inferred.get("edge_weight"))
        if inferred_weight:
            return inferred_weight
        domain_bonus = 0.15 if parent.get("domain") == child.get("domain") else 0.0
        relation_bonus = 0.15 if child.get("public_relation_hint") else 0.0
        source_bonus = 0.12 if parent.get("is_suspected_source") else 0.0
        key_parent_bonus = 0.06 if parent.get("is_key_node") else 0.0
        confidence = self._clamp_float(child.get("llm_confidence"))
        child_weight = self._clamp_float(child.get("node_weight"))
        weight = (
            0.30
            + domain_bonus
            + relation_bonus
            + source_bonus
            + key_parent_bonus
            + confidence * 0.2
            + child_weight * 0.15
        )
        return round(min(weight, 1.0), 2)

    def _build_candidate_edges(self, nodes: List[Dict[str, Any]], index: int) -> List[Dict[str, Any]]:
        child = nodes[index]
        if not self._is_mainstream_platform(child) and not self._boolish(
            child.get("allow_cross_platform_relation_candidate")
        ):
            return []
        edges = []
        for parent in nodes[:index]:
            if parent.get("id") == child.get("id"):
                continue
            if not self._is_mainstream_platform(parent) and not self._boolish(
                parent.get("allow_cross_platform_relation_candidate")
            ):
                continue
            edge = self._infer_edge(parent, child)
            if self._clamp_float(edge.get("edge_weight")) <= 0:
                continue
            edges.append(edge)
        edges.sort(key=lambda item: self._clamp_float(item.get("edge_weight")), reverse=True)
        return edges[:5]

    def _infer_edge(self, parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
        parent_record = parent.get("normalized_record") if isinstance(parent.get("normalized_record"), dict) else {}
        child_record = child.get("normalized_record") if isinstance(child.get("normalized_record"), dict) else {}
        parent_post = parent_record.get("post", {}) if isinstance(parent_record.get("post"), dict) else {}
        child_post = child_record.get("post", {}) if isinstance(child_record.get("post"), dict) else {}

        parent_id = parent.get("id")
        child_id = child.get("id")
        candidates: List[Dict[str, Any]] = []

        parent_post_id = str(
            parent_post.get("post_id")
            or parent.get("post_id")
            or parent.get("note_id")
            or parent.get("status_id")
            or parent.get("idstr")
            or ""
        )
        child_original_id = str(
            child_post.get("original_post_id")
            or child.get("original_post_id")
            or child.get("retweeted_status_id")
            or child.get("retweeted_id")
            or child.get("retweeted_mid")
            or self._nested_get(child, "retweeted_status.idstr")
            or self._nested_get(child, "retweeted_status.mid")
            or self._nested_get(child, "raw.retweeted_status.idstr")
            or ""
        )
        if parent_post_id and child_original_id and parent_post_id == child_original_id:
            candidates.append({
                "edge_type": "REPOST",
                "edge_weight": 0.98,
                "method": "relations.original_post_id",
                "evidence": [f"child original_post_id matches parent post_id: {parent_post_id}"],
            })

        parent_url = str(parent.get("url") or parent_record.get("source_url") or "")
        child_source_url = str(child_post.get("source_url") or child.get("source_url") or "")
        if parent_url and child_source_url and (parent_url in child_source_url or child_source_url in parent_url):
            candidates.append({
                "edge_type": "REPOST",
                "edge_weight": 0.92,
                "method": "relations.source_url",
                "evidence": [f"child source_url references parent: {child_source_url}"],
            })

        if self._watermark_mentions_parent(child, parent):
            candidates.append({
                "edge_type": "watermark_account_match",
                "edge_weight": 0.7,
                "method": "watermark+publisher",
                "evidence": ["child watermark account/platform matches parent publisher/platform"],
            })

        if self._ocr_mentions_parent(child, parent):
            candidates.append({
                "edge_type": "ocr_account_match",
                "edge_weight": 0.68,
                "method": "ocr+publisher",
                "evidence": ["child OCR text contains parent publisher/account signature"],
            })

        parent_platform = self._normalize_platform(parent.get("platform"), str(parent.get("url") or ""))
        child_platform = self._normalize_platform(child.get("platform"), str(child.get("url") or ""))
        child_watermark_platforms = [
            self._normalize_platform(item)
            for item in self._coerce_list(child.get("watermark_platforms"))
        ]
        if parent_platform != child_platform and parent_platform in child_watermark_platforms:
            candidates.append({
                "edge_type": "CROSS_PLATFORM",
                "edge_weight": 0.72,
                "method": "watermark+time",
                "evidence": [f"child platform={child_platform} carries watermark platform={parent_platform}"],
            })

        if candidates:
            best = max(candidates, key=lambda item: self._clamp_float(item.get("edge_weight")))
        else:
            best = {"edge_type": "isolated", "edge_weight": 0.0, "method": "none", "evidence": ["no reliable parent evidence"]}

        confidence = round(self._clamp_float(best.get("edge_weight")), 2)
        return {
            "source": parent_id,
            "target": child_id,
            "from": parent_id,
            "to": child_id,
            "edge_type": best.get("edge_type", "isolated"),
            "edge_weight": confidence,
            "confidence": confidence,
            "method": best.get("method") or best.get("edge_type", "isolated"),
            "evidence": best.get("evidence", []),
            "score_components": best.get("score_components") or {},
        }

    def _watermark_mentions_parent(self, child: Dict[str, Any], parent: Dict[str, Any]) -> bool:
        parent_platform = self._normalize_platform(parent.get("platform"), str(parent.get("url") or ""))
        parent_publisher = str(parent.get("publisher") or parent.get("author") or "").lower()
        watermark_platforms = [
            self._normalize_platform(item)
            for item in self._coerce_list(child.get("watermark_platforms"))
        ]
        if watermark_platforms and parent_platform in watermark_platforms:
            return True

        accounts = child.get("watermark_accounts")
        account_values: List[str] = []
        if isinstance(accounts, dict):
            for value in accounts.values():
                account_values.extend(str(item).lower() for item in self._coerce_list(value))
        else:
            account_values.extend(str(item).lower() for item in self._coerce_list(accounts))
        return bool(
            parent_publisher
            and account_values
            and any(parent_publisher in account or account in parent_publisher for account in account_values)
        )

    def _ocr_mentions_parent(self, child: Dict[str, Any], parent: Dict[str, Any]) -> bool:
        validation_signals = child.get("validation_signals") if isinstance(child.get("validation_signals"), dict) else {}
        ocr_text = " ".join(
            str(value or "")
            for value in (
                child.get("candidate_content_text"),
                validation_signals.get("candidate_content_text"),
                validation_signals.get("candidate_ocr_text"),
            )
        ).lower()
        if not ocr_text.strip():
            return False
        parent_record = parent.get("normalized_record") if isinstance(parent.get("normalized_record"), dict) else {}
        parent_user = parent_record.get("user", {}) if isinstance(parent_record.get("user"), dict) else {}
        candidates = [
            parent.get("publisher"),
            parent.get("author"),
            parent_user.get("username"),
            parent_user.get("nickname"),
            parent_user.get("user_id"),
        ]
        for value in candidates:
            text = str(value or "").strip().lower()
            if len(text) >= 3 and text in ocr_text:
                return True
        return False

    def _is_key_topology_candidate(self, node: Dict[str, Any]) -> bool:
        role = str(node.get("propagation_role") or "")
        role_markers = (
            "源头",
            "原创",
            "首发",
            "转载",
            "转发",
            "扩散",
            "媒体",
            "社交",
            "source",
            "original",
            "first",
            "repost",
            "share",
            "spread",
            "media",
            "social",
        )
        has_role_signal = any(marker in role for marker in role_markers)
        has_relation = bool(node.get("public_relation_hint"))
        engagement_total = self._engagement_total(node)
        node_weight = self._clamp_float(node.get("node_weight"))
        source_score = self._clamp_float(node.get("source_score"))

        return (
            node_weight >= self.config.key_node_weight_threshold
            or source_score >= 0.72
            or has_relation
            or has_role_signal
            or (
                engagement_total >= 1000
                and node_weight >= self.config.topology_visibility_threshold
            )
            or (
                bool((node.get("influence_analysis") or {}).get("is_big_v"))
                and (node_weight >= 0.45 or engagement_total >= 300)
            )
        )

    def _engagement_total(self, node: Dict[str, Any]) -> int:
        return (
            self._to_int(node.get("view_count"))
            + self._to_int(node.get("repost_count"))
            + self._to_int(node.get("comment_count"))
            + self._to_int(node.get("like_count"))
        )

    def _nearest_visible_parent_id(
        self,
        nodes: List[Dict[str, Any]],
        node: Dict[str, Any],
        visible_ids: set[str],
    ) -> Optional[Any]:
        parent_id = node.get("parent_id")
        while parent_id:
            if str(parent_id) in visible_ids:
                return parent_id
            parent = next((item for item in nodes if item.get("id") == parent_id), None)
            parent_id = parent.get("parent_id") if parent else None
        return None

    def _sort_key(self, node: Dict[str, Any]) -> str:
        return str(node.get("published_at") or self.config.unknown_time_sort_value)

    def _flatten_llm_external_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict) or not data:
            return {}
        content = data.get("content") if isinstance(data.get("content"), dict) else {}
        classification = (
            data.get("page_classification")
            if isinstance(data.get("page_classification"), dict)
            else {}
        )
        occurrence = (
            data.get("image_occurrence")
            if isinstance(data.get("image_occurrence"), dict)
            else {}
        )
        provenance = data.get("provenance") if isinstance(data.get("provenance"), dict) else {}
        metrics = (
            data.get("optional_metrics")
            if isinstance(data.get("optional_metrics"), dict)
            else data.get("metrics")
            if isinstance(data.get("metrics"), dict)
            else {}
        )
        decision = data.get("node_decision") if isinstance(data.get("node_decision"), dict) else {}
        evidence_node_status = str(decision.get("evidence_node_status") or "")
        status_map = {
            "direct_evidence": "confirmed_image_occurrence",
            "no_evidence": "rejected_after_page_review",
        }
        if evidence_node_status in status_map:
            decision = {**decision, "evidence_node_status": status_map[evidence_node_status]}

        flattened = dict(data)
        flattened["llm_page_analysis"] = data
        mapping = {
            "platform_family": classification.get("platform_family") or data.get("platform_family"),
            "page_type": classification.get("page_type") or data.get("page_type"),
            "title": content.get("title"),
            "description": content.get("description") or data.get("description"),
            "published_at": self.normalize_time(content.get("published_at") or data.get("published_at")),
            "modified_at": self.normalize_time(content.get("modified_at") or data.get("modified_at")),
            "publisher": content.get("publisher") or data.get("publisher"),
            "author": content.get("author") or data.get("author"),
            "canonical_url": content.get("canonical_url") or data.get("canonical_url"),
            "image_urls": content.get("image_urls") or data.get("image_urls"),
            "image_caption": occurrence.get("caption") or data.get("image_caption"),
            "image_credit": occurrence.get("image_credit") or data.get("image_credit"),
            "source_text": provenance.get("source_text") or data.get("source_text"),
            "source_url": provenance.get("source_url") or data.get("source_url"),
            "source_platform_hint": provenance.get("source_platform_hint"),
            "source_account_hint": provenance.get("source_account_hint"),
            "public_relation_hint": provenance.get("source_text") or provenance.get("source_url") or data.get("public_relation_hint"),
            "view_count": metrics.get("view_count", data.get("view_count")),
            "comment_count": metrics.get("comment_count", data.get("comment_count")),
            "like_count": metrics.get("like_count", data.get("like_count")),
            "share_count": metrics.get("share_count", data.get("share_count")),
            "repost_count": metrics.get("repost_count", data.get("repost_count")),
            "image_occurrence": occurrence,
            "provenance": provenance,
            "node_decision": decision,
            "evidence_node_status": decision.get("evidence_node_status"),
            "allow_in_external_timeline": decision.get("allow_in_external_timeline"),
            "allow_cross_platform_relation_candidate": decision.get("allow_cross_platform_relation_candidate"),
            "node_decision_reason": decision.get("reason"),
            "field_evidence": data.get("field_evidence"),
            "confidence": data.get("confidence") or classification.get("confidence") or occurrence.get("confidence") or provenance.get("confidence"),
            "reason": data.get("reason") or decision.get("reason"),
        }
        for key, value in mapping.items():
            if value not in (None, "", [], {}):
                flattened[key] = value
        return flattened

    def _count_for_node_output(self, data: Dict[str, Any], field: str, keep_missing_as_none: bool = False) -> Optional[int]:
        value = data.get(field)
        if keep_missing_as_none and value in (None, "", "null"):
            return None
        return self._to_int(value)

    def _nullable_count_for_display(self, value: Any) -> Optional[int]:
        if value in (None, "", "null"):
            return None
        return self._to_int(value)

    @staticmethod
    def _remove_rule_derived_external_metrics(data: Dict[str, Any]) -> None:
        for field in (
            "view_count",
            "like_count",
            "comment_count",
            "repost_count",
            "share_count",
            "collect_count",
        ):
            data.pop(field, None)

    def _merge_analysis(self, *items: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        count_fields = {
            "view_count",
            "repost_count",
            "comment_count",
            "like_count",
            "share_count",
            "collect_count",
            "follower_count",
            "following_count",
        }
        for item in items:
            for key, value in item.items():
                if value not in (None, "", [], {}):
                    if (
                        key in count_fields
                        and self._to_int(value) == 0
                        and self._to_int(merged.get(key)) > 0
                    ):
                        continue
                    merged[key] = value
        return merged

    def _extract_time_with_evidence(
        self,
        metadata: Dict[str, Any],
        text: str,
        url: str,
    ) -> tuple[Optional[str], Optional[str], Dict[str, str]]:
        evidence = {
            "search_result": "",
            "page_metadata": "",
            "time_tag": "",
            "visible_text": "",
            "url_pattern": "",
            "http_last_modified": "",
        }
        url_time = self._extract_platform_time_from_url(url) or self._extract_time_from_text(url)
        candidates = [
            ("page_metadata", self._extract_time_from_metadata(metadata)),
            ("time_tag", self.normalize_time(metadata.get("time_tag"))),
            ("visible_text", self._extract_time_from_text(text)),
            ("url_pattern", url_time),
            ("http_last_modified", self.normalize_time(metadata.get("http_last_modified"))),
        ]
        for source, value in candidates:
            if value:
                evidence[source] = value
        for source in ("page_metadata", "time_tag", "visible_text", "url_pattern", "http_last_modified"):
            if evidence[source]:
                return evidence[source], source, evidence
        return None, None, evidence

    def _extract_platform_time_from_url(self, url: str) -> Optional[str]:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path
        url_text = f"{path} {parsed.query}"
        if "douyin.com" in host:
            match = re.search(r"/(?:note|video)/(\d{16,22})", path)
            if match:
                decoded = self._decode_snowflake_seconds(match.group(1))
                if decoded:
                    return decoded
        compact_date = re.search(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])(?!\d)", url_text)
        if compact_date:
            return self.normalize_time(
                f"{compact_date.group(1)}-{compact_date.group(2)}-{compact_date.group(3)}"
            )
        separated_date = re.search(
            r"(?<!\d)(20\d{2})[-_/](0?[1-9]|1[0-2])[-_/]([0-2]?\d|3[01])(?!\d)",
            url_text,
        )
        if separated_date:
            return self.normalize_time(
                f"{separated_date.group(1)}-{separated_date.group(2)}-{separated_date.group(3)}"
            )
        return None

    @staticmethod
    def _decode_snowflake_seconds(value: str) -> Optional[str]:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return None
        seconds = numeric >> 32
        if seconds < 946684800 or seconds > 4102444800:
            return None
        return datetime.fromtimestamp(seconds).strftime("%Y-%m-%d %H:%M:%S")

    def _extract_time_from_metadata(self, metadata: Dict[str, Any]) -> Optional[str]:
        keys = [
            "publishedTime",
            "article:published_time",
            "article_published_time",
            "article:modified_time",
            "article_modified_time",
            "og:published_time",
            "og_published_time",
            "og:updated_time",
            "og_updated_time",
            "pubdate",
            "publish_time",
            "public_time",
            "pubtime",
            "datePublished",
            "publishDate",
            "published_at",
            "dateCreated",
            "create_time",
            "created",
            "date",
            "created_at",
            "uploadDate",
            "dateModified",
            "date_modified",
            "modified_time",
            "lastmod",
            "last_modified",
            "dc_terms_created",
            "dc_date_created",
            "dc_date",
            "dc.date",
            "dc.date.created",
            "dc.terms.created",
            "pubDate",
            "post_time",
            "postTime",
            "releaseDate",
            "release_time",
            "releaseTime",
            "display_time",
            "displayTime",
            "update_time",
            "updateTime",
            "modifiedTime",
            "modifyTime",
            "ctime",
            "mtime",
            "visible_time_hint",
        ]
        for key in keys:
            normalized = self.normalize_time(metadata.get(key))
            if normalized:
                return normalized
        normalized_metadata = {
            re.sub(r"[^a-z0-9]", "", str(key).lower()): value for key, value in metadata.items()
        }
        for key in keys:
            value = normalized_metadata.get(re.sub(r"[^a-z0-9]", "", key.lower()))
            normalized = self.normalize_time(value)
            if normalized:
                return normalized
        for key, value in metadata.items():
            if re.search(r"(publish|pubdate|posttime|release|displaytime|creat|update|modified|date|time)", str(key), re.I):
                normalized = self.normalize_time(value)
                if normalized:
                    return normalized
        return None

    def _extract_time_from_text(self, text: str) -> Optional[str]:
        relative_time = self._extract_relative_time_from_text(text)
        if relative_time:
            return relative_time

        label_patterns = [
            r"(?:发布时间|发布于|发表于|编辑于|更新于|创建时间|日期|时间)[:：\s]*([0-9]{2,4}[-/.年][0-9]{1,2}[-/.月][0-9]{1,2}(?:日)?(?:\s*[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)?)",
            r"(?:发布时间|发布于|发表于|编辑于|更新于|创建时间|日期|时间)[:：\s]*([0-9]{1,2}月[0-9]{1,2}日(?:\s*[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)?)",
            r"(?:published|posted|updated|created|date|time)\s*(?:at|on)?\s*[:：]?\s*([0-9]{2,4}[-/.年][0-9]{1,2}[-/.月][0-9]{1,2}(?:日)?(?:[ T]\s*[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?)?)",
        ]
        for pattern in label_patterns:
            match = re.search(pattern, text, flags=re.I)
            if not match:
                continue
            normalized = self.normalize_time(match.group(1))
            if normalized:
                return normalized

        patterns = [
            r"\d{4}-\d{1,2}-\d{1,2}T\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?",
            r"\d{4}\.\d{1,2}\.\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?",
            r"\d{4}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}(?::\d{2})?",
            r"\d{2}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}(?::\d{2})?",
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
            r"\d{4}\.\d{1,2}\.\d{1,2}",
            r"\d{4}年\d{1,2}月\d{1,2}日",
            r"\d{2}年\d{1,2}月\d{1,2}日",
            r"\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2}(?::\d{2})?",
            r"\d{1,2}月\d{1,2}日",
            r"\b20\d{2}(0[1-9]|1[0-2])([0-2]\d|3[01])\b",
            r"\b(?:1[0-9]{9}|[12][0-9]{12})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            normalized = self.normalize_time(match.group(0))
            if normalized:
                return normalized
        return None

    def _extract_relative_time_from_text(self, text: str) -> Optional[str]:
        now = datetime.now()
        match = re.search(r"(今天|昨日|昨天|前天)\s*(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
        if match:
            day_offset = {"今天": 0, "昨日": 1, "昨天": 1, "前天": 2}[match.group(1)]
            base = now.replace(hour=0, minute=0, second=0, microsecond=0)
            parsed = base.replace(
                day=base.day,
                hour=int(match.group(2)),
                minute=int(match.group(3)),
                second=int(match.group(4) or 0),
            )
            from datetime import timedelta

            return self._format_publish_time(parsed - timedelta(days=day_offset))

        match = re.search(r"(\d{1,2})月(\d{1,2})日(?:\s*(\d{1,2}):(\d{2})(?::(\d{2}))?)?", text)
        if match:
            hour = int(match.group(3) or 0)
            minute = int(match.group(4) or 0)
            second = int(match.group(5) or 0)
            try:
                parsed = datetime(now.year, int(match.group(1)), int(match.group(2)), hour, minute, second)
                if parsed > now:
                    parsed = parsed.replace(year=now.year - 1)
                return self._format_publish_time(parsed)
            except ValueError:
                return None

        from datetime import timedelta

        relative_patterns = [
            (r"(\d{1,3})\s*分钟前", "minutes"),
            (r"(\d{1,3})\s*小时前", "hours"),
            (r"(\d{1,3})\s*天前", "days"),
        ]
        for pattern, unit in relative_patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            amount = int(match.group(1))
            delta = timedelta(**{unit: amount})
            return self._format_publish_time(now - delta)
        return None

    def _extract_count(self, text: str, labels: List[str]) -> int:
        best = 0
        for label in labels:
            escaped = re.escape(label)
            patterns = [
                rf'["\']?{escaped}["\']?\s*[:=]\s*["\']?([0-9][0-9,.]*)\s*([万千kKmMwW]?)',
                rf'["\']?{escaped}["\']?\s*[:=]\s*\{{[^}}]*?["\']?(?:count|value|total)["\']?\s*[:=]\s*["\']?([0-9][0-9,.]*)\s*([万千kKmMwW]?)',
                rf"{escaped}\s*[:：]?\s*([0-9][0-9,.]*)\s*([万千kKmMwW]?)",
                rf"{escaped}[^\d]{{0,16}}([0-9][0-9,.]*)\s*([万千kKmMwW]?)",
            ]
            if re.search(r"[\u4e00-\u9fff]", label):
                patterns.append(rf"([0-9][0-9,.]*)\s*([万千kKmMwW]?)\s*{escaped}")
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    best = max(best, self._parse_count(match.group(1), match.group(2)))
        return best

    @staticmethod
    def _extract_relation_hint(text: str) -> Optional[str]:
        relation_patterns = [
            r"(转发自|来源于|转载自|引用自|via|from)\s*[:：]?\s*([^\n，。；;]{1,60})",
            r"(关注了|followed)\s*([^\n，。；;]{1,60})",
        ]
        for pattern in relation_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return None

    @staticmethod
    def _infer_role_from_text(text: str) -> str:
        lowered = text.lower()
        if any(keyword in text for keyword in ["原创", "首发", "作者发布", "原始发布"]):
            return "疑似源头"
        if any(keyword in text for keyword in ["转载", "来源于", "引用自"]) or "via" in lowered:
            return "媒体转载"
        if any(keyword in text for keyword in ["转发", "分享", "评论"]) or "share" in lowered:
            return "社交扩散"
        return "未知"

    @staticmethod
    def _first_value(metadata: Dict[str, Any], keys: List[str]) -> Optional[str]:
        for key in keys:
            value = metadata.get(key)
            if value not in (None, "", [], {}):
                return str(value)
        return None

    @staticmethod
    def _parse_count(value: Any, unit: str = "") -> int:
        try:
            number = float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return 0
        multiplier = {
            "万": 10000,
            "千": 1000,
            "k": 1000,
            "K": 1000,
            "m": 1000000,
            "M": 1000000,
            "w": 10000,
            "W": 10000,
        }.get(unit, 1)
        return int(number * multiplier)

    @classmethod
    def _to_int(cls, value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            match = re.search(r"([0-9,.]+)\s*([万千kKmM]?)", value)
            if match:
                return cls._parse_count(match.group(1), match.group(2))
        return 0

    @staticmethod
    def _clamp_float(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(max(number, 0.0), 1.0)

    @staticmethod
    def _safe_mermaid_id(value: Any) -> str:
        raw = str(value or "node")
        safe = re.sub(r"[^0-9A-Za-z_]", "_", raw)
        if not safe or safe[0].isdigit():
            safe = f"N_{safe}"
        return safe

    @staticmethod
    def _escape_mermaid_label(value: str) -> str:
        return value.replace('"', "'").replace("\n", "<br/>")


def parse_node(state: AgentState) -> AgentState:
    """时空解析智能体节点：抓取网页证据，补全时间、传播指标和拓扑字段。"""
    agent = TimeSpaceAnalyzerAgent()
    result = agent.parse(state)
    # 如果有 progress_callback，转发进度日志
    callback = state.get("_progress_callback") if isinstance(state, dict) else None
    if callback:
        for log_line in result.get("execution_logs", []):
            if "stage" in log_line.lower() or "progress" in log_line.lower() or "llmscrapy" in log_line.lower():
                callback("parse_progress", 0, 0, 0, log_line)
    return result


def write_mermaid_preview(mermaid_graph: str, output_path: Path) -> None:
    """Write a standalone HTML preview for local Analyzer debugging."""
    escaped_graph = escape(mermaid_graph)
    escaped_source = escape(mermaid_graph)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Analyzer Mermaid Preview</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #f8fafc;
      color: #1f2937;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: 22px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 16px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }}
    .mermaid {{
      display: flex;
      justify-content: center;
      overflow-x: auto;
      min-height: 240px;
      background: #ffffff;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #f8fafc;
      color: #334155;
      border: 1px solid #e2e8f0;
      padding: 12px;
      border-radius: 6px;
      overflow-x: auto;
    }}
    h2 {{
      font-size: 16px;
      margin: 0 0 12px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Analyzer Mermaid Preview</h1>
    <section class="panel">
      <pre class="mermaid">{escaped_graph}</pre>
    </section>
    <section class="panel">
      <h2>Mermaid Source</h2>
      <pre>{escaped_source}</pre>
    </section>
  </main>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{
      startOnLoad: true,
      securityLevel: "loose",
      theme: "base",
      themeVariables: {{
        background: "#ffffff",
        mainBkg: "#ffffff",
        primaryColor: "#f8fafc",
        primaryTextColor: "#1f2937",
        primaryBorderColor: "#94a3b8",
        lineColor: "#64748b",
        edgeLabelBackground: "#ffffff",
        fontFamily: "Arial, Microsoft YaHei, sans-serif"
      }}
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def build_mock_state() -> AgentState:
    return {
        "nodes_data": [
            {
                "id": "n2",
                "title": "地方新闻转载：暴雨中的暖心瞬间",
                "url": "https://local-news.example.com/city/rain-street-story",
                "source_type": "news_repost",
                "similarity": 0.93,
                "is_similar": True,
            },
            {
                "id": "n1",
                "title": "摄影师原始发布：城市暴雨后的街角",
                "url": "https://photo-origin.example.cn/posts/rain-street-original",
                "source_type": "original_post",
                "similarity": 0.97,
                "is_similar": True,
            },
            {
                "id": "n3",
                "title": "社交平台热帖：这张图刷屏了",
                "url": "https://social.example.net/status/88481231",
                "source_type": "social_share",
                "similarity": 0.88,
                "is_similar": True,
            },
        ],
        "execution_logs": [],
    }


def load_debug_state(nodes_path: Optional[Path]) -> AgentState:
    if nodes_path is None:
        return build_mock_state()

    payload = json.loads(nodes_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        nodes_data = payload
        target_image: Dict[str, Any] = {"filename": "debug.jpg"}
    elif isinstance(payload, dict):
        nodes_data = payload.get("nodes_data") or payload.get("nodes") or []
        target_image = payload.get("target_image", {"filename": "debug.jpg"})
    else:
        raise ValueError("debug nodes file must be a JSON list or AgentState-like object")

    if not isinstance(nodes_data, list):
        raise ValueError("nodes_data must be a list")

    return {
        "target_image": target_image,
        "nodes_data": nodes_data,
        "execution_logs": [],
    }


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug the Analyzer node only.")
    parser.add_argument(
        "--nodes",
        type=Path,
        default=None,
        help="Path to a JSON file containing nodes_data list or AgentState-like object.",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "analyzer_mermaid_preview.html",
        help="Path to write the Mermaid HTML preview.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    mock_state = load_debug_state(args.nodes)
    result = parse_node(mock_state)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    preview_path = args.preview
    write_mermaid_preview(result.get("mermaid_graph", "graph TD"), preview_path)
    print(f"\nMermaid preview written to: {preview_path}")
