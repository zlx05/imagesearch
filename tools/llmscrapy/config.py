"""Configuration management for llmscrapy.

Loads environment variables from .env file automatically via python-dotenv.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Auto-load .env from project root ────────────────────────────────
# Searches upward from this file's location to find .env
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=False)
    except ImportError:
        pass  # python-dotenv not installed, rely on system env vars


@dataclass
class LLMConfig:
    """DeepSeek (OpenAI-compatible) API configuration."""

    api_key: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", "")
    )
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
    )
    model: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    )
    max_tokens: int = 4096  # larger output for enriched schema
    temperature: float = 0.0  # deterministic output for extraction


@dataclass
class CrawlerConfig:
    """Crawler configuration."""

    # Fetcher type: "direct" | "jina"
    fetcher_type: str = "direct"

    # HTTP request
    request_timeout: int = 30
    max_retries: int = 3
    warm_cookies: bool = True  # visit homepage first for cookies
    user_agent: str = ""  # deprecated — DirectFetcher uses UA pool

    # Parsing
    max_text_length: int = 8000  # max chars to send to LLM

    # Pipeline
    max_workers: int = 3  # parallel workers for batch processing (1 = sequential)
    enrich_mode: str = "baidu"  # "none" | "baidu" | "playwright" | "both"

    # Output
    output_dir: str = "output"
