from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.state import AgentState, append_log, make_log_line

# ── SerpApi Google Lens ──────────────────────────────────────────────
SERPAPI_URL = "https://serpapi.com/search"
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID", "")
# HTTP 代理（国内访问 Imgur/SerpAPI 需要）
_HTTP_PROXY = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY") or ""
_SERPAPI_PROXIES = {"http": _HTTP_PROXY, "https": _HTTP_PROXY} if _HTTP_PROXY else None

# ── mitmproxy captured results ───────────────────────────────────────
MITMPROXY_CAPTURED_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "output", "_mitmproxy_captured.json",
)
UNIFIED_OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "output", "unified_results.json",
)


DEFAULT_ENGINES = (
    "baidu",
    "serpapi_lens",
    "mitmproxy",
    "tineye",
    "yandex",
    "bing",
    "google",
    "saucenao",
    "ascii2d",
)
DEFAULT_ENGINE_LIMITS = {
    "baidu": 9999,
    "serpapi_lens": 9999,
    "tineye": 9999,
    "yandex": 9999,
    "bing": 9999,
    "google": 9999,
    "saucenao": 9999,
    "ascii2d": 9999,
    "iqdb": 9999,
}
ENGINE_ATTRS = {
    "baidu": ("Baidu", "BaiDu", "BaiduImage", "BaiDuImage"),
    "tineye": ("Tineye", "TinEye", "TineyeSearch", "TinEyeSearch"),
    "yandex": ("Yandex",),
    "bing": ("Bing",),
    "google": ("Google",),
    "saucenao": ("SauceNAO",),
    "ascii2d": ("Ascii2D",),
    "iqdb": ("IQDB", "Iqdb"),
}


def _env_list(name: str, default: Iterable[str]) -> List[str]:
    raw_value = os.getenv(name, "")
    if not raw_value.strip():
        return list(default)
    return [item.strip().lower() for item in raw_value.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_engine_limits(name: str = "RETRIEVER_ENGINE_LIMITS") -> Dict[str, int]:
    limits = dict(DEFAULT_ENGINE_LIMITS)
    raw_value = os.getenv(name, "")
    if not raw_value.strip():
        return limits

    for item in re.split(r"[,;]", raw_value):
        if not item.strip() or ":" not in item:
            continue
        engine_name, limit_text = item.split(":", 1)
        try:
            limits[engine_name.strip().lower()] = max(0, int(limit_text.strip()))
        except ValueError:
            continue
    return limits


def _first_attr(item: Any, names: Iterable[str]) -> str:
    for name in names:
        if isinstance(item, dict):
            value = item.get(name)
        else:
            value = getattr(item, name, None)
        if value:
            return str(value)
    return ""


def _result_items(response: Any) -> List[Any]:
    for attr_name in ("raw", "items", "results", "result", "data", "pages", "images", "posts", "entries", "matches"):
        value = getattr(response, attr_name, None)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested_items = _result_items(value)
            if nested_items:
                return nested_items
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        for key in ("raw", "items", "results", "result", "data", "pages", "images", "posts", "entries", "matches", "docs", "hits"):
            value = response.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                nested_items = _result_items(value)
                if nested_items:
                    return nested_items
    return []


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc
    except ValueError:
        return ""


def _upload_to_imgur(file_path: str) -> str:
    """Upload image to Imgur anonymously, return public URL."""
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(
        "https://api.imgur.com/3/image",
        headers={"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"},
        data={"image": b64},
        timeout=30,
        proxies=_SERPAPI_PROXIES,
    )
    data = resp.json()
    if data.get("success"):
        return data["data"]["link"]
    raise RuntimeError(f"Imgur upload failed: {data.get('data', {}).get('error', resp.text[:100])}")


def _serpapi_lens_search(image_url: str) -> Dict[str, Any]:
    """Search via SerpApi Google Lens. Returns {nodes, raw_count, normalized_count}."""
    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": "google_lens",
        "url": image_url,
    }
    resp = requests.get(SERPAPI_URL, params=params, timeout=120,
                       proxies=_SERPAPI_PROXIES)
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"SerpApi Google Lens: {data['error']}")

    items = data.get("visual_matches") or []
    raw_count = len(items)
    nodes: List[Dict[str, Any]] = []
    seen: set = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        page_url = item.get("link") or ""
        if not page_url or page_url in seen:
            continue
        seen.add(page_url)

        title = (item.get("title") or "")[:300]
        source = item.get("source") or ""
        image_url_match = item.get("image") or ""
        thumbnail = item.get("thumbnail") or ""

        node: Dict[str, Any] = {
            "title": title,
            "url": page_url,
            "page_url": page_url,
            "image_url": image_url_match,
            "source_type": "reverse_image_search_result",
            "engine": "serpapi_lens",
            "source": source,
            "author": "",
            "published_at": "",
        }
        if thumbnail:
            node["thumbnail_url"] = thumbnail

        nodes.append(node)

    return {
        "nodes": nodes,
        "raw_count": raw_count,
        "normalized_count": len(nodes),
    }


def _mitmproxy_load_results(image_path: Optional[str] = None) -> Dict[str, Any]:
    """Load mitmproxy captured results.  If *image_path* is given, run the
    full ADB automation flow (push → search → capture).  Otherwise just read
    the shared JSON file from a previous manual capture.
    """
    auto_nodes: List[Dict[str, Any]] = []
    auto_errors: List[str] = []
    if image_path:
        try:
            from tools.mitmproxy.mitmproxy_auto import run_auto_search
            auto_result = run_auto_search(image_path)
            auto_nodes = auto_result.get("nodes", []) if isinstance(auto_result, dict) else []
            auto_errors = auto_result.get("errors", []) if isinstance(auto_result, dict) else auto_errors
        except ImportError as exc:
            auto_errors.append(f"mitmproxy automation import failed: {exc}")
        except Exception as exc:
            auto_errors.append(f"mitmproxy automation failed: {exc}")

        # 自动化有结果就直接返回
        if auto_nodes:
            return {
                "nodes": auto_nodes,
                "raw_count": len(auto_nodes),
                "normalized_count": len(auto_nodes),
                "source": "automation",
            }

        # 自动化失败或无结果 → fallback 到本地缓存文件
        # 继续走下面的文件读取逻辑（auto_errors 会透传到返回值）

    output_path = Path(MITMPROXY_CAPTURED_FILE).resolve()
    if not output_path.exists():
        return {"nodes": [], "raw_count": 0, "normalized_count": 0,
                "errors": auto_errors if auto_errors else ["mitmproxy: no cached capture file found"]}

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"nodes": [], "raw_count": 0, "normalized_count": 0,
                "errors": auto_errors + [f"mitmproxy: failed to read cached file"] if auto_errors
                else ["mitmproxy: failed to read cached file"]}

    results = data.get("results", []) if isinstance(data, dict) else []
    raw_count = len(results)
    # Results are already normalized by the addon; strip legacy fields
    _LEGACY_FIELDS = {"stage", "domain", "sub_engine", "retrieved_rank", "engine_rank", "variant"}
    nodes: List[Dict[str, Any]] = []
    seen: set = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        key = item.get("page_url") or item.get("url") or ""
        if key in seen:
            continue
        seen.add(key)
        for f in _LEGACY_FIELDS:
            item.pop(f, None)
        nodes.append(item)

    fallback_info = {}
    if auto_errors:
        fallback_info = {"errors": auto_errors, "source": "cached_fallback"}

    return {
        "nodes": nodes,
        "raw_count": raw_count,
        "normalized_count": len(nodes),
        **fallback_info,
    }


def _extract_date_hint(*values: str) -> Optional[str]:
    text = " ".join(value for value in values if value)
    patterns = (
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?",
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"\d{4}年\d{1,2}月\d{1,2}日(?:\s*\d{1,2}:\d{2})?",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def _normalize_result(item: Any, engine_name: str) -> Optional[Dict[str, Any]]:
    page_url = _first_attr(
        item,
        (
            "url",
            "origin",
            "source",
            "source_url",
            "page_url",
            "link",
            "content",
            "href",
        ),
    )
    image_url = _first_attr(
        item,
        (
            "image",
            "image_url",
            "img_url",
            "original",
            "thumbnail",
            "thumbnail_url",
            "thumb",
            "pic_url",
            "src",
        ),
    )
    title = _first_attr(item, ("title", "text", "description", "desc", "name", "caption"))
    snippet = _first_attr(item, ("snippet", "summary", "content", "text", "description", "desc"))
    date_hint = _first_attr(item, ("date", "datetime", "time", "timestamp", "published_at"))
    published_at = date_hint or _extract_date_hint(title, snippet, page_url, image_url)

    # More lenient: allow nodes without URL if they have a meaningful title
    node_url = page_url or image_url or ""
    if not node_url and not title:
        return None

    source = _first_attr(item, ("source", "source_name", "site_name", "website"))
    if not source and (page_url or image_url):
        source = _domain_from_url(page_url or image_url)
    author = _first_attr(item, ("author", "uploader", "creator", "publisher", "user"))

    node: Dict[str, Any] = {
        "title": title,
        "url": node_url,
        "source_type": "reverse_image_search_result",
        "engine": engine_name,
        "source": source,
        "author": author,
        "published_at": published_at or "",
    }
    if snippet:
        node["snippet"] = snippet
    if page_url:
        node["page_url"] = page_url
    if image_url:
        node["image_url"] = image_url
    return node


def _load_picimagesearch_classes() -> tuple[Any, Dict[str, Any]]:
    try:
        import PicImageSearch as pic_image_search  # type: ignore
        from PicImageSearch import Network  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PicImageSearch is not installed. Run: pip install PicImageSearch") from exc

    engines: Dict[str, Any] = {}
    for engine_name, attr_names in ENGINE_ATTRS.items():
        for attr_name in attr_names:
            engine_class = getattr(pic_image_search, attr_name, None)
            if engine_class is not None:
                engines[engine_name] = engine_class
                break
    return Network, engines


async def _search_engine(
    network_class: Any,
    engine_class: Any,
    engine_name: str,
    image_path: str,
) -> Dict[str, Any]:
    """Search one engine and return both normalized nodes and diagnostics."""
    async with network_class() as client:
        engine_kwargs: Dict[str, Any] = {}
        # yandex.com image search is taken down; yandex.ru still works
        if engine_name == "yandex":
            engine_kwargs["base_url"] = "https://yandex.ru"
        engine = engine_class(client=client, **engine_kwargs)
        response = await engine.search(file=image_path)

    raw_items = _result_items(response)
    raw_count = len(raw_items)
    nodes: List[Dict[str, Any]] = []
    for item in raw_items:
        node = _normalize_result(item, engine_name)
        if node is None:
            continue
        nodes.append(node)
    return {
        "nodes": nodes,
        "raw_count": raw_count,
        "normalized_count": len(nodes),
    }


def _make_variants(image_path: str) -> List[Tuple[str, str]]:
    """Generate 5 image variants (original + 4 transforms). Returns [(label, filepath), ...]."""
    from PIL import Image, ImageEnhance

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    variants: List[Tuple[str, str]] = [("原图", image_path)]

    def _save(pil_img, label):
        fd, fp = tempfile.mkstemp(suffix=".jpg", prefix="v_")
        os.close(fd)
        pil_img.save(fp, "JPEG", quality=92)
        variants.append((label, fp))

    r = 0.80
    cw, ch = int(w * r), int(h * r)
    _save(img.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2)), "中心80%")

    _save(img.rotate(5, expand=False, fillcolor=(255, 255, 255)), "旋转+5°")
    _save(img.rotate(-5, expand=False, fillcolor=(255, 255, 255)), "旋转-5°")

    _save(ImageEnhance.Contrast(img).enhance(1.3), "高对比度")

    return variants


def _save_unified_output(
    nodes: List[Dict[str, Any]],
    used_engines: List[str],
    errors: List[str],
    diagnostics: Dict[str, Dict[str, Any]],
) -> None:
    """Save all search results to a single JSON file (overwritten each run)."""
    import datetime
    output_path = Path(UNIFIED_OUTPUT_FILE).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_results": len(nodes),
        "engines_used": used_engines,
        "errors": errors,
        "per_engine_counts": {
            eng: diag.get("returned", 0) for eng, diag in diagnostics.items()
            if diag.get("status") == "success"
        },
        "results": nodes,
    }
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[retriever] unified results saved → {output_path}")


async def _search_all_engines(
    image_path: str,
    selected_engines: List[str],
    engine_limits: Dict[str, int],
) -> tuple[List[Dict[str, Any]], List[str], List[str], Dict[str, Dict[str, Any]]]:
    network_class, available_engines = _load_picimagesearch_classes()
    engine_result_groups: List[List[Dict[str, Any]]] = []
    engine_diagnostics: Dict[str, Dict[str, Any]] = {}
    used_engines: List[str] = []
    errors: List[str] = []
    engine_timeout = _env_int("RETRIEVER_ENGINE_TIMEOUT_SECONDS", 25)
    concurrency = _env_int("RETRIEVER_CONCURRENCY", 15)

    # Generate image variants
    variants = _make_variants(image_path)
    print(f"[retriever] {len(variants)} variants × {len(selected_engines)} engines "
          f"= {len(variants) * len(selected_engines)} searches, concurrency={concurrency}")

    sem = asyncio.Semaphore(concurrency)

    # Build all search tasks: variant × engine (PicImageSearch engines)
    pis_engines = [e for e in selected_engines if e != "serpapi_lens" and e in available_engines]
    search_tasks: List[Dict[str, Any]] = []
    for v_label, v_path in variants:
        for eng in pis_engines:
            search_tasks.append({
                "variant_label": v_label,
                "variant_path": v_path,
                "engine_name": eng,
            })

    async def _run_one(task_info: dict) -> Dict[str, Any]:
        """Run a single engine search with semaphore-based concurrency."""
        eng = task_info["engine_name"]
        v_path = task_info["variant_path"]
        v_label = task_info["variant_label"]
        async with sem:
            try:
                result = await asyncio.wait_for(
                    _search_engine(
                        network_class=network_class,
                        engine_class=available_engines[eng],
                        engine_name=eng,
                        image_path=v_path,
                    ),
                    timeout=engine_timeout,
                )
                return {"status": "ok", "engine": eng, "variant": v_label, "result": result}
            except asyncio.TimeoutError:
                return {"status": "timeout", "engine": eng, "variant": v_label}
            except Exception as exc:
                return {"status": "error", "engine": eng, "variant": v_label, "error": str(exc)}

    # Launch all PicImageSearch tasks concurrently
    pis_results = await asyncio.gather(*(_run_one(t) for t in search_tasks), return_exceptions=True)

    # Collect PicImageSearch results — dedup within each engine first
    engine_hits: Dict[str, List[Dict[str, Any]]] = {}
    engine_raw_counts: Dict[str, int] = {}
    for r in pis_results:
        if isinstance(r, Exception):
            continue
        eng = r["engine"]
        if r["status"] == "ok":
            nodes = r["result"]["nodes"]
            if nodes:
                engine_hits.setdefault(eng, []).extend(nodes)
                engine_raw_counts[eng] = engine_raw_counts.get(eng, 0) + r["result"]["raw_count"]
        elif r["status"] == "timeout":
            errors.append(f"{eng}[{r['variant']}]: timeout")
        elif r["status"] == "error":
            errors.append(f"{eng}[{r['variant']}]: {r['error']}")

    for eng in pis_engines:
        raw_nodes = engine_hits.get(eng, [])
        raw_count = engine_raw_counts.get(eng, 0)
        # Per-engine dedup by page_url — variants of the same image yield many
        # identical results; only count each unique page once per engine
        seen: Set[str] = set()
        deduped_nodes: List[Dict[str, Any]] = []
        for node in raw_nodes:
            key = node.get("page_url") or node.get("url") or ""
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped_nodes.append(node)
        engine_diagnostics[eng] = {
            "status": "success" if deduped_nodes else "empty",
            "raw_count": raw_count,
            "normalized_count": len(raw_nodes),
            "returned": len(deduped_nodes),
        }
        if deduped_nodes:
            used_engines.append(eng)
            engine_result_groups.append(deduped_nodes)

    # Cleanup variant temp files
    for v_label, v_path in variants:
        if v_label != "原图" and Path(v_path).exists():
            try:
                os.unlink(v_path)
            except Exception:
                pass

    # Handle SerpApi Google Lens separately (variant-based search doesn't help Lens)
    if "serpapi_lens" in selected_engines:
        try:
            image_url = _upload_to_imgur(image_path)
            lens_result = await asyncio.to_thread(_serpapi_lens_search, image_url)
            lens_nodes = lens_result["nodes"]
            engine_diagnostics["serpapi_lens"] = {
                "status": "success" if lens_nodes else "empty",
                "raw_count": lens_result["raw_count"],
                "normalized_count": lens_result["normalized_count"],
                "returned": len(lens_nodes),
            }
            if lens_nodes:
                used_engines.append("serpapi_lens")
                engine_result_groups.append(lens_nodes)
        except Exception as exc:
            errors.append(f"serpapi_lens: {exc}")
            engine_diagnostics["serpapi_lens"] = {"status": "error", "error": str(exc)}

    # Handle mitmproxy captured results (from 小红书/微博 mobile apps)
    if "mitmproxy" in selected_engines:
        try:
            mitm_result = await asyncio.to_thread(_mitmproxy_load_results, image_path)
            mitm_nodes = mitm_result["nodes"]
            engine_diagnostics["mitmproxy"] = {
                "status": "success" if mitm_nodes else "empty",
                "raw_count": mitm_result["raw_count"],
                "normalized_count": mitm_result["normalized_count"],
                "returned": len(mitm_nodes),
            }
            if mitm_nodes:
                used_engines.append("mitmproxy")
                engine_result_groups.append(mitm_nodes)
        except Exception as exc:
            errors.append(f"mitmproxy: {exc}")
            engine_diagnostics["mitmproxy"] = {"status": "error", "error": str(exc)}

    # Concatenate all results from all engines
    nodes: List[Dict[str, Any]] = []
    for group in engine_result_groups:
        nodes.extend(group)

    # Deduplicate by page_url across all engines
    seen_urls: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for node in nodes:
        key = node.get("page_url") or node.get("url") or ""
        if key and key in seen_urls:
            continue
        if key:
            seen_urls.add(key)
        deduped.append(node)
    nodes = deduped

    for index, node in enumerate(nodes, start=1):
        nid = f"r{index}"
        # Move id to the first key
        node_items = list(node.items())
        node.clear()
        node["id"] = nid
        node.update(node_items)

    _save_unified_output(nodes, used_engines, errors, engine_diagnostics)
    return nodes, used_engines, errors, engine_diagnostics


def _run_single_search(
    image_path: str,
    engine_name: str,
    engine_class: Any,
    network_class: Any,
    engine_limit: int,
    engine_timeout: int,
) -> Dict[str, Any]:
    """Run a single-engine search synchronously (wraps async). Supports serpapi_lens."""
    # SerpApi Google Lens uses HTTP, not PicImageSearch
    if engine_name == "serpapi_lens":
        try:
            image_url = _upload_to_imgur(image_path)
            return _serpapi_lens_search(image_url)
        except Exception as exc:
            raise RuntimeError(f"serpapi_lens: {exc}") from exc

    # mitmproxy: reads from shared captured JSON file, or runs automation
    if engine_name == "mitmproxy":
        return _mitmproxy_load_results(image_path)

    coroutine = asyncio.wait_for(
        _search_engine(
            network_class=network_class,
            engine_class=engine_class,
            engine_name=engine_name,
            image_path=image_path,
            max_results=engine_limit,
        ),
        timeout=engine_timeout,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # Running inside an event loop — delegate to background thread.
    result: Dict[str, Any] | None = None
    error: BaseException | None = None

    def _runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coroutine)
        except BaseException as exc:
            error = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    if result is None:
        raise RuntimeError(f"{engine_name}: search did not return a result")
    return result


def _update_summary_error(state: AgentState, engine_name: str, error: str) -> Dict[str, Any]:
    prev = state.get("retrieval_summary", {})
    diag = dict(prev.get("engine_diagnostics", {}))
    diag[engine_name] = {"status": "error", "error": error}
    errors = list(prev.get("errors", []))
    errors.append(f"{engine_name}: {error}")
    return {
        **prev,
        "engine_diagnostics": diag,
        "errors": errors,
    }


def _finalize_engine_node(
    state: AgentState,
    logs: List[str],
    engine_name: str,
    label: str,
    engine_nodes: List[Dict[str, Any]],
    result: Dict[str, Any],
    engine_limit: int,
    engine_timeout: int,
) -> AgentState:
    """Shared result-merging logic used by both PicImageSearch and SerpApi engine nodes."""
    existing_nodes: List[Dict[str, Any]] = list(state.get("nodes_data", []))
    start_index = len(existing_nodes)

    for i, node in enumerate(engine_nodes, start=1):
        nid = f"r{start_index + i}"
        node_items = list(node.items())
        node.clear()
        node["id"] = nid
        node.update(node_items)

    all_nodes = [*existing_nodes, *engine_nodes]

    # Deduplicate by page_url (or url fallback), keep first occurrence
    seen_urls: set = set()
    deduped: List[Dict[str, Any]] = []
    for node in all_nodes:
        key = node.get("page_url") or node.get("url") or ""
        if key and key in seen_urls:
            continue
        if key:
            seen_urls.add(key)
        deduped.append(node)
    all_nodes = deduped

    prev_summary = state.get("retrieval_summary", {})
    per_engine = dict(prev_summary.get("per_engine_counts", {}))
    per_engine[engine_name] = len(engine_nodes)

    engines_used = list(prev_summary.get("engines_used", []))
    if engine_nodes:
        engines_used.append(engine_name)

    engines_requested = list(prev_summary.get("engines_requested", state.get("search_engines", [])))

    diag = dict(prev_summary.get("engine_diagnostics", {}))
    diag[engine_name] = {
        "status": "success" if engine_nodes else "empty",
        "raw_count": result["raw_count"],
        "normalized_count": result["normalized_count"],
        "returned": len(engine_nodes),
    }

    errors = list(prev_summary.get("errors", []))

    engine_limits_for_summary = dict(prev_summary.get("engine_limits", {}))
    engine_limits_for_summary[engine_name] = engine_limit

    summary_log = make_log_line(
        f"retrieve_{engine_name}: {label}搜索 → {len(engine_nodes)} 条 "
        f"(raw={result['raw_count']}, norm={result['normalized_count']})."
    )
    print(summary_log)

    return {
        "nodes_data": all_nodes,
        "retrieved_nodes": all_nodes,
        "search_engines": engines_used or state.get("search_engines", []),
        "retrieval_summary": {
            **prev_summary,
            "status": "success" if all_nodes else prev_summary.get("status", "empty"),
            "result_count": len(all_nodes),
            "per_engine_counts": per_engine,
            "engines_requested": engines_requested,
            "engines_used": engines_used,
            "engine_diagnostics": diag,
            "engine_limits": engine_limits_for_summary,
            "engine_timeout_seconds": engine_timeout,
            "errors": errors,
        },
        "execution_logs": [*logs, summary_log],
    }


def make_retrieve_engine_node(engine_name: str):
    """Create a LangGraph node function that searches a single engine.

    Each engine becomes its own graph node so that streaming yields
    intermediate results engine-by-engine, giving the user progressive
    visibility into the retrieval phase.
    """
    engine_cn = {
        "baidu": "百度", "tineye": "TinEye",
        "yandex": "Yandex", "bing": "Bing", "google": "Google",
        "saucenao": "SauceNAO", "ascii2d": "Ascii2D", "iqdb": "IQDB",
        "serpapi_lens": "Google Lens",
        "mitmproxy": "小红书/微博(抓包)",
    }

    def _retrieve_engine_node(state: AgentState) -> AgentState:
        label = engine_cn.get(engine_name, engine_name)
        logs = append_log(state, f"retrieve_{engine_name}: {label}搜索 starting.")

        target_image = state.get("target_image", {})
        image_path = str(target_image.get("local_path") or "")

        if not image_path or not Path(image_path).exists():
            error_log = make_log_line(f"retrieve_{engine_name}: image path missing, skip.")
            return {
                "execution_logs": [*logs, error_log],
                "retrieval_summary": _update_summary_error(state, engine_name, "missing image path"),
            }

        engine_limits = _env_engine_limits()
        engine_limit = engine_limits.get(engine_name, _env_int("RETRIEVER_DEFAULT_ENGINE_LIMIT", 25))
        engine_timeout = _env_int("RETRIEVER_ENGINE_TIMEOUT_SECONDS", 25)

        if engine_limit <= 0:
            skip_log = make_log_line(f"retrieve_{engine_name}: limit is 0, skip.")
            return {
                "execution_logs": [*logs, skip_log],
                "retrieval_summary": _update_summary_error(state, engine_name, "limit_zero"),
            }

        # SerpApi Google Lens: uses HTTP API, not PicImageSearch
        if engine_name == "serpapi_lens":
            try:
                image_url_lens = _upload_to_imgur(image_path)
                result = _serpapi_lens_search(image_url_lens)
            except Exception as exc:
                error_log = make_log_line(f"retrieve_{engine_name}: search error: {exc}")
                print(error_log)
                return {
                    "execution_logs": [*logs, error_log],
                    "retrieval_summary": _update_summary_error(state, engine_name, str(exc)),
                }
            engine_nodes = result["nodes"]
            # Continue with shared result-handling below
            return _finalize_engine_node(state, logs, engine_name, label, engine_nodes, result, engine_limit, engine_timeout)

        # mitmproxy: reads from shared captured JSON file, or runs automation
        if engine_name == "mitmproxy":
            try:
                result = _mitmproxy_load_results(image_path)
            except Exception as exc:
                error_log = make_log_line(f"retrieve_{engine_name}: load error: {exc}")
                print(error_log)
                return {
                    "execution_logs": [*logs, error_log],
                    "retrieval_summary": _update_summary_error(state, engine_name, str(exc)),
                }
            engine_nodes = result["nodes"]
            return _finalize_engine_node(state, logs, engine_name, label, engine_nodes, result, engine_limit, engine_timeout)

        try:
            network_class, available_engines = _load_picimagesearch_classes()
        except RuntimeError as exc:
            error_log = make_log_line(f"retrieve_{engine_name}: PicImageSearch not available: {exc}")
            return {
                "execution_logs": [*logs, error_log],
                "retrieval_summary": _update_summary_error(state, engine_name, str(exc)),
            }

        engine_class = available_engines.get(engine_name)
        if engine_class is None:
            error_log = make_log_line(f"retrieve_{engine_name}: engine not supported by PicImageSearch.")
            return {
                "execution_logs": [*logs, error_log],
                "retrieval_summary": _update_summary_error(state, engine_name, "unsupported"),
            }

        try:
            result = _run_single_search(
                image_path=image_path,
                engine_name=engine_name,
                engine_class=engine_class,
                network_class=network_class,
                engine_limit=engine_limit,
                engine_timeout=engine_timeout,
            )
        except asyncio.TimeoutError:
            error_log = make_log_line(f"retrieve_{engine_name}: timeout ({engine_timeout}s).")
            print(error_log)
            return {
                "execution_logs": [*logs, error_log],
                "retrieval_summary": _update_summary_error(state, engine_name, f"timeout ({engine_timeout}s)"),
            }
        except Exception as exc:
            error_log = make_log_line(f"retrieve_{engine_name}: search error: {exc}")
            print(error_log)
            return {
                "execution_logs": [*logs, error_log],
                "retrieval_summary": _update_summary_error(state, engine_name, str(exc)),
            }

        engine_nodes = result["nodes"]
        return _finalize_engine_node(state, logs, engine_name, label, engine_nodes, result, engine_limit, engine_timeout)

    return _retrieve_engine_node


def _run_async_search(
    image_path: str,
    selected_engines: List[str],
    engine_limits: Dict[str, int],
) -> tuple[List[Dict[str, Any]], List[str], List[str], Dict[str, Dict[str, Any]]]:
    coroutine = _search_all_engines(
        image_path=image_path,
        selected_engines=selected_engines,
        engine_limits=engine_limits,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result: tuple[List[Dict[str, Any]], List[str], List[str], Dict[str, Dict[str, Any]]] | None = None
    error: BaseException | None = None

    def runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coroutine)
        except BaseException as exc:  # pragma: no cover - depends on host event loop.
            error = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    if result is None:
        return [], [], ["async search did not return a result"], {}
    return result


def retrieve_node(state: AgentState, progress_callback=None) -> AgentState:
    logs = append_log(state, "retrieve_node: starting real reverse image search.")
    target_image = state.get("target_image", {})
    image_path = str(target_image.get("local_path") or "")
    selected_engines = state.get("search_engines") or _env_list("SEARCH_ENGINE", DEFAULT_ENGINES)
    engine_limits = _env_engine_limits()

    if not image_path or not Path(image_path).exists():
        error_log = make_log_line("retrieve_node: target image local_path is missing or does not exist.")
        print(error_log)
        return {
            "nodes_data": [],
            "search_engines": selected_engines,
            "retrieval_summary": {
                "status": "failed",
                "reason": "missing target_image.local_path",
                "engines_requested": selected_engines,
            },
            "execution_logs": [*logs, error_log],
        }

    try:
        nodes, used_engines, errors, engine_diagnostics = _run_async_search(
            image_path=image_path,
            selected_engines=selected_engines,
            engine_limits=engine_limits,
        )
    except RuntimeError as exc:
        error_log = make_log_line(f"retrieve_node: reverse image search failed: {exc}")
        print(error_log)
        return {
            "nodes_data": [],
            "search_engines": selected_engines,
            "retrieval_summary": {
                "status": "failed",
                "reason": str(exc),
                "engines_requested": selected_engines,
            },
            "execution_logs": [*logs, error_log],
        }

    per_engine_counts = {
        engine_name: sum(1 for n in nodes if n.get("engine") == engine_name)
        for engine_name in used_engines
    }
    if progress_callback:
        for eng_name, eng_count in per_engine_counts.items():
            progress_callback("engine_done", eng_name, eng_count, len(selected_engines), f"{eng_name}: {eng_count} 结果")
        # 报告失败的引擎
        for eng_name in selected_engines:
            if eng_name not in used_engines and eng_name not in per_engine_counts:
                progress_callback("engine_fail", eng_name, 0, len(selected_engines), f"{eng_name}: 超时/失败")
    summary_log = make_log_line(
        f"retrieve_node: collected {len(nodes)} result(s) from "
        f"{', '.join(used_engines) or 'no engines'}. "
        f"per-engine: {per_engine_counts}"
    )
    print(summary_log)
    if errors:
        errors_log = make_log_line(f"retrieve_node: engine warnings: {'; '.join(errors)}")
        print(errors_log)
        logs = [*logs, summary_log, errors_log]
    else:
        logs = [*logs, summary_log]

    return {
        "nodes_data": nodes,
        "retrieved_nodes": nodes,
        "search_engines": used_engines or selected_engines,
        "retrieval_summary": {
            "status": "success" if nodes else "empty",
            "result_count": len(nodes),
            "per_engine_counts": per_engine_counts,
            "engine_limits": {engine: engine_limits.get(engine, 0) for engine in selected_engines},
            "engine_timeout_seconds": _env_int("RETRIEVER_ENGINE_TIMEOUT_SECONDS", 25),
            "engines_requested": selected_engines,
            "engines_used": used_engines,
            "engine_diagnostics": engine_diagnostics,
            "errors": errors,
        },
        "execution_logs": logs,
    }


if __name__ == "__main__":
    mock_state: AgentState = {
        "target_image": {
            "filename": "demo.jpg",
            "content_type": "image/jpeg",
            "size_bytes": 1024,
            "local_path": "demo.jpg",
        },
        "nodes_data": [],
        "execution_logs": [],
    }

    result = retrieve_node(mock_state)
    print(json.dumps(result, ensure_ascii=False, indent=2))
