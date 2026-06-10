from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from PIL import Image, ImageChops, ImageOps, ImageStat, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 允许开发者直接运行 python agents/validator.py 进行本地调试。
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.state import AgentState, SIMILARITY_THRESHOLD, append_log, make_log_line

try:
    import imagehash
except ImportError:  # pragma: no cover - 便于缺依赖时给出结构化原因。
    imagehash = None


REQUEST_TIMEOUT_SECONDS = 12
REQUEST_RETRY_TOTAL = 3
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASH_MAX_DISTANCE = 64
RESIZED_IMAGE_SIZE = (256, 256)
LLM_IMAGE_MAX_SIZE = (768, 768)
HASH_STRONG_PASS_THRESHOLD = 0.92
HASH_WEAK_PASS_THRESHOLD = 0.72
CLIP_REVIEW_THRESHOLD = 0.78
LLM_BOUNDARY_SIMILARITY_FLOOR = 0.75
JOINT_DEDUP_THRESHOLD = 0.86
JOINT_DEDUP_TEXT_WEIGHT = 0.90
JOINT_DEDUP_IMAGE_WEIGHT = 0.10
SAME_IMAGE_TEXT_DIFFERENT_THRESHOLD = 0.25
CLIP_MODEL_NAME = os.getenv("VALIDATOR_CLIP_MODEL", "openai/clip-vit-base-patch32")
DEFAULT_LOCAL_CLIP_MODEL_PATH = "models/clip-vit-base-patch32"
VALIDATOR_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "validator_system_prompt.md"
PLACEHOLDER_MARKERS = ("xxxx", "your-", "replace-", "sk-xxxx", "fc-xxxx")
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "yclid", "spm", "from", "share_source"}
TEXT_CONTEXT_FIELDS = (
    "title",
    "snippet",
    "summary",
    "description",
    "desc",
    "text",
    "content",
    "caption",
    "alt",
    "ocr_text",
)
WATERMARK_PLATFORM_ALIASES = {
    "weibo": ("微博", "新浪微博", "微博号", "微博id", "weibo", " wb "),
    "xiaohongshu": (
        "小红书",
        "小紅書",
        "小红薯",
        "小紅薯",
        "小红书号",
        "小红书id",
        "xhs",
        "red id",
        "rednote",
        "red note",
        "xiaohongshu",
    ),
    "douyin": ("抖音", "抖音号", "抖音id", "douyin", "tiktok"),
    "kuaishou": ("快手", "快手号", "快手id", "kuaishou"),
    "bilibili": ("哔哩哔哩", "bilibili", "b站", "uid"),
}
ACCOUNT_CHARS = r"[\w.\-\u4e00-\u9fff]{2,40}"
ASCII_ACCOUNT_CHARS = r"[A-Za-z0-9_.-]{3,40}"

BAIDU_OCR_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
BAIDU_OCR_API_URL = os.getenv(
    "BAIDU_OCR_API_URL",
    "https://aip.baidubce.com/rest/2.0/ocr/v1/general",
)
BAIDU_OCR_TOKEN_CACHE: Dict[str, Any] = {"token": "", "expires_at": 0.0}
RAPID_OCR_RUNTIME: Dict[str, Any] = {}
PADDLE_OCR_VL_RUNTIME: Dict[str, Any] = {}
CLIP_IMAGE_FEATURE_CACHE: Dict[str, Any] = {}
CLIP_TEXT_FEATURE_CACHE: Dict[str, Any] = {}
CLIP_RUNTIME_FAILURE: Dict[str, str] = {"reason": ""}


def env_flag(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if minimum is not None:
        return max(value, minimum)
    return value


def env_list(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip().lower() for item in re.split(r"[,|]", raw) if item.strip()]


def normalize_ocr_provider_name(provider: str) -> str:
    aliases = {
        "rapid": "rapidocr",
        "rapid_ocr": "rapidocr",
        "rapid-ocr": "rapidocr",
        "rapidocr_onnxruntime": "rapidocr",
        "paddle": "paddleocr_vl",
        "paddleocr": "paddleocr_vl",
        "paddleocr-vl": "paddleocr_vl",
        "paddle_ocr_vl": "paddleocr_vl",
        "baidu_ocr": "baidu",
    }
    return aliases.get(provider.strip().lower(), provider.strip().lower())


def cache_root() -> Path:
    """Validator 本地缓存根目录。"""
    configured = Path(os.getenv("VALIDATOR_CACHE_DIR", "data/cache/validator"))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def cache_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cache_file(cache_name: str, key: str, suffix: str) -> Path:
    return cache_root() / cache_name / f"{key}{suffix}"


def similarity_threshold() -> float:
    return min(max(env_float("SIMILARITY_THRESHOLD", SIMILARITY_THRESHOLD), 0.0), 1.0)


def bounded_threshold(name: str, default: float) -> float:
    return min(max(env_float(name, default), 0.0), 1.0)


def hash_strong_threshold() -> float:
    return bounded_threshold("VALIDATOR_HASH_STRONG_PASS_THRESHOLD", HASH_STRONG_PASS_THRESHOLD)


def hash_weak_threshold() -> float:
    return bounded_threshold("VALIDATOR_HASH_WEAK_PASS_THRESHOLD", HASH_WEAK_PASS_THRESHOLD)


def clip_review_threshold() -> float:
    return bounded_threshold("VALIDATOR_CLIP_REVIEW_THRESHOLD", CLIP_REVIEW_THRESHOLD)


def llm_boundary_similarity_floor() -> float:
    return bounded_threshold("VALIDATOR_LLM_BOUNDARY_SIMILARITY_FLOOR", LLM_BOUNDARY_SIMILARITY_FLOOR)


def joint_dedup_threshold() -> float:
    return bounded_threshold("VALIDATOR_JOINT_DEDUP_THRESHOLD", JOINT_DEDUP_THRESHOLD)


def same_image_text_different_threshold() -> float:
    return bounded_threshold(
        "VALIDATOR_SAME_IMAGE_TEXT_DIFFERENT_THRESHOLD",
        SAME_IMAGE_TEXT_DIFFERENT_THRESHOLD,
    )


def ocr_prefilter_enabled() -> bool:
    return env_flag("VALIDATOR_ENABLE_OCR_PREFILTER", True)


def ocr_prefilter_visual_threshold() -> float:
    return bounded_threshold("VALIDATOR_OCR_PREFILTER_VISUAL_THRESHOLD", 0.62)


def ocr_prefilter_text_threshold() -> float:
    return bounded_threshold("VALIDATOR_OCR_PREFILTER_TEXT_THRESHOLD", 0.12)


def tampering_text_signal_threshold() -> float:
    return bounded_threshold("VALIDATOR_TAMPERING_TEXT_SIGNAL_THRESHOLD", 0.35)


def text_only_min_visual_threshold() -> float:
    return bounded_threshold("VALIDATOR_TEXT_ONLY_MIN_VISUAL_THRESHOLD", 0.65)


def looks_like_placeholder(value: str) -> bool:
    normalized = (value or "").strip().lower()
    return not normalized or any(marker in normalized for marker in PLACEHOLDER_MARKERS)


@lru_cache(maxsize=1)
def load_validator_system_prompt() -> str:
    """读取视觉校验智能体 System Prompt，便于后续接入 LLM Agent。"""
    # 后续接入多模态 LLM 时，在 core/llm_client.py 中统一读取 API Key，
    # 并把本 prompt、validation_signals、目标图/候选图一起发送给模型做二次判定。
    return VALIDATOR_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def get_retry_session() -> requests.Session:
    """构造带重试的 HTTP 会话，用于候选图下载和 OCR 请求。"""
    retry = Retry(
        total=REQUEST_RETRY_TOTAL,
        connect=REQUEST_RETRY_TOTAL,
        read=REQUEST_RETRY_TOTAL,
        status=REQUEST_RETRY_TOTAL,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        }
    )
    return session


def normalize_image(image: Image.Image) -> Image.Image:
    """统一图片方向和色彩模式，降低 EXIF 与透明通道对相似度的影响。"""
    return ImageOps.exif_transpose(image).convert("RGB")


def load_image_from_path(path: str) -> Image.Image:
    """从本地路径读取图片。"""
    with Image.open(path) as image:
        return normalize_image(image.copy())


def image_cache_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"} else ".img"


def image_request_headers(url: str) -> Dict[str, str]:
    host = urlparse(url).netloc.lower()
    if "xhscdn.com" in host or "xiaohongshu.com" in host:
        return {"Referer": "https://www.xiaohongshu.com/"}
    if "sinaimg.cn" in host or "weibo.cn" in host or "weibo.com" in host:
        return {"Referer": "https://m.weibo.cn/"}
    return {}


def image_download_variants(url: str) -> List[str]:
    variants = [url]
    host = urlparse(url).netloc.lower()
    if "xhscdn.com" in host or "xiaohongshu.com" in host:
        webp_url = url.replace("format/heif", "format/webp").replace("format/heic", "format/webp")
        if webp_url != url:
            variants.insert(0, webp_url)
    return list(dict.fromkeys(variants))


def open_image_bytes(content: bytes) -> Image.Image:
    with Image.open(io.BytesIO(content)) as image:
        return normalize_image(image.copy())


def download_image_bytes(url: str) -> bytes:
    response = get_retry_session().get(
        url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers=image_request_headers(url),
    )
    response.raise_for_status()
    return response.content


def read_or_download_image_bytes(url: str, cache_path: Path, image_cache_enabled: bool) -> bytes:
    if image_cache_enabled and cache_path.exists():
        return cache_path.read_bytes()
    return download_image_bytes(url)


def load_image_from_url(url: str) -> Image.Image:
    """从候选图片 URL 下载并读取图片；默认缓存到本地，便于中断后续跑。"""
    image_cache_enabled = env_flag("VALIDATOR_ENABLE_IMAGE_CACHE", True)
    last_error: Optional[Exception] = None

    for candidate_url in image_download_variants(url):
        path = cache_file("images", cache_key(candidate_url), image_cache_suffix(candidate_url))
        try:
            content = read_or_download_image_bytes(candidate_url, path, image_cache_enabled)
            image = open_image_bytes(content)
            if image_cache_enabled and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            return image
        except (OSError, UnidentifiedImageError, requests.RequestException, ValueError) as exc:
            last_error = exc
            if image_cache_enabled and path.exists():
                try:
                    content = download_image_bytes(candidate_url)
                    image = open_image_bytes(content)
                    path.write_bytes(content)
                    return image
                except (OSError, UnidentifiedImageError, requests.RequestException, ValueError) as retry_exc:
                    last_error = retry_exc

    if last_error is not None:
        raise last_error
    raise ValueError("empty image url")


def get_target_image(state: AgentState) -> Optional[Image.Image]:
    """读取上传目标图；没有 local_path 时返回 None。"""
    local_path = state.get("target_image", {}).get("local_path")
    if not local_path:
        return None
    return load_image_from_path(local_path)


def get_candidate_image(node: Dict[str, Any], target_image: Image.Image) -> Image.Image:
    """按优先级获取候选图：本地缓存 > 图片 URL > 搜索缩略图。"""
    for path_key in ("cached_image_path", "local_image_path"):
        path = node.get(path_key)
        if path:
            return load_image_from_path(path)

    for url_key in ("image_url", "thumbnail_url"):
        url = node.get(url_key)
        if url:
            return load_image_from_url(url)

    raise ValueError("缺少可校验图片地址：需要 cached_image_path、image_url 或 thumbnail_url")


def compute_phash_similarity(target_image: Image.Image, candidate_image: Image.Image) -> float:
    """使用感知哈希快速判断两张图片是否为同图变体。"""
    if imagehash is None:
        raise RuntimeError("缺少 imagehash 依赖，请先安装 requirements.txt")

    target_hash = imagehash.phash(target_image)
    candidate_hash = imagehash.phash(candidate_image)
    distance = target_hash - candidate_hash
    return max(0.0, 1.0 - distance / PHASH_MAX_DISTANCE)


def compute_image_phash(image: Image.Image) -> str:
    """生成图片 pHash 字符串，用于候选之间的联合去重。"""
    if imagehash is None:
        raise RuntimeError("缺少 imagehash 依赖，请先安装 requirements.txt")
    return str(imagehash.phash(image))


def compute_hash_string_similarity(left_hash: str, right_hash: str) -> float:
    """根据两个 pHash 字符串的汉明距离计算候选图片相似度。"""
    if imagehash is None or not left_hash or not right_hash:
        return 0.0
    try:
        left = imagehash.hex_to_hash(left_hash)
        right = imagehash.hex_to_hash(right_hash)
    except (TypeError, ValueError):
        return 0.0
    distance = left - right
    return max(0.0, 1.0 - distance / PHASH_MAX_DISTANCE)


def resize_for_direct_compare(image: Image.Image) -> Image.Image:
    """把图片拉到同一尺寸，便于做像素级弱比较。"""
    return normalize_image(image).resize(RESIZED_IMAGE_SIZE, Image.Resampling.LANCZOS)


def compute_resized_image_similarity(target_image: Image.Image, candidate_image: Image.Image) -> float:
    """先统一尺寸，再用平均像素差衡量低层视觉相似度。"""
    target_resized = resize_for_direct_compare(target_image)
    candidate_resized = resize_for_direct_compare(candidate_image)
    diff = ImageChops.difference(target_resized, candidate_resized)
    channel_means = ImageStat.Stat(diff).mean
    mean_difference = sum(channel_means) / len(channel_means)
    return max(0.0, min(1.0, 1.0 - mean_difference / 255.0))


def compute_grayscale_similarity(target_image: Image.Image, candidate_image: Image.Image) -> float:
    """忽略颜色，仅比较灰度结构，适合识别调色、滤镜、黑白化变体。"""
    target_gray = ImageOps.grayscale(resize_for_direct_compare(target_image))
    candidate_gray = ImageOps.grayscale(resize_for_direct_compare(candidate_image))
    diff = ImageChops.difference(target_gray, candidate_gray)
    mean_difference = ImageStat.Stat(diff).mean[0]
    return max(0.0, min(1.0, 1.0 - mean_difference / 255.0))


def compact_color_histogram(image: Image.Image, bins_per_channel: int = 16) -> List[int]:
    """压缩 RGB 直方图，降低尺寸和轻微压缩差异对颜色比较的影响。"""
    resized = resize_for_direct_compare(image)
    raw_histogram = resized.histogram()
    bin_size = 256 // bins_per_channel
    compacted: List[int] = []
    for channel_index in range(3):
        channel_histogram = raw_histogram[channel_index * 256 : (channel_index + 1) * 256]
        for start in range(0, 256, bin_size):
            compacted.append(sum(channel_histogram[start : start + bin_size]))
    return compacted


def compute_color_hist_similarity(target_image: Image.Image, candidate_image: Image.Image) -> float:
    """比较压缩后的 RGB 颜色分布，作为调色/换色的弱辅助信号。"""
    target_histogram = compact_color_histogram(target_image)
    candidate_histogram = compact_color_histogram(candidate_image)
    intersection = sum(min(left, right) for left, right in zip(target_histogram, candidate_histogram))
    union = sum(max(left, right) for left, right in zip(target_histogram, candidate_histogram))
    if union <= 0:
        return 0.0
    return max(0.0, min(1.0, intersection / union))


@lru_cache(maxsize=1)
def get_clip_runtime():
    """懒加载 CLIP，避免 Streamlit 刷新时重复加载大模型。"""
    if CLIP_RUNTIME_FAILURE["reason"]:
        raise RuntimeError(CLIP_RUNTIME_FAILURE["reason"])
    import torch
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_source = os.getenv("VALIDATOR_CLIP_MODEL_PATH") or (
        DEFAULT_LOCAL_CLIP_MODEL_PATH if Path(DEFAULT_LOCAL_CLIP_MODEL_PATH).exists() else CLIP_MODEL_NAME
    )
    local_files_only = env_flag(
        "VALIDATOR_CLIP_LOCAL_ONLY",
        bool(os.getenv("VALIDATOR_CLIP_MODEL_PATH")) or Path(DEFAULT_LOCAL_CLIP_MODEL_PATH).exists(),
    )
    if local_files_only and not Path(model_source).exists():
        raise RuntimeError(
            "CLIP 本地模型不存在，请把模型放到 models/clip-vit-base-patch32，"
            "或设置 VALIDATOR_CLIP_LOCAL_ONLY=false 允许从 HuggingFace 下载。"
        )

    try:
        processor = CLIPImageProcessor.from_pretrained(
            model_source,
            local_files_only=local_files_only,
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            model_source,
            local_files_only=local_files_only,
        )
        model = CLIPModel.from_pretrained(
            model_source,
            local_files_only=local_files_only,
        ).to(device)
    except Exception as exc:
        CLIP_RUNTIME_FAILURE["reason"] = f"CLIP runtime unavailable: {exc}"
        raise RuntimeError(CLIP_RUNTIME_FAILURE["reason"]) from exc
    model.eval()
    return processor, tokenizer, model, torch, device


def clip_feature_cache_enabled() -> bool:
    return env_flag("VALIDATOR_ENABLE_CLIP_FEATURE_CACHE", True)


def validator_llm_complexity_threshold() -> float:
    return bounded_threshold("VALIDATOR_LLM_COMPLEXITY_THRESHOLD", 0.62)


def max_validator_llm_nodes() -> int:
    return env_int("VALIDATOR_MAX_LLM_NODES", 12, 0)


def watermark_ocr_enabled() -> bool:
    return env_flag("VALIDATOR_ENABLE_WATERMARK_OCR", False) and bool(watermark_ocr_provider_chain())


def watermark_ocr_visual_threshold() -> float:
    return bounded_threshold("VALIDATOR_WATERMARK_OCR_VISUAL_THRESHOLD", 0.80)


def max_watermark_ocr_nodes() -> int:
    return env_int("VALIDATOR_MAX_WATERMARK_OCR_NODES", 30, 0)


def include_debug_signals() -> bool:
    return env_flag("VALIDATOR_INCLUDE_DEBUG_SIGNALS", False)


def compute_image_complexity(image: Image.Image) -> float:
    prepared = normalize_image(image).resize((192, 192), Image.Resampling.LANCZOS)
    grayscale = ImageOps.grayscale(prepared)
    entropy_score = max(0.0, min(1.0, grayscale.entropy() / 8.0))
    color_stat = ImageStat.Stat(prepared)
    color_std = sum(color_stat.stddev[:3]) / max(1, len(color_stat.stddev[:3]))
    color_score = max(0.0, min(1.0, color_std / 90.0))
    equalized = ImageOps.equalize(grayscale)
    diff = ImageChops.difference(grayscale, equalized)
    contrast_score = max(0.0, min(1.0, ImageStat.Stat(diff).mean[0] / 64.0))
    return round(entropy_score * 0.5 + color_score * 0.3 + contrast_score * 0.2, 4)


def image_content_hash(image: Image.Image) -> str:
    normalized = normalize_image(image)
    digest = hashlib.sha256()
    digest.update(normalized.mode.encode("utf-8"))
    digest.update(str(normalized.size).encode("utf-8"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def text_content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def as_clip_tensor(features: Any, torch: Any) -> Any:
    if torch.is_tensor(features):
        return features
    for attr_name in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(features, attr_name, None)
        if torch.is_tensor(value):
            return value
    if isinstance(features, (tuple, list)) and features:
        return as_clip_tensor(features[0], torch)
    last_hidden_state = getattr(features, "last_hidden_state", None)
    if torch.is_tensor(last_hidden_state):
        return last_hidden_state[:, 0]
    raise TypeError(f"unsupported CLIP feature output: {type(features).__name__}")


def normalize_clip_feature(feature: Any, torch: Any) -> Any:
    return torch.nn.functional.normalize(feature, p=2, dim=-1)


def get_clip_image_feature(image: Image.Image) -> Any:
    processor, _, model, torch, device = get_clip_runtime()
    cache_key = f"{id(model)}:{device}:{image_content_hash(image)}"
    if clip_feature_cache_enabled() and cache_key in CLIP_IMAGE_FEATURE_CACHE:
        return CLIP_IMAGE_FEATURE_CACHE[cache_key]

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        feature = normalize_clip_feature(as_clip_tensor(model.get_image_features(**inputs), torch), torch).detach()

    if clip_feature_cache_enabled():
        CLIP_IMAGE_FEATURE_CACHE[cache_key] = feature
    return feature


def get_clip_text_feature(text: str) -> Any:
    _, tokenizer, model, torch, device = get_clip_runtime()
    normalized_text = text.strip()
    cache_key = f"{id(model)}:{device}:{text_content_hash(normalized_text)}"
    if clip_feature_cache_enabled() and cache_key in CLIP_TEXT_FEATURE_CACHE:
        return CLIP_TEXT_FEATURE_CACHE[cache_key]

    inputs = tokenizer(
        [normalized_text],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        feature = normalize_clip_feature(as_clip_tensor(model.get_text_features(**inputs), torch), torch).detach()

    if clip_feature_cache_enabled():
        CLIP_TEXT_FEATURE_CACHE[cache_key] = feature
    return feature


def compute_clip_similarity(target_image: Image.Image, candidate_image: Image.Image) -> float:
    """使用 CLIP 图像向量计算视觉语义相似度。"""
    # 后续可把本地 CLIP 替换为 embedding API：
    # 1. 新增 VALIDATOR_EMBEDDING_API_URL 环境变量；
    # 2. 优先请求远程 embedding 服务；
    # 3. 未配置服务时再 fallback 到本地 CLIP。
    _, _, _, torch, _ = get_clip_runtime()
    target_feature = get_clip_image_feature(target_image)
    candidate_feature = get_clip_image_feature(candidate_image)
    similarity = torch.matmul(target_feature[0], candidate_feature[0]).item()

    return max(0.0, min(1.0, float(similarity)))


def compute_clip_text_similarity(left: str, right: str) -> float:
    """使用 CLIP 文本向量计算检索文本与目标图文本的语义相似度。"""
    if not left.strip() or not right.strip():
        return 0.0

    _, _, _, torch, _ = get_clip_runtime()
    left_feature = get_clip_text_feature(left)
    right_feature = get_clip_text_feature(right)
    similarity = torch.matmul(left_feature[0], right_feature[0]).item()

    return max(0.0, min(1.0, float(similarity)))


def get_baidu_ocr_access_token() -> str:
    """获取百度 OCR access_token。"""
    now = time.time()
    cached_token = BAIDU_OCR_TOKEN_CACHE.get("token")
    if cached_token and float(BAIDU_OCR_TOKEN_CACHE.get("expires_at", 0.0)) > now:
        return str(cached_token)

    api_key = os.getenv("BAIDU_OCR_API_KEY")
    secret_key = os.getenv("BAIDU_OCR_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("未配置 BAIDU_OCR_API_KEY 或 BAIDU_OCR_SECRET_KEY")

    response = get_retry_session().post(
        BAIDU_OCR_TOKEN_URL,
        params={
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": secret_key,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"百度 OCR token 获取失败：{payload}")

    expires_in = int(payload.get("expires_in") or 0)
    BAIDU_OCR_TOKEN_CACHE["token"] = token
    BAIDU_OCR_TOKEN_CACHE["expires_at"] = now + max(expires_in - 300, 60)
    return str(token)


def image_to_jpeg_bytes(image: Image.Image, quality: int = 92) -> bytes:
    """把图片稳定编码为 JPEG 字节，用于 OCR 请求和缓存 key。"""
    buffer = io.BytesIO()
    normalize_image(image).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def image_to_base64(image: Image.Image) -> str:
    """把图片转为百度 OCR 接口需要的 base64 字符串。"""
    return base64.b64encode(image_to_jpeg_bytes(image)).decode("utf-8")


def image_to_data_url(image: Image.Image) -> str:
    """把图片压缩为多模态 LLM 可接收的 data URL。"""
    prepared = normalize_image(image)
    prepared.thumbnail(LLM_IMAGE_MAX_SIZE, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    prepared.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def parse_llm_json(content: str) -> Optional[Dict[str, Any]]:
    """解析 LLM 返回的 JSON，兼容 ```json 代码块。"""
    text = (content or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


class ValidatorMultimodalLLMClient:
    """OpenAI-compatible multimodal client for Validator explanations."""

    def __init__(self) -> None:
        self.api_key = os.getenv("VALIDATOR_VISION_API_KEY", "")
        self.base_url = os.getenv(
            "VALIDATOR_VISION_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = os.getenv("VALIDATOR_VISION_MODEL", "qwen-vl-plus")
        self.timeout_seconds = env_int("VALIDATOR_MULTIMODAL_TIMEOUT_SECONDS", 30, 1)
        self.max_tokens = env_int("VALIDATOR_MULTIMODAL_MAX_TOKENS", 900, 128)
        self.client: Any = None
        self.enabled = False
        self.reason = ""

        if looks_like_placeholder(self.api_key):
            self.reason = "missing or placeholder VALIDATOR_VISION_API_KEY"
            return
        if looks_like_placeholder(self.model):
            self.reason = "missing or placeholder VALIDATOR_VISION_MODEL"
            return

        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            self.reason = "openai package is not installed"
            return

        try:
            client_kwargs: Dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout_seconds,
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**client_kwargs)
            self.enabled = True
            self.reason = "ready"
        except Exception as exc:  # pragma: no cover - depends on SDK config.
            self.reason = f"failed to initialize Validator LLM client: {exc}"

    def review(
        self,
        *,
        target_image: Image.Image,
        candidate_image: Image.Image,
        node: Dict[str, Any],
        validation_signals: Dict[str, Any],
        similarity: float,
        is_similar: bool,
        image_variant: str,
        validation_reason: str,
    ) -> Dict[str, Any]:
        if not self.enabled or self.client is None:
            return {"llm_status": "disabled", "llm_reason": self.reason}

        payload = {
            "node": {
                key: node.get(key)
                for key in (
                    "id",
                    "title",
                    "url",
                    "author",
                    "publisher",
                    "image_url",
                    "thumbnail_url",
                    "source_type",
                    "engine",
                    "retrieved_rank",
                )
            },
            "similarity": round(similarity, 4),
            "is_similar_readonly": is_similar,
            "image_variant_readonly": image_variant,
            "validation_reason_readonly": validation_reason,
            "validation_signals": compact_llm_payload(validation_signals),
        }
        user_text = (
            "请分别观察第一张目标图和第二张候选图，只输出一个 JSON 对象。"
            "不要改变 is_similar 决策；重点做复杂图片语义描述，并严格区分目标图和候选图。"
            "不要识别、提取或猜测平台水印、账号、ID；水印证据只由 OCR 模块负责。"
            "\n\nJSON schema:\n"
            "{\n"
            '  "target_scene_description": "目标图场景描述",\n'
            '  "candidate_scene_description": "候选图场景描述",\n'
            '  "specific_entities": [],\n'
            '  "location_or_landmark": "",\n'
            '  "target_main_text": [],\n'
            '  "candidate_main_text": [],\n'
            '  "logos_or_symbols": [],\n'
            '  "editing_or_montage_signals": [],\n'
            '  "semantic_comparison": "两图语义关系",\n'
            '  "reason": "一句中文解释"\n'
            "}\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": load_validator_system_prompt()},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_to_data_url(target_image)},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_to_data_url(candidate_image)},
                            },
                        ],
                    },
                ],
            )
            content = response.choices[0].message.content or ""
            parsed = parse_llm_json(content)
            if parsed is None:
                return {
                    "llm_status": "error",
                    "llm_reason": "Validator LLM returned non-JSON content",
                    "llm_raw_excerpt": compact_text(content, 300),
                }
            return {
                "llm_status": "used",
                "llm_reason": str(parsed.get("reason") or "Validator LLM review completed"),
                "semantic_caption": parsed,
            }
        except Exception as exc:
            return {"llm_status": "error", "llm_reason": str(exc)}


def ocr_provider_chain() -> List[str]:
    providers = env_list("VALIDATOR_OCR_PROVIDER", "rapidocr")
    fallback = env_list("VALIDATOR_OCR_FALLBACK_PROVIDER", "")
    normalized = []
    for provider in [*providers, *fallback]:
        provider = normalize_ocr_provider_name(provider)
        if provider in {"rapidocr", "paddleocr_vl", "baidu"} and provider not in normalized:
            normalized.append(provider)
    return normalized or ["rapidocr"]


def watermark_ocr_provider_chain() -> List[str]:
    providers = env_list("VALIDATOR_WATERMARK_OCR_PROVIDER", "")
    normalized: List[str] = []
    for provider in providers:
        provider = normalize_ocr_provider_name(provider)
        if provider in {"paddleocr_vl", "baidu"} and provider not in normalized:
            normalized.append(provider)
    return normalized


def ocr_cache_key(provider_key: str, image_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(provider_key.encode("utf-8"))
    digest.update(image_bytes)
    return digest.hexdigest()


def should_fallback_on_empty_ocr() -> bool:
    return env_flag("VALIDATOR_OCR_FALLBACK_ON_EMPTY", False)


def get_rapidocr_runtime() -> Any:
    if "engine" in RAPID_OCR_RUNTIME:
        return RAPID_OCR_RUNTIME["engine"]
    if "error" in RAPID_OCR_RUNTIME:
        raise RuntimeError(str(RAPID_OCR_RUNTIME["error"]))

    try:
        from rapidocr import RapidOCR  # type: ignore
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError as exc:
            message = "未安装 rapidocr，无法使用 RapidOCR"
            RAPID_OCR_RUNTIME["error"] = message
            raise RuntimeError(message) from exc

    kwargs: Dict[str, Any] = {}
    params_json = os.getenv("VALIDATOR_RAPIDOCR_PARAMS_JSON", "").strip()
    if params_json:
        try:
            kwargs["params"] = json.loads(params_json)
        except json.JSONDecodeError as exc:
            message = f"VALIDATOR_RAPIDOCR_PARAMS_JSON 不是合法 JSON: {exc}"
            RAPID_OCR_RUNTIME["error"] = message
            raise RuntimeError(message) from exc

    try:
        engine = RapidOCR(**kwargs)
    except TypeError:
        engine = RapidOCR()
    except Exception as exc:
        RAPID_OCR_RUNTIME["error"] = str(exc)
        raise

    RAPID_OCR_RUNTIME["engine"] = engine
    return engine


def get_paddleocr_vl_runtime() -> Any:
    if "pipeline" in PADDLE_OCR_VL_RUNTIME:
        return PADDLE_OCR_VL_RUNTIME["pipeline"]
    if "error" in PADDLE_OCR_VL_RUNTIME:
        raise RuntimeError(str(PADDLE_OCR_VL_RUNTIME["error"]))

    try:
        from paddleocr import PaddleOCRVL  # type: ignore
    except ImportError as exc:
        message = "未安装 paddleocr[doc-parser]，无法使用 PaddleOCR-VL"
        PADDLE_OCR_VL_RUNTIME["error"] = message
        raise RuntimeError(message) from exc

    kwargs: Dict[str, Any] = {
        "pipeline_version": os.getenv("VALIDATOR_PADDLEOCR_VL_VERSION", "v1.6"),
    }
    device = os.getenv("VALIDATOR_PADDLEOCR_DEVICE", "").strip()
    if device:
        kwargs["device"] = device

    try:
        pipeline = PaddleOCRVL(**kwargs)
        PADDLE_OCR_VL_RUNTIME["pipeline"] = pipeline
        return pipeline
    except Exception as exc:
        PADDLE_OCR_VL_RUNTIME["error"] = str(exc)
        raise


def box_to_location(box: Any) -> Dict[str, int]:
    if isinstance(box, dict):
        if all(key in box for key in ("left", "top", "width", "height")):
            return {
                "left": int(box.get("left", 0) or 0),
                "top": int(box.get("top", 0) or 0),
                "width": int(box.get("width", 0) or 0),
                "height": int(box.get("height", 0) or 0),
            }
        for key in ("bbox", "box", "points", "poly", "polygon", "block_bbox"):
            if key in box:
                return box_to_location(box[key])

    if isinstance(box, (list, tuple)) and box:
        if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
            left, top, right, bottom = [float(value) for value in box]
            if right < left or bottom < top:
                left, top, width, height = left, top, right, bottom
                return {
                    "left": int(left),
                    "top": int(top),
                    "width": max(0, int(width)),
                    "height": max(0, int(height)),
                }
            return {
                "left": int(left),
                "top": int(top),
                "width": max(0, int(right - left)),
                "height": max(0, int(bottom - top)),
            }

        points: List[Tuple[float, float]] = []
        for item in box:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    points.append((float(item[0]), float(item[1])))
                except (TypeError, ValueError):
                    continue
        if points:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            return {
                "left": int(left),
                "top": int(top),
                "width": max(0, int(right - left)),
                "height": max(0, int(bottom - top)),
            }

    return {"left": 0, "top": 0, "width": 0, "height": 0}


def object_to_plain_data(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [object_to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [object_to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): object_to_plain_data(item) for key, item in value.items()}

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return object_to_plain_data(tolist())
        except TypeError:
            pass

    for attr_name in ("json", "res", "data"):
        attr = getattr(value, attr_name, None)
        if attr is not None and attr is not value:
            try:
                return object_to_plain_data(attr() if callable(attr) else attr)
            except TypeError:
                return object_to_plain_data(attr)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return object_to_plain_data(to_dict())
    return str(value)


def append_ocr_block(
    blocks: List[Dict[str, Any]],
    text: Any,
    location: Any = None,
    confidence: Any = None,
    provider: str = "",
) -> None:
    normalized_text = compact_text(str(text or ""), 500).strip()
    if not normalized_text:
        return
    block: Dict[str, Any] = {
        "text": normalized_text,
        "location": box_to_location(location),
    }
    try:
        if confidence is not None:
            block["confidence"] = float(confidence)
    except (TypeError, ValueError):
        pass
    if provider:
        block["provider"] = provider
    blocks.append(block)


def extract_blocks_from_parsing_text(text: str, provider: str = "") -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if not text:
        return blocks
    chunks = re.split(r"#+", text)
    for chunk in chunks:
        bbox_match = re.search(r"bbox:\s*\[([^\]]+)\]", chunk)
        content_match = re.search(r"content:\s*(.*)", chunk, flags=re.S)
        if not content_match:
            continue
        content = content_match.group(1).strip()
        if not content:
            continue
        bbox = None
        if bbox_match:
            try:
                bbox = [float(item.strip()) for item in bbox_match.group(1).split(",")[:4]]
            except ValueError:
                bbox = None
        append_ocr_block(blocks, content, bbox, None, provider)
    return blocks


def extract_ocr_blocks_from_payload(payload: Any, provider: str = "") -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("parsing_res_list"), list):
                for item in value.get("parsing_res_list") or []:
                    blocks.extend(extract_blocks_from_parsing_text(str(item or ""), provider))

            if isinstance(value.get("rec_texts"), list):
                texts = value.get("rec_texts") or []
                boxes = value.get("rec_polys") or value.get("dt_polys") or value.get("boxes") or []
                scores = value.get("rec_scores") or value.get("scores") or []
                for index, text in enumerate(texts):
                    append_ocr_block(
                        blocks,
                        text,
                        boxes[index] if index < len(boxes) else None,
                        scores[index] if index < len(scores) else None,
                        provider,
                    )
                return

            if isinstance(value.get("txts"), (list, tuple)):
                texts = value.get("txts") or []
                boxes = value.get("boxes") or value.get("dt_boxes") or value.get("rec_polys") or []
                scores = value.get("scores") or value.get("rec_scores") or []
                for index, text in enumerate(texts):
                    append_ocr_block(
                        blocks,
                        text,
                        boxes[index] if index < len(boxes) else None,
                        scores[index] if index < len(scores) else None,
                        provider,
                    )
                return

            text = (
                value.get("text")
                or value.get("content")
                or value.get("block_content")
                or value.get("rec_text")
                or value.get("markdown")
                or value.get("plain_text")
            )
            if text:
                append_ocr_block(
                    blocks,
                    text,
                    value.get("location")
                    or value.get("bbox")
                    or value.get("box")
                    or value.get("points")
                    or value.get("poly")
                    or value.get("polygon")
                    or value.get("block_bbox"),
                    value.get("confidence") or value.get("score") or value.get("rec_score"),
                    provider,
                )

            for item in value.values():
                if isinstance(item, (dict, list, tuple)):
                    walk(item)
            return

        if isinstance(value, (list, tuple)):
            if len(value) >= 2 and isinstance(value[1], str) and isinstance(value[0], (list, tuple)):
                append_ocr_block(
                    blocks,
                    value[1],
                    value[0],
                    value[2] if len(value) >= 3 else None,
                    provider,
                )
                return
            for item in value:
                walk(item)

    walk(object_to_plain_data(payload))
    unique_blocks: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, Tuple[int, int, int, int]]] = set()
    for block in blocks:
        location = block.get("location") or {}
        key = (
            str(block.get("text") or ""),
            (
                int(location.get("left", 0) or 0),
                int(location.get("top", 0) or 0),
                int(location.get("width", 0) or 0),
                int(location.get("height", 0) or 0),
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_blocks.append(block)
    return unique_blocks


def rapidocr_result_to_plain_data(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return object_to_plain_data(result[0])

    direct: Dict[str, Any] = {}
    for attr_name in ("txts", "boxes", "scores"):
        attr = getattr(result, attr_name, None)
        if attr is not None:
            direct[attr_name] = object_to_plain_data(attr)
    if direct:
        return direct

    for method_name in ("to_json", "to_list", "to_dict"):
        method = getattr(result, method_name, None)
        if callable(method):
            try:
                data = method()
            except TypeError:
                continue
            if isinstance(data, str):
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return data
            return object_to_plain_data(data)

    return object_to_plain_data(result)


def rapidocr_blocks(image: Image.Image) -> List[Dict[str, Any]]:
    image_bytes = image_to_jpeg_bytes(image)
    ocr_cache_enabled = env_flag("VALIDATOR_ENABLE_OCR_CACHE", True)
    provider_params = os.getenv("VALIDATOR_RAPIDOCR_PARAMS_JSON", "").strip()
    provider_key = "rapidocr"
    if provider_params:
        provider_key = f"{provider_key}:{cache_key(provider_params)}"
    ocr_cache_path = cache_file("ocr", ocr_cache_key(provider_key, image_bytes), ".json")
    if ocr_cache_enabled and ocr_cache_path.exists():
        cached = json.loads(ocr_cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, list):
            return cached

    engine = get_rapidocr_runtime()
    prepared = normalize_image(image)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        prepared.save(tmp, format="JPEG", quality=92)
    try:
        result = engine(str(tmp_path))
        blocks = extract_ocr_blocks_from_payload(rapidocr_result_to_plain_data(result), "rapidocr")
        if ocr_cache_enabled:
            ocr_cache_path.parent.mkdir(parents=True, exist_ok=True)
            ocr_cache_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
        return blocks
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def paddleocr_vl_blocks(image: Image.Image) -> List[Dict[str, Any]]:
    image_bytes = image_to_jpeg_bytes(image)
    ocr_cache_enabled = env_flag("VALIDATOR_ENABLE_OCR_CACHE", True)
    provider_key = f"paddleocr_vl:{os.getenv('VALIDATOR_PADDLEOCR_VL_VERSION', 'v1.6')}"
    ocr_cache_path = cache_file("ocr", ocr_cache_key(provider_key, image_bytes), ".json")
    if ocr_cache_enabled and ocr_cache_path.exists():
        cached = json.loads(ocr_cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, list):
            return cached

    pipeline = get_paddleocr_vl_runtime()
    prepared = normalize_image(image)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        prepared.save(tmp, format="JPEG", quality=92)
    try:
        result = pipeline.predict(str(tmp_path))
        if not isinstance(result, (dict, list, tuple, str)):
            try:
                result = list(result)
            except TypeError:
                pass
        blocks = extract_ocr_blocks_from_payload(result, "paddleocr_vl")
        if not blocks:
            raise RuntimeError("PaddleOCR-VL 未返回可解析的文字块")
        if ocr_cache_enabled:
            ocr_cache_path.parent.mkdir(parents=True, exist_ok=True)
            ocr_cache_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
        return blocks
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def baidu_ocr_blocks(image: Image.Image) -> List[Dict[str, Any]]:
    """调用百度通用文字识别，返回带位置的 OCR 文字块。"""
    image_bytes = image_to_jpeg_bytes(image)
    ocr_cache_enabled = env_flag("VALIDATOR_ENABLE_OCR_CACHE", True)
    ocr_cache_path = cache_file("ocr", ocr_cache_key("baidu", image_bytes), ".json")
    if ocr_cache_enabled and ocr_cache_path.exists():
        cached = json.loads(ocr_cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, list):
            return cached

    token = get_baidu_ocr_access_token()
    response = get_retry_session().post(
        BAIDU_OCR_API_URL,
        params={"access_token": token},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "image": base64.b64encode(image_bytes).decode("utf-8"),
            "language_type": "CHN_ENG",
            "detect_direction": "true",
            "vertexes_location": "false",
            "paragraph": "false",
            "probability": "false",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error_code"):
        raise RuntimeError(f"百度 OCR 调用失败：{payload}")
    blocks: List[Dict[str, Any]] = []
    for item in payload.get("words_result", []):
        text = str(item.get("words") or "").strip()
        if not text:
            continue
        location = item.get("location") if isinstance(item.get("location"), dict) else {}
        blocks.append(
            {
                "text": text,
                "location": {
                    "left": int(location.get("left", 0) or 0),
                    "top": int(location.get("top", 0) or 0),
                    "width": int(location.get("width", 0) or 0),
                    "height": int(location.get("height", 0) or 0),
                },
                "provider": "baidu",
            }
        )
    if ocr_cache_enabled:
        ocr_cache_path.parent.mkdir(parents=True, exist_ok=True)
        ocr_cache_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    return blocks


def ocr_blocks(
    image: Image.Image,
    providers: Optional[List[str]] = None,
    fallback_on_empty: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """统一 OCR 入口；默认使用配置的主体 OCR provider 链。"""
    providers = ocr_provider_chain() if providers is None else providers
    fallback_on_empty = should_fallback_on_empty_ocr() if fallback_on_empty is None else fallback_on_empty

    errors: List[str] = []
    for provider in providers:
        try:
            if provider == "rapidocr":
                blocks = rapidocr_blocks(image)
            elif provider == "paddleocr_vl":
                blocks = paddleocr_vl_blocks(image)
            elif provider == "baidu":
                blocks = baidu_ocr_blocks(image)
            else:
                continue
            if blocks or not fallback_on_empty:
                return blocks
            errors.append(f"{provider}: empty OCR result")
        except Exception as exc:
            errors.append(f"{provider}: {exc}")

    raise RuntimeError("OCR provider 全部失败：" + " | ".join(errors))


def watermark_ocr_blocks(image: Image.Image) -> List[Dict[str, Any]]:
    providers = watermark_ocr_provider_chain()
    if not providers:
        return []
    return ocr_blocks(
        image,
        providers=providers,
        fallback_on_empty=True,
    )


def ocr_blocks_to_text(blocks: List[Dict[str, Any]]) -> str:
    """把 OCR 文字块按返回顺序拼接为文本。"""
    return "\n".join(str(block.get("text") or "").strip() for block in blocks if str(block.get("text") or "").strip())


def baidu_ocr_text(image: Image.Image) -> str:
    """调用百度通用文字识别，返回拼接后的识别文本。"""
    return ocr_blocks_to_text(baidu_ocr_blocks(image))


def ocr_text(image: Image.Image) -> str:
    """使用当前 OCR provider 链返回拼接后的识别文本。"""
    return ocr_blocks_to_text(ocr_blocks(image))


def compact_text(text: str, limit: int = 500) -> str:
    """压缩长文本，避免 validation_signals 过大。"""
    compacted = re.sub(r"\s+", " ", text or "").strip()
    if len(compacted) <= limit:
        return compacted
    return f"{compacted[:limit]}..."


def compact_llm_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """压缩发给 Validator LLM 的信号，避免请求体过大。"""
    compacted: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            compacted[key] = compact_text(value, 500)
        elif isinstance(value, list):
            compacted[key] = value[:30]
        elif isinstance(value, dict):
            compacted[key] = compact_llm_payload(value)
        else:
            compacted[key] = value
    return compacted


def collect_text_context(data: Dict[str, Any]) -> str:
    """收集 retriever 或 target_image 中可用于文本校验的短文本。"""
    parts: List[str] = []
    for field_name in TEXT_CONTEXT_FIELDS:
        value = data.get(field_name)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            parts.extend(str(item).strip() for item in value if str(item).strip())
    return "\n".join(unique_values(parts))


def tokenize_text(text: str) -> Set[str]:
    """中英文混合分词：中文按字和相邻 bigram，英文/数字按词。"""
    raw_tokens = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", (text or "").lower())
    tokens: Set[str] = {token for token in raw_tokens if token.strip()}
    for left, right in zip(raw_tokens, raw_tokens[1:]):
        if len(left) == 1 and len(right) == 1:
            tokens.add(f"{left}{right}")
    return tokens


def compute_text_overlap(left: str, right: str) -> float:
    """计算 OCR 文本集合重合度，作为弱辅助信号。"""
    left_tokens = tokenize_text(left)
    right_tokens = tokenize_text(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def shared_text_terms(left: str, right: str, limit: int = 20) -> List[str]:
    """返回文本交集，便于解释为什么两段文本相似。"""
    shared = tokenize_text(left) & tokenize_text(right)
    return sorted(shared, key=lambda item: (-len(item), item))[:limit]


def split_ocr_lines(text: str) -> List[str]:
    """把 OCR 文本拆成可判断水印的短行。"""
    raw_lines = re.split(r"[\r\n]+| {2,}", text or "")
    return [line.strip() for line in raw_lines if line.strip()]


def normalize_watermark_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = normalized.replace("＠", "@").replace("：", ":")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f" {normalized} "


def detect_platforms(text: str) -> List[str]:
    """基于 OCR 文本识别可能的平台水印，允许多个平台同时存在。"""
    normalized = normalize_watermark_text(text)
    platforms: List[str] = []
    for platform, aliases in WATERMARK_PLATFORM_ALIASES.items():
        for alias in aliases:
            if alias.lower() in normalized:
                platforms.append(platform)
                break
    return unique_values(platforms)


def is_valid_watermark_account(account: str, platform: str = "") -> bool:
    normalized = str(account or "").strip(" _-:：@#，,。.;；|/\\")
    lowered = normalized.lower()
    blocked = {
        "",
        "id",
        "uid",
        "号",
        "账号",
        "帳號",
        "微博",
        "weibo",
        "小红书",
        "小紅書",
        "xhs",
        "rednote",
        "red",
        "抖音",
        "douyin",
        "tiktok",
        "快手",
        "kuaishou",
        "bilibili",
        "b站",
    }
    if lowered in blocked:
        return False
    if platform in {"xiaohongshu", "douyin", "kuaishou"} and not re.search(r"[A-Za-z0-9]", normalized):
        return False
    return len(normalized) >= 2


def extract_accounts_for_platform(text: str, platform: str) -> List[str]:
    patterns = {
        "weibo": [
            rf"(?:微博(?:号|id|用户|博主)?|新浪微博|weibo|wb)\s*[:：@]?\s*({ACCOUNT_CHARS})",
            rf"@({ACCOUNT_CHARS})",
        ],
        "xiaohongshu": [
            rf"\b((?:xhs|red)[_-]?[A-Za-z0-9][A-Za-z0-9_.-]{{2,39}})\b",
            rf"(?:小红书(?:号|id|ID|账号)|小紅書(?:號|id|ID|帳號)|小红薯(?:号|id|ID|账号)|xhs(?:id)?|RED\s*ID|red\s*id|rednote\s*id|xiaohongshu\s*id)\s*[:：@]?\s*({ASCII_ACCOUNT_CHARS})",
        ],
        "douyin": [
            rf"(?:抖音(?:号|id|账号)?|douyin|tiktok)\s*[:：@]?\s*({ASCII_ACCOUNT_CHARS})",
        ],
        "kuaishou": [
            rf"(?:快手(?:号|id|账号)?|kuaishou)\s*[:：@]?\s*({ASCII_ACCOUNT_CHARS})",
        ],
        "bilibili": [
            rf"(?:哔哩哔哩|bilibili|b站)\s*[:：@]?\s*({ACCOUNT_CHARS})",
            rf"(?:uid|UID)\s*[:：]?\s*([0-9]{{3,20}})",
        ],
    }
    accounts: List[str] = []
    for pattern in patterns.get(platform, []):
        for match in re.finditer(pattern, text or "", flags=re.I):
            account = match.group(1).strip(" _-:：@")
            if is_valid_watermark_account(account, platform):
                accounts.append(account)
    unique_accounts = unique_values(accounts)
    return [
        account
        for account in unique_accounts
        if not any(account != other and account in other for other in unique_accounts)
    ]


def normalize_ocr_location(location: Dict[str, Any], image_size: Optional[Tuple[int, int]]) -> Dict[str, Any]:
    """把百度 OCR location 补充为相对坐标，便于后续 Analyzer 使用。"""
    left = int(location.get("left", 0) or 0)
    top = int(location.get("top", 0) or 0)
    width = int(location.get("width", 0) or 0)
    height = int(location.get("height", 0) or 0)
    result: Dict[str, Any] = {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }
    if image_size and image_size[0] > 0 and image_size[1] > 0:
        image_width, image_height = image_size
        result.update(
            {
                "relative_left": round(left / image_width, 4),
                "relative_top": round(top / image_height, 4),
                "relative_width": round(width / image_width, 4),
                "relative_height": round(height / image_height, 4),
            }
        )
    return result


def ocr_location_hint(location: Dict[str, Any], image_size: Optional[Tuple[int, int]]) -> Dict[str, Any]:
    """判断 OCR 块是否位于常见水印区域。"""
    if not image_size or image_size[0] <= 0 or image_size[1] <= 0:
        return {"near_edge": False, "region": "unknown", "small_block": False}

    image_width, image_height = image_size
    left = int(location.get("left", 0) or 0)
    top = int(location.get("top", 0) or 0)
    width = int(location.get("width", 0) or 0)
    height = int(location.get("height", 0) or 0)
    right = left + width
    bottom = top + height
    margin_x = image_width * 0.12
    margin_y = image_height * 0.12

    horizontal = "left" if left <= margin_x else "right" if right >= image_width - margin_x else "center"
    vertical = "top" if top <= margin_y else "bottom" if bottom >= image_height - margin_y else "middle"
    near_edge = horizontal != "center" or vertical != "middle"
    small_block = (width * height) <= (image_width * image_height * 0.035)
    return {
        "near_edge": near_edge,
        "region": f"{vertical}_{horizontal}",
        "small_block": small_block,
    }


def build_watermark_summary(watermarks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总水印明细为 Analyzer 方便消费的顶层字段。"""
    unique_watermarks: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for item in watermarks:
        key = (
            str(item.get("platform") or ""),
            str(item.get("account") or ""),
            str(item.get("raw_text") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_watermarks.append(item)
    watermarks = unique_watermarks
    platforms = unique_values([item["platform"] for item in watermarks])
    accounts_by_platform: Dict[str, List[str]] = {}
    for item in watermarks:
        platform = item["platform"]
        accounts_by_platform.setdefault(platform, [])
        accounts_by_platform[platform] = unique_values(
            [*accounts_by_platform[platform], *item.get("accounts", [])]
        )
    watermark_text = unique_values([item["raw_text"] for item in watermarks])
    return {
        "watermark_detected": bool(watermarks),
        "watermarks": watermarks,
        "watermark_platforms": platforms,
        "watermark_accounts": accounts_by_platform,
        "watermark_text": watermark_text,
    }


def empty_watermark_summary() -> Dict[str, Any]:
    return build_watermark_summary([])


def blocks_have_watermark_capable_provider(blocks: List[Dict[str, Any]]) -> bool:
    return any(str(block.get("provider") or "") in {"paddleocr_vl", "baidu"} for block in blocks)


def merge_watermark_summary(
    base_analysis: Dict[str, Any],
    extra_summary: Dict[str, Any],
) -> Dict[str, Any]:
    combined = build_watermark_summary(
        [
            *list(base_analysis.get("watermarks", [])),
            *list(extra_summary.get("watermarks", [])),
        ]
    )
    return {
        **base_analysis,
        **combined,
    }


def ocr_block_output(block: Dict[str, Any], image_size: Optional[Tuple[int, int]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "text": compact_text(str(block.get("text") or ""), 160),
        "location": normalize_ocr_location(
            block.get("location") if isinstance(block.get("location"), dict) else {},
            image_size,
        ),
    }
    if block.get("provider"):
        output["provider"] = block.get("provider")
    if isinstance(block.get("confidence"), (int, float)):
        output["confidence"] = round(float(block["confidence"]), 4)
    return output


def ocr_block_confidence(block: Dict[str, Any]) -> Optional[float]:
    if isinstance(block.get("confidence"), (int, float)):
        return float(block["confidence"])
    return None


def merge_ocr_locations(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, int]:
    left_box = left.get("location") if isinstance(left.get("location"), dict) else {}
    right_box = right.get("location") if isinstance(right.get("location"), dict) else {}
    x1 = min(int(left_box.get("left", 0) or 0), int(right_box.get("left", 0) or 0))
    y1 = min(int(left_box.get("top", 0) or 0), int(right_box.get("top", 0) or 0))
    x2 = max(
        int(left_box.get("left", 0) or 0) + int(left_box.get("width", 0) or 0),
        int(right_box.get("left", 0) or 0) + int(right_box.get("width", 0) or 0),
    )
    y2 = max(
        int(left_box.get("top", 0) or 0) + int(left_box.get("height", 0) or 0),
        int(right_box.get("top", 0) or 0) + int(right_box.get("height", 0) or 0),
    )
    return {"left": x1, "top": y1, "width": max(0, x2 - x1), "height": max(0, y2 - y1)}


def watermark_candidate_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = list(blocks)
    sorted_blocks = sorted(
        blocks,
        key=lambda block: (
            int((block.get("location") or {}).get("top", 0) or 0),
            int((block.get("location") or {}).get("left", 0) or 0),
        ),
    )
    for left, right in zip(sorted_blocks, sorted_blocks[1:]):
        left_text = str(left.get("text") or "").strip()
        right_text = str(right.get("text") or "").strip()
        if not left_text or not right_text:
            continue
        left_location = left.get("location") if isinstance(left.get("location"), dict) else {}
        right_location = right.get("location") if isinstance(right.get("location"), dict) else {}
        top_delta = abs(int(left_location.get("top", 0) or 0) - int(right_location.get("top", 0) or 0))
        max_height = max(int(left_location.get("height", 0) or 0), int(right_location.get("height", 0) or 0), 1)
        if top_delta > max_height * 1.5:
            continue
        if not detect_platforms(left_text) and not detect_platforms(right_text):
            continue
        confidence_values = [value for value in (ocr_block_confidence(left), ocr_block_confidence(right)) if value is not None]
        merged: Dict[str, Any] = {
            "text": f"{left_text} {right_text}",
            "location": merge_ocr_locations(left, right),
            "provider": left.get("provider") or right.get("provider"),
            "source": "ocr_adjacent_merge",
        }
        if confidence_values:
            merged["confidence"] = sum(confidence_values) / len(confidence_values)
        candidates.append(merged)
    return candidates


def detect_watermarks_from_blocks(
    blocks: List[Dict[str, Any]],
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """从带位置 OCR 块中提取一个或多个平台水印候选。"""
    watermarks: List[Dict[str, Any]] = []
    for block in watermark_candidate_blocks(blocks):
        line = str(block.get("text") or "").strip()
        if not line:
            continue
        platforms = detect_platforms(line)
        if not platforms:
            continue
        location = block.get("location") if isinstance(block.get("location"), dict) else {}
        position = normalize_ocr_location(location, image_size)
        location_hint = ocr_location_hint(location, image_size)
        overlap_status = "possible_overlap" if len(platforms) > 1 else "separate"
        for platform in platforms:
            accounts = extract_accounts_for_platform(line, platform)
            if block.get("source") == "ocr_adjacent_merge" and not accounts and len(line) > 36:
                continue
            confidence = 0.62
            ocr_confidence = ocr_block_confidence(block)
            if ocr_confidence is not None:
                confidence += max(0.0, min(ocr_confidence, 1.0)) * 0.08
            if accounts:
                confidence += 0.18
            if location_hint["near_edge"]:
                confidence += 0.08
            if location_hint["small_block"]:
                confidence += 0.04
            if len(line) > 36 and not accounts:
                confidence -= 0.12
            watermarks.append(
                {
                    "platform": platform,
                    "account": accounts[0] if accounts else "",
                    "accounts": accounts,
                    "raw_text": compact_text(line, 160),
                    "confidence": round(max(0.0, min(confidence, 0.95)), 2),
                    "source": block.get("source") or ("ocr_rule_location" if image_size else "ocr_rule"),
                    "ocr_provider": block.get("provider", ""),
                    "overlap_status": overlap_status,
                    "watermark_position": position,
                    "position_hint": location_hint,
                }
            )
    return build_watermark_summary(watermarks)


def detect_watermarks(text: str) -> Dict[str, Any]:
    """从 OCR 文本中提取一个或多个平台水印候选，重叠时全部保留。"""
    lines = split_ocr_lines(text)
    if not lines and text.strip():
        lines = [text.strip()]

    watermarks: List[Dict[str, Any]] = []
    for line in lines:
        platforms = detect_platforms(line)
        if not platforms:
            continue
        overlap_status = "possible_overlap" if len(platforms) > 1 else "separate"
        for platform in platforms:
            accounts = extract_accounts_for_platform(line, platform)
            confidence = 0.85 if accounts else 0.62
            if not accounts and len(line) > 36:
                confidence = 0.50
            watermarks.append(
                {
                    "platform": platform,
                    "account": accounts[0] if accounts else "",
                    "accounts": accounts,
                    "raw_text": compact_text(line, 160),
                    "confidence": confidence,
                    "source": "ocr_rule",
                    "overlap_status": overlap_status,
                }
            )

    return build_watermark_summary(watermarks)


def is_watermark_line(line: str) -> bool:
    """判断 OCR 行是否更像平台水印；主体字幕不轻易删除。"""
    platforms = detect_platforms(line)
    if not platforms:
        return False
    has_account = any(extract_accounts_for_platform(line, platform) for platform in platforms)
    return has_account or len(line.strip()) <= 18


def is_watermark_block(block: Dict[str, Any], image_size: Optional[Tuple[int, int]] = None) -> bool:
    """结合文字和位置判断 OCR 块是否更像平台水印。"""
    line = str(block.get("text") or "").strip()
    platforms = detect_platforms(line)
    if not platforms:
        return False
    has_account = any(extract_accounts_for_platform(line, platform) for platform in platforms)
    location = block.get("location") if isinstance(block.get("location"), dict) else {}
    location_hint = ocr_location_hint(location, image_size)
    return (
        has_account
        or len(line) <= 18
        or (location_hint["near_edge"] and location_hint["small_block"])
    )


def strip_watermark_text(text: str) -> str:
    """从 OCR 文本中剥离水印行，留下字幕/新闻条/海报正文等主体文本。"""
    content_lines = [line for line in split_ocr_lines(text) if not is_watermark_line(line)]
    return "\n".join(content_lines).strip()


def analyze_ocr_text(text: str, detect_watermark: bool = False) -> Dict[str, Any]:
    """把 OCR 文本拆成完整文本、主体文本和水印线索。"""
    full_text = text or ""
    content_text = strip_watermark_text(full_text) if detect_watermark else full_text
    watermarks = detect_watermarks(full_text) if detect_watermark else empty_watermark_summary()
    return {
        "ocr_text": compact_text(full_text),
        "ocr_content_text": compact_text(content_text),
        "watermark_source": "ocr_rule" if detect_watermark else "none",
        **watermarks,
    }


def analyze_ocr_blocks(
    blocks: List[Dict[str, Any]],
    image_size: Optional[Tuple[int, int]] = None,
    detect_watermark: Optional[bool] = None,
) -> Dict[str, Any]:
    """把带位置 OCR 块拆成完整文本、主体文本和水印线索。"""
    detect_watermark = blocks_have_watermark_capable_provider(blocks) if detect_watermark is None else detect_watermark
    full_text = ocr_blocks_to_text(blocks)
    content_text = "\n".join(
        str(block.get("text") or "").strip()
        for block in blocks
        if str(block.get("text") or "").strip()
        and (not detect_watermark or not is_watermark_block(block, image_size))
    )
    watermarks = detect_watermarks_from_blocks(blocks, image_size) if detect_watermark else empty_watermark_summary()
    return {
        "ocr_text": compact_text(full_text),
        "ocr_content_text": compact_text(content_text),
        "ocr_blocks": [ocr_block_output(block, image_size) for block in blocks],
        "watermark_source": "ocr_rule" if detect_watermark else "none",
        **watermarks,
    }


def compute_ocr_similarity(
    target_image: Image.Image,
    candidate_image: Image.Image,
    target_text: Optional[str] = None,
    target_analysis: Optional[Dict[str, Any]] = None,
    enable_watermark_ocr: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """OCR 校验：适合海报、截图、表情包等含文字图片。"""
    if target_analysis is None:
        if target_text is not None:
            target_analysis = analyze_ocr_text(target_text)
        else:
            target_blocks = ocr_blocks(target_image)
            target_analysis = analyze_ocr_blocks(target_blocks, target_image.size)
    candidate_blocks = ocr_blocks(candidate_image)
    candidate_analysis = analyze_ocr_blocks(candidate_blocks, candidate_image.size)
    watermark_ocr_used = False
    watermark_ocr_error = ""
    if enable_watermark_ocr:
        try:
            candidate_watermark_blocks = watermark_ocr_blocks(candidate_image)
            candidate_watermark_summary = detect_watermarks_from_blocks(
                candidate_watermark_blocks,
                candidate_image.size,
            )
            candidate_analysis = merge_watermark_summary(candidate_analysis, candidate_watermark_summary)
            watermark_ocr_used = True
        except Exception as exc:
            watermark_ocr_error = str(exc)
    target_content_text = str(target_analysis.get("ocr_content_text") or "")
    candidate_content_text = str(candidate_analysis.get("ocr_content_text") or "")
    overlap = compute_text_overlap(target_content_text, candidate_content_text)
    watermark_source = str(candidate_analysis.get("watermark_source") or "none")
    if watermark_ocr_used:
        watermark_source = "watermark_ocr" if watermark_source == "none" else f"{watermark_source}+watermark_ocr"
    return overlap, {
        "candidate_ocr_text": candidate_analysis["ocr_text"],
        "candidate_content_text": candidate_content_text,
        "candidate_ocr_blocks": candidate_analysis.get("ocr_blocks", []),
        "ocr_shared_terms": shared_text_terms(target_content_text, candidate_content_text),
        "watermark_detected": candidate_analysis["watermark_detected"],
        "watermarks": candidate_analysis["watermarks"],
        "watermark_platforms": candidate_analysis["watermark_platforms"],
        "watermark_accounts": candidate_analysis["watermark_accounts"],
        "watermark_text": candidate_analysis["watermark_text"],
        "watermark_evidence": {
            "source": watermark_source,
            "confidence": max(
                [item.get("confidence", 0.0) for item in candidate_analysis["watermarks"]],
                default=0.0,
            ),
        },
        "watermark_source": watermark_source,
        "watermark_ocr_used": watermark_ocr_used,
        "watermark_ocr_provider_chain": watermark_ocr_provider_chain() if watermark_ocr_used else [],
        "watermark_ocr_error": watermark_ocr_error,
    }


def compute_retrieved_text_similarity(target_text: str, node: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """比较目标图文本和 retriever 返回的标题/摘要等短文本。"""
    candidate_text = collect_text_context(node)
    overlap = compute_text_overlap(target_text, candidate_text)
    return overlap, {
        "retrieved_text_excerpt": compact_text(candidate_text),
        "retrieved_text_shared_terms": shared_text_terms(target_text, candidate_text),
    }


def hard_text_similarity_score(signals: Dict[str, Any]) -> float:
    """Text evidence that is specific enough to preserve a candidate."""
    scores = [
        float(signals[name])
        for name in (
            "ocr_content_overlap",
            "ocr_text_overlap",
            "retrieved_text_overlap",
        )
        if isinstance(signals.get(name), (int, float))
    ]
    return max(scores, default=0.0)


def has_hard_text_similarity_signal(signals: Dict[str, Any]) -> bool:
    return any(
        isinstance(signals.get(name), (int, float))
        for name in (
            "ocr_content_overlap",
            "ocr_text_overlap",
            "retrieved_text_overlap",
        )
    )


def should_run_candidate_ocr(signals: Dict[str, Any]) -> Tuple[bool, str]:
    if not ocr_prefilter_enabled():
        return True, "prefilter_disabled"

    phash_similarity = float(signals.get("phash_similarity", 0.0) or 0.0)
    resized_similarity = float(signals.get("resized_image_similarity", 0.0) or 0.0)
    grayscale_similarity = float(signals.get("grayscale_similarity", 0.0) or 0.0)
    color_hist_similarity = float(signals.get("color_hist_similarity", 0.0) or 0.0)
    clip_similarity = float(signals.get("clip_similarity", 0.0) or 0.0)
    has_resized_signal = isinstance(signals.get("resized_image_similarity"), (int, float))
    has_color_signal = isinstance(signals.get("color_hist_similarity"), (int, float))
    text_score = hard_text_similarity_score(signals)
    visual_score = max(
        phash_similarity,
        clip_similarity,
        min(resized_similarity, grayscale_similarity),
        min(grayscale_similarity, color_hist_similarity),
    )

    if phash_similarity >= hash_weak_threshold():
        return True, "phash_candidate"
    if visual_score >= ocr_prefilter_visual_threshold():
        return True, "visual_candidate"
    if text_score >= ocr_prefilter_text_threshold():
        return True, "retrieved_text_overlap_candidate"
    return False, "obvious_visual_text_mismatch"


def weighted_similarity(signals: Dict[str, float]) -> float:
    """融合多路相似度；缺失信号会自动重新归一化权重。"""
    weights = {
        "phash_similarity": 0.22,
        "resized_image_similarity": 0.08,
        "grayscale_similarity": 0.08,
        "color_hist_similarity": 0.05,
        "clip_similarity": 0.34,
        "ocr_content_overlap": 0.10,
        "ocr_text_overlap": 0.0,
        "retrieved_text_overlap": 0.05,
        "retrieved_text_semantic_similarity": 0.03,
    }
    available = {
        name: value
        for name, value in signals.items()
        if value is not None and name in weights
    }
    total_weight = sum(weights[name] for name in available)
    if total_weight <= 0:
        return 0.0
    weighted_score = sum(available[name] * weights[name] for name in available) / total_weight
    phash_similarity = available.get("phash_similarity", 0.0)
    resized_image_similarity = available.get("resized_image_similarity", 0.0)
    grayscale_similarity = available.get("grayscale_similarity", 0.0)
    color_hist_similarity = available.get("color_hist_similarity", 0.0)
    clip_similarity = available.get("clip_similarity", 0.0)

    text_support = max(
        [
            available[name]
            for name in (
                "ocr_text_overlap",
                "ocr_content_overlap",
                "retrieved_text_overlap",
            )
            if name in available
        ],
        default=None,
    )

    if phash_similarity >= 0.92 and resized_image_similarity >= 0.85:
        return max(weighted_score, phash_similarity)
    if phash_similarity >= 0.88 and grayscale_similarity >= 0.88:
        return max(weighted_score, min(phash_similarity, grayscale_similarity))
    if grayscale_similarity >= 0.90 and color_hist_similarity >= 0.75 and clip_similarity >= 0.80:
        return max(weighted_score, clip_similarity)
    if clip_similarity >= 0.85 and phash_similarity >= 0.35 and (
        text_support is None or text_support >= 0.12
    ):
        return max(weighted_score, clip_similarity)
    return weighted_score


def passes_text_filter(signals: Dict[str, float]) -> bool:
    """只用硬文本重合过滤；检索文本语义相似度不能单独救回弱候选。"""
    if not has_hard_text_similarity_signal(signals):
        return True

    text_score = hard_text_similarity_score(signals)
    strong_visual, _ = tampering_visual_candidate(signals)
    if strong_visual:
        return True
    return text_score >= tampering_text_signal_threshold()


def tampering_visual_candidate(signals: Dict[str, float]) -> Tuple[bool, str]:
    phash_similarity = float(signals.get("phash_similarity", 0.0) or 0.0)
    resized_similarity = float(signals.get("resized_image_similarity", 0.0) or 0.0)
    grayscale_similarity = float(signals.get("grayscale_similarity", 0.0) or 0.0)
    color_hist_similarity = float(signals.get("color_hist_similarity", 0.0) or 0.0)
    clip_similarity = float(signals.get("clip_similarity", 0.0) or 0.0)
    if phash_similarity >= hash_weak_threshold():
        return True, "phash_visual_evidence"
    if grayscale_similarity >= 0.88 and resized_similarity >= 0.80:
        return True, "structure_visual_evidence"
    if grayscale_similarity >= 0.88 and color_hist_similarity >= 0.70:
        return True, "gray_color_visual_evidence"
    if clip_similarity >= clip_review_threshold() and max(phash_similarity, grayscale_similarity) >= 0.60:
        return True, "clip_with_structure_evidence"
    return False, "not_visual_candidate"


def tampering_text_candidate(signals: Dict[str, float]) -> Tuple[bool, str]:
    text_score = hard_text_similarity_score(signals)
    if text_score >= tampering_text_signal_threshold():
        return True, "text_evidence"
    return False, "not_text_candidate"


def text_only_visual_score(signals: Dict[str, float]) -> float:
    phash_similarity = float(signals.get("phash_similarity", 0.0) or 0.0)
    resized_similarity = float(signals.get("resized_image_similarity", 0.0) or 0.0)
    grayscale_similarity = float(signals.get("grayscale_similarity", 0.0) or 0.0)
    color_hist_similarity = float(signals.get("color_hist_similarity", 0.0) or 0.0)
    clip_similarity = float(signals.get("clip_similarity", 0.0) or 0.0)
    return max(
        phash_similarity,
        min(resized_similarity, grayscale_similarity),
        min(grayscale_similarity, color_hist_similarity),
        min(clip_similarity, max(phash_similarity, grayscale_similarity, resized_similarity)),
    )


def tampering_candidate_filter(signals: Dict[str, float]) -> Tuple[bool, str]:
    visual_passed, visual_reason = tampering_visual_candidate(signals)
    text_passed, text_reason = tampering_text_candidate(signals)
    text_visual_score = text_only_visual_score(signals)
    if visual_passed and not text_passed:
        return True, f"visual_preserved_for_tampering:{visual_reason}"
    if text_passed and not visual_passed:
        if text_visual_score >= text_only_min_visual_threshold():
            return True, f"text_preserved_for_tampering:{text_reason}:visual_score={text_visual_score:.2f}"
        return False, f"text_only_visual_too_low:{text_reason}:visual_score={text_visual_score:.2f}"
    if visual_passed and text_passed:
        return True, "visual_text_preserved"
    return False, "not_tampering_candidate"


def signal_visual_score(signals: Dict[str, Any]) -> float:
    phash_similarity = float(signals.get("phash_similarity", 0.0) or 0.0)
    resized_similarity = float(signals.get("resized_image_similarity", 0.0) or 0.0)
    grayscale_similarity = float(signals.get("grayscale_similarity", 0.0) or 0.0)
    color_hist_similarity = float(signals.get("color_hist_similarity", 0.0) or 0.0)
    clip_similarity = float(signals.get("clip_similarity", 0.0) or 0.0)
    return max(
        phash_similarity,
        clip_similarity,
        min(resized_similarity, grayscale_similarity),
        min(grayscale_similarity, color_hist_similarity),
    )


def signal_text_score(signals: Dict[str, Any]) -> float:
    return hard_text_similarity_score(signals)


def has_text_similarity_signal(signals: Dict[str, Any]) -> bool:
    return has_hard_text_similarity_signal(signals)


def layered_visual_filter(signals: Dict[str, float]) -> Tuple[bool, str]:
    """层级过滤：先看强哈希，再进入 CLIP/灰度/文本复核，降低拼接图漏检。"""
    phash_similarity = signals.get("phash_similarity", 0.0)
    resized_image_similarity = signals.get("resized_image_similarity", 0.0)
    grayscale_similarity = signals.get("grayscale_similarity", 0.0)
    color_hist_similarity = signals.get("color_hist_similarity", 0.0)
    clip_similarity = signals.get("clip_similarity", 0.0)
    has_resized_signal = isinstance(signals.get("resized_image_similarity"), (int, float))
    has_color_signal = isinstance(signals.get("color_hist_similarity"), (int, float))
    text_similarity = hard_text_similarity_score(signals)

    if phash_similarity >= hash_strong_threshold():
        return True, "hash_strong_pass"
    if phash_similarity >= hash_weak_threshold() and clip_similarity >= clip_review_threshold():
        return True, "hash_then_clip_pass"
    if grayscale_similarity >= 0.88 and clip_similarity >= clip_review_threshold():
        return True, "grayscale_then_clip_pass"
    if clip_similarity >= 0.82 and text_similarity >= 0.20:
        return True, "clip_text_joint_pass"
    if clip_similarity >= 0.86 and grayscale_similarity >= 0.70:
        return True, "montage_semantic_pass"
    if grayscale_similarity >= 0.90 and color_hist_similarity >= 0.75:
        return True, "gray_color_channel_pass"
    return False, "not_passed"


def describe_variant(signals: Dict[str, float]) -> str:
    """根据相似度信号给出候选图片变体描述。"""
    phash_similarity = signals.get("phash_similarity", 0.0)
    resized_image_similarity = signals.get("resized_image_similarity", 0.0)
    grayscale_similarity = signals.get("grayscale_similarity", 0.0)
    color_hist_similarity = signals.get("color_hist_similarity", 0.0)
    clip_similarity = signals.get("clip_similarity", 0.0)
    has_resized_signal = isinstance(signals.get("resized_image_similarity"), (int, float))
    has_color_signal = isinstance(signals.get("color_hist_similarity"), (int, float))
    ocr_text_overlap = max(signals.get("ocr_content_overlap", 0.0), signals.get("ocr_text_overlap", 0.0))
    retrieved_text_overlap = signals.get("retrieved_text_overlap", 0.0)
    text_semantic_similarity = signals.get("retrieved_text_semantic_similarity", 0.0) * 0.5

    if phash_similarity >= 0.95:
        return "疑似完全一致或仅轻微压缩"
    if phash_similarity >= 0.84 and max(ocr_text_overlap, retrieved_text_overlap, text_semantic_similarity) < 0.35:
        if has_resized_signal and resized_image_similarity < 0.85:
            return "pHash 较高但像素/文字差异明显，可能是裁剪、压缩或文字替换版本"
        return "pHash 较高但文字内容差异明显，疑似改字或再传播变体"
    if phash_similarity >= 0.80 and has_resized_signal and resized_image_similarity >= 0.85:
        return "疑似同图等尺寸比较高度一致"
    if has_color_signal and grayscale_similarity >= 0.88 and color_hist_similarity < 0.55:
        return "结构高度相似但颜色差异明显，可能是调色、滤镜或黑白化变体"
    if has_color_signal and grayscale_similarity >= 0.88 and color_hist_similarity >= 0.75:
        return "灰度结构和颜色分布均较一致"
    if phash_similarity >= 0.80 and clip_similarity >= 0.85:
        if ocr_text_overlap >= 0.60:
            return "疑似同图裁剪、水印或尺寸变化，图中文字高度一致"
        return "疑似同图裁剪、水印或尺寸变化"
    if clip_similarity >= 0.85:
        if max(ocr_text_overlap, retrieved_text_overlap, text_semantic_similarity) >= 0.60:
            return "语义和主体高度一致，且文字内容相近"
        return "语义和主体高度一致，可能存在明显编辑变体"
    if phash_similarity >= 0.72 or grayscale_similarity >= 0.75 or resized_image_similarity >= 0.75:
        return "存在一定视觉相似证据，但不足以判定为强同图"
    return "视觉相似度不足"


def detect_tampering_signals(signals: Dict[str, Any], image_variant: str) -> Dict[str, Any]:
    """基于已有视觉/OCR/水印信号标记疑似改图或同图不同语境。"""
    tampering_signals: List[str] = []
    phash_similarity = float(signals.get("phash_similarity", 0.0) or 0.0)
    resized_similarity = float(signals.get("resized_image_similarity", 0.0) or 0.0)
    grayscale_similarity = float(signals.get("grayscale_similarity", 0.0) or 0.0)
    color_hist_similarity = float(signals.get("color_hist_similarity", 0.0) or 0.0)
    clip_similarity = float(signals.get("clip_similarity", 0.0) or 0.0)
    has_resized_signal = isinstance(signals.get("resized_image_similarity"), (int, float))
    has_color_signal = isinstance(signals.get("color_hist_similarity"), (int, float))
    ocr_content_overlap = float(signals.get("ocr_content_overlap", 0.0) or 0.0)
    retrieved_text_overlap = float(signals.get("retrieved_text_overlap", 0.0) or 0.0)
    watermark_detected = bool(signals.get("watermark_detected"))

    if watermark_detected:
        tampering_signals.append("watermark_added_or_changed")
    if has_color_signal and grayscale_similarity >= 0.88 and color_hist_similarity < 0.55:
        tampering_signals.append("color_or_filter_changed")
    if has_resized_signal and phash_similarity >= 0.80 and resized_similarity < 0.85:
        tampering_signals.append("crop_resize_or_compression_changed")
    if clip_similarity >= 0.82 and phash_similarity < hash_weak_threshold():
        tampering_signals.append("semantic_same_but_structure_changed")
    if max(grayscale_similarity, clip_similarity, phash_similarity) >= 0.82 and 0.0 < ocr_content_overlap < 0.35:
        tampering_signals.append("content_text_changed")
    if max(grayscale_similarity, clip_similarity, phash_similarity) >= 0.82 and 0.0 < retrieved_text_overlap < 0.25:
        tampering_signals.append("retrieved_context_changed")
    if signals.get("tampering_candidate_passed"):
        reason = str(signals.get("tampering_candidate_reason") or "")
        if reason.startswith("visual_preserved_for_tampering"):
            tampering_signals.append("content_text_changed")
        elif reason.startswith("text_preserved_for_tampering"):
            tampering_signals.append("visual_variant_detected")
        else:
            tampering_signals.append("possible_tampering_candidate")
    if any(keyword in image_variant for keyword in ("裁剪", "水印", "调色", "滤镜", "黑白化", "编辑变体")):
        tampering_signals.append("visual_variant_detected")

    tampering_signals = unique_values(tampering_signals)
    suspected = bool(tampering_signals)
    reason = (
        "候选图与目标图存在同源或近似证据，同时出现水印、文本、颜色、裁剪或结构变化。"
        if suspected
        else "未发现明确的改图、篡改或同图不同语境信号。"
    )
    return {
        "suspected_tampering": suspected,
        "tampering_signals": tampering_signals,
        "tampering_reason": reason,
    }


def validation_decision_kind(
    similarity: float,
    is_similar: bool,
    layered_filter_passed: bool,
    layered_filter_stage: str,
    text_filter_passed: bool,
    tampering_candidate_passed: bool,
    tampering_candidate_reason: str,
) -> Tuple[str, str]:
    if tampering_candidate_passed and not (
        (similarity >= similarity_threshold() or layered_filter_passed) and text_filter_passed
    ):
        return "tampering_candidate_preserved", tampering_candidate_reason
    if is_similar and similarity >= similarity_threshold() and layered_filter_passed:
        return "weighted_and_layered_pass", layered_filter_stage
    if is_similar and similarity >= similarity_threshold():
        return "weighted_similarity_pass", "similarity_threshold"
    if is_similar and layered_filter_passed:
        return "layered_visual_pass", layered_filter_stage
    if not text_filter_passed:
        return "rejected_text_filter_failed", "text_filter"
    if tampering_candidate_reason.startswith("text_only_visual_too_low"):
        return "rejected_text_only_visual_too_low", tampering_candidate_reason
    return "rejected_low_similarity", "not_passed"


def classify_tampering_type(
    signals: Dict[str, Any],
    image_variant: str,
    tampering_signals: List[str],
) -> str:
    """把底层 tampering_signals 压成 Analyzer 易消费的单一类型。"""
    if not tampering_signals:
        return "none"

    phash_similarity = float(signals.get("phash_similarity", 0.0) or 0.0)
    visual_score = signal_visual_score(signals)
    text_score = signal_text_score(signals)
    reason = str(signals.get("tampering_candidate_reason") or "")

    if "watermark_added_or_changed" in tampering_signals:
        return "watermark_changed"
    if "content_text_changed" in tampering_signals and visual_score >= 0.82:
        return "text_replacement_or_recontextualized"
    if "semantic_same_but_structure_changed" in tampering_signals:
        return "visual_remix"
    if "color_or_filter_changed" in tampering_signals:
        return "color_or_filter_variant"
    if "crop_resize_or_compression_changed" in tampering_signals:
        return "cropped_or_recompressed"
    if reason.startswith("text_preserved_for_tampering") and text_score >= tampering_text_signal_threshold():
        return "text_match_visual_changed"
    if phash_similarity >= hash_weak_threshold() or visual_score >= 0.75:
        return "weak_visual_variant"
    if any(keyword in image_variant for keyword in ("同图不同", "文字差异", "改字")):
        return "same_image_different_context"
    return "possible_tampering"


def duplicate_cluster_id_for(node_id: Any) -> str:
    """生成稳定重复组 ID；只基于节点 ID，避免引入传播排序判断。"""
    normalized = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(node_id or "unknown")).strip("_")
    return f"dup_{normalized or 'unknown'}"


def build_target_topology_hints(
    node: Dict[str, Any],
    signals: Dict[str, Any],
    similarity: float,
    is_similar: bool,
    decision_type: str,
    decision_stage: str,
    tampering_type: str,
    tampering_signals: List[str],
) -> List[Dict[str, Any]]:
    """把目标图到候选图的视觉关系整理成拓扑边提示。"""
    if not is_similar:
        return []

    node_id = node.get("id", "unknown")
    if decision_type == "tampering_candidate_preserved":
        relation_type = "target_visual_variant"
    elif tampering_type != "none":
        relation_type = "target_visual_variant"
    else:
        relation_type = "target_visual_match"

    evidence = unique_values(
        [
            decision_type,
            decision_stage,
            *tampering_signals,
        ]
    )
    hint: Dict[str, Any] = {
        "relation_type": relation_type,
        "source_id": "target_image",
        "target_id": node_id,
        "confidence": round(similarity, 4),
        "evidence": [item for item in evidence if item],
    }
    if tampering_type != "none":
        hint["variant_type"] = tampering_type
    if isinstance(signals.get("phash_similarity"), (int, float)):
        hint["phash_similarity"] = round(float(signals["phash_similarity"]), 4)
    if isinstance(signals.get("clip_similarity"), (int, float)):
        hint["clip_similarity"] = round(float(signals["clip_similarity"]), 4)
    if isinstance(signals.get("ocr_content_overlap"), (int, float)):
        hint["ocr_content_overlap"] = round(float(signals["ocr_content_overlap"]), 4)
    return [hint]


def build_validator_evidence(
    signals: Dict[str, Any],
    similarity: float,
    decision_type: str,
    decision_stage: str,
    image_variant: str,
    suspected_tampering: bool,
    tampering_signals: List[str],
) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "decision_type": decision_type,
        "decision_stage": decision_stage,
        "similarity": round(similarity, 4),
        "visual_score": round(signal_visual_score(signals), 4),
        "text_score": round(signal_text_score(signals), 4),
        "phash_similarity": signals.get("phash_similarity"),
        "clip_similarity": signals.get("clip_similarity"),
        "ocr_content_overlap": signals.get("ocr_content_overlap"),
        "retrieved_text_overlap": signals.get("retrieved_text_overlap"),
        "image_variant": image_variant,
        "suspected_tampering": suspected_tampering,
        "tampering_signals": tampering_signals,
        "watermark_detected": bool(signals.get("watermark_detected")),
        "watermark_platforms": signals.get("watermark_platforms", []),
        "watermark_accounts": signals.get("watermark_accounts", {}),
        "watermark_text": signals.get("watermark_text", []),
        "watermark_source": signals.get("watermark_source", "none"),
        "watermark_ocr_used": bool(signals.get("watermark_ocr_used")),
        "candidate_content_text": signals.get("candidate_content_text", ""),
        "ocr_skipped": bool(signals.get("ocr_skipped")),
        "ocr_skip_reason": signals.get("ocr_skip_reason", ""),
        "tampering_type": signals.get("tampering_type"),
        "topology_hints": signals.get("topology_hints", []),
    }
    return {key: value for key, value in evidence.items() if value is not None}


def make_validation_decision(signals: Dict[str, Any]) -> Dict[str, Any]:
    """基于全部信号集中生成保留/拒绝、解释和 LLM 触发策略。"""
    numeric_signals = {
        key: value
        for key, value in signals.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    similarity = weighted_similarity(numeric_signals)
    layered_filter_passed, layered_filter_stage = layered_visual_filter(numeric_signals)
    text_filter_passed = passes_text_filter(numeric_signals)
    tampering_candidate_passed, tampering_candidate_reason = tampering_candidate_filter(numeric_signals)
    enriched_signals = {
        **signals,
        "layered_filter_passed": layered_filter_passed,
        "layered_filter_stage": layered_filter_stage,
        "text_filter_passed": text_filter_passed,
        "tampering_candidate_passed": tampering_candidate_passed,
        "tampering_candidate_reason": tampering_candidate_reason,
    }
    rule_passed = (similarity >= similarity_threshold() or layered_filter_passed) and text_filter_passed
    is_similar = rule_passed or tampering_candidate_passed
    enriched_signals["rule_passed"] = rule_passed
    image_variant = describe_variant(numeric_signals)
    validation_reason = (
        "疑似篡改候选：图像或文本存在单路强证据，保留给 Analyzer 复核"
        if tampering_candidate_passed and not rule_passed
        else (
        "可用视觉信号融合相似度通过阈值"
        if is_similar
        else (
            "文本二次过滤未通过，判定为误召回"
            if not text_filter_passed
            else "融合相似度低于阈值且层级过滤未通过，判定为误召回"
        )
        )
    )
    tampering = detect_tampering_signals(enriched_signals, image_variant)
    decision_type, decision_stage = validation_decision_kind(
        similarity=similarity,
        is_similar=is_similar,
        layered_filter_passed=layered_filter_passed,
        layered_filter_stage=layered_filter_stage,
        text_filter_passed=text_filter_passed,
        tampering_candidate_passed=tampering_candidate_passed,
        tampering_candidate_reason=tampering_candidate_reason,
    )
    enriched_signals["validation_decision_type"] = decision_type
    enriched_signals["validation_decision_stage"] = decision_stage
    should_run_llm, llm_trigger = should_run_validator_llm(is_similar, similarity, enriched_signals)
    return {
        "signals": enriched_signals,
        "similarity": similarity,
        "is_similar": is_similar,
        "image_variant": image_variant,
        "validation_reason": validation_reason,
        "validation_decision_type": decision_type,
        "validation_decision_stage": decision_stage,
        **tampering,
        "should_run_llm": should_run_llm,
        "llm_trigger": llm_trigger,
    }


def should_run_validator_llm(
    is_similar: bool,
    similarity: float,
    signals: Dict[str, Any],
) -> Tuple[bool, str]:
    """只让视觉 LLM 处理复杂图片语义生成，避免对所有疑似篡改节点批量调用。"""
    phash_similarity = float(signals.get("phash_similarity", 0.0) or 0.0)
    resized_similarity = float(signals.get("resized_image_similarity", 0.0) or 0.0)
    grayscale_similarity = float(signals.get("grayscale_similarity", 0.0) or 0.0)
    color_hist_similarity = float(signals.get("color_hist_similarity", 0.0) or 0.0)
    clip_similarity = float(signals.get("clip_similarity", 0.0) or 0.0)
    image_complexity = float(signals.get("candidate_image_complexity", 0.0) or 0.0)
    text_scores = [
        float(signals[name])
        for name in (
            "ocr_content_overlap",
            "ocr_text_overlap",
            "retrieved_text_overlap",
        )
        if isinstance(signals.get(name), (int, float))
    ]
    if isinstance(signals.get("retrieved_text_semantic_similarity"), (int, float)):
        text_scores.append(float(signals["retrieved_text_semantic_similarity"]) * 0.5)
    text_similarity = max(text_scores, default=0.0)
    has_text_signal = bool(text_scores)
    visual_score = max(
        phash_similarity,
        grayscale_similarity,
        min(grayscale_similarity, color_hist_similarity),
        clip_similarity,
    )
    boundary_floor = min(
        llm_boundary_similarity_floor(),
        max(0.0, similarity_threshold() - 0.10),
    )
    threshold = similarity_threshold()
    tampering_reason = str(signals.get("tampering_candidate_reason") or "")
    tampering_preserved = bool(signals.get("tampering_candidate_passed")) and (
        tampering_reason.startswith("visual_preserved_for_tampering")
        or tampering_reason.startswith("text_preserved_for_tampering")
    )

    if image_complexity < validator_llm_complexity_threshold():
        return False, f"not_complex_image:complexity={image_complexity:.2f}"

    if tampering_preserved and visual_score >= 0.82 and has_text_signal and text_similarity < 0.20:
        return True, f"complex_image_semantic_caption:{tampering_reason or 'tampering_candidate'}"
    if clip_similarity >= 0.82 and phash_similarity < hash_weak_threshold():
        return True, "complex_image_semantic_caption:clip_high_phash_low"
    if visual_score >= 0.88 and has_text_signal and text_similarity < 0.20:
        return True, "complex_image_semantic_caption:visual_text_conflict"
    if signals.get("layered_filter_passed") and not signals.get("text_filter_passed", True):
        return True, "complex_image_semantic_caption:layered_visual_pass_text_failed"
    if boundary_floor <= similarity < threshold:
        return True, "complex_image_semantic_caption:near_similarity_threshold"

    strong_rule_passed = bool(signals.get("rule_passed")) and (
        (phash_similarity >= hash_strong_threshold() and resized_similarity >= 0.85)
        or (phash_similarity >= 0.88 and grayscale_similarity >= 0.88)
        or (similarity >= min(1.0, threshold + 0.08) and visual_score >= 0.88)
    )
    if is_similar and strong_rule_passed:
        return False, "strong_validated_candidate_no_llm"
    if is_similar and similarity <= min(1.0, threshold + 0.05):
        return True, "complex_image_semantic_caption:validated_near_threshold"
    if is_similar:
        return False, "validated_candidate_no_boundary_no_llm"
    return False, "not_validated_or_boundary"


def normalize_url(url: str) -> str:
    """标准化 URL，用于消除跟踪参数、fragment 和大小写差异。"""
    if not url:
        return ""

    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip().rstrip("/")

    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered_key = key.lower()
        if lowered_key.startswith("utm_") or lowered_key in TRACKING_QUERY_KEYS:
            continue
        query_items.append((key, value))

    normalized_path = parsed.path.rstrip("/") or "/"
    normalized_query = urlencode(sorted(query_items), doseq=True)
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            normalized_path,
            "",
            normalized_query,
            "",
        )
    )


def normalize_title(title: str) -> str:
    """标准化标题，辅助识别同站点重复内容。"""
    title = re.sub(r"\s+", " ", title or "").strip().lower()
    return re.sub(r"[|｜_\-—]+", " ", title)


def get_domain(url: str) -> str:
    """提取 URL 域名。"""
    parsed = urlparse(url)
    return parsed.netloc.lower()


def build_dedup_keys(node: Dict[str, Any]) -> List[str]:
    """构建强确定去重键；疑似重复只标记不合并，避免吞掉潜在源头。"""
    keys: List[str] = []

    page_url = normalize_url(node.get("url", ""))
    if page_url:
        keys.append(f"url:{page_url}")

    if not page_url:
        for field_name in ("cached_image_path", "local_image_path", "image_url", "thumbnail_url"):
            value = node.get(field_name)
            if not value:
                continue
            if field_name.endswith("path"):
                keys.append(f"{field_name}:{Path(value).as_posix().lower()}")
            else:
                keys.append(f"{field_name}:{normalize_url(value)}")

    if not keys:
        keys.append(f"id:{node.get('id', 'unknown')}")
    return keys


def build_possible_duplicate_keys(node: Dict[str, Any]) -> List[str]:
    """构建弱重复线索，只用于标记，不用于合并。"""
    keys: List[str] = []
    page_url = normalize_url(node.get("url", ""))
    title = normalize_title(node.get("title", ""))
    domain = get_domain(page_url)
    if domain and title:
        keys.append(f"domain_title:{domain}|{title}")
    return keys


def node_dedup_text(node: Dict[str, Any]) -> str:
    """汇总用于联合去重的网页文本和 OCR 文本。"""
    signals = node.get("validation_signals", {})
    parts = [
        collect_text_context(node),
        str(signals.get("candidate_content_text") or ""),
        str(signals.get("retrieved_text_excerpt") or ""),
    ]
    return "\n".join(part for part in parts if part.strip())


def can_joint_deduplicate(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    """联合去重只在同站点或无页面 URL 场景使用，避免破坏跨域传播链。"""
    left_url = normalize_url(str(left.get("url") or ""))
    right_url = normalize_url(str(right.get("url") or ""))
    left_domain = get_domain(left_url)
    right_domain = get_domain(right_url)
    if left_domain and right_domain:
        return left_domain == right_domain
    return not left_url and not right_url


def joint_duplicate_score(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """图片 pHash 和文本相似度联合判断重复，文本权重更高。"""
    if not can_joint_deduplicate(left, right):
        return {
            "joint_score": 0.0,
            "image_similarity": 0.0,
            "text_similarity": 0.0,
            "same_image_different_text": False,
            "is_duplicate": False,
            "skipped_reason": "cross_domain_or_url_boundary",
        }

    left_signals = left.get("validation_signals", {})
    right_signals = right.get("validation_signals", {})
    image_similarity = compute_hash_string_similarity(
        str(left_signals.get("candidate_phash") or ""),
        str(right_signals.get("candidate_phash") or ""),
    )
    text_similarity = compute_text_overlap(node_dedup_text(left), node_dedup_text(right))
    joint_score = (
        text_similarity * JOINT_DEDUP_TEXT_WEIGHT
        + image_similarity * JOINT_DEDUP_IMAGE_WEIGHT
    )
    same_image_different_text = (
        image_similarity >= hash_strong_threshold()
        and text_similarity < same_image_text_different_threshold()
    )
    return {
        "joint_score": round(joint_score, 4),
        "image_similarity": round(image_similarity, 4),
        "text_similarity": round(text_similarity, 4),
        "same_image_different_text": same_image_different_text,
        "is_duplicate": (
            joint_score >= joint_dedup_threshold()
            and not same_image_different_text
        ),
    }


def unique_values(values: List[Any]) -> List[Any]:
    """按出现顺序去重。"""
    result: List[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def prepare_analyzer_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """输出给 Analyzer 的节点默认去掉大块调试字段，保留核心证据。"""
    if include_debug_signals():
        return node
    debug_keys = {
        "validation_signals",
        "_dedup_order",
        "dedup_keys",
        "possible_duplicate_keys",
        "source_types",
        "merged_urls",
    }
    return {key: value for key, value in node.items() if key not in debug_keys}


def sort_nodes_for_analyzer(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 Validator 视觉置信度排序，不混入传播拓扑判断。"""
    return sorted(
        nodes,
        key=lambda node: (
            -float(node.get("similarity", 0.0) or 0.0),
            int(node.get("_dedup_order", 999999) or 999999),
        ),
    )


def run_global_validator_llm_reviews(
    nodes: List[Dict[str, Any]],
    target_image: Image.Image,
    llm_client: Optional[ValidatorMultimodalLLMClient],
    max_nodes: int,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """在所有候选完成规则校验和排序后，择优调用视觉 LLM。"""
    if llm_client is None or not llm_client.enabled:
        return nodes, 0, 0, 0

    reviewed_nodes: List[Dict[str, Any]] = []
    used_count = 0
    skipped_count = 0
    error_count = 0
    attempted_count = 0

    for node in nodes:
        if node.get("llm_status") != "pending":
            reviewed_nodes.append(node)
            if node.get("llm_status") == "skipped":
                skipped_count += 1
            elif node.get("llm_status") == "error":
                error_count += 1
            continue

        if max_nodes > 0 and attempted_count >= max_nodes:
            updated = {
                **node,
                "llm_status": "skipped",
                "llm_reason": "llm_budget_exhausted，跳过 Validator LLM 以控制成本",
                "llm_used": False,
                "llm_trigger": "llm_budget_exhausted",
            }
            validator_evidence = dict(updated.get("validator_evidence", {}))
            validator_evidence.update(
                {
                    "llm_status": "skipped",
                    "llm_used": False,
                    "llm_trigger": "llm_budget_exhausted",
                }
            )
            updated["validator_evidence"] = validator_evidence
            reviewed_nodes.append(updated)
            skipped_count += 1
            continue

        attempted_count += 1
        try:
            candidate_image = get_candidate_image(node, target_image)
            llm_result = llm_client.review(
                target_image=target_image,
                candidate_image=candidate_image,
                node=node,
                validation_signals=node.get("validation_signals", {}),
                similarity=float(node.get("similarity", 0.0) or 0.0),
                is_similar=bool(node.get("is_similar")),
                image_variant=str(node.get("image_variant") or ""),
                validation_reason=str(node.get("validation_reason") or ""),
            )
            updated = {
                **node,
                "llm_status": llm_result.get("llm_status"),
                "llm_reason": llm_result.get("llm_reason"),
                "llm_used": llm_result.get("llm_status") == "used",
            }
            if llm_result.get("semantic_caption"):
                updated["semantic_caption"] = llm_result["semantic_caption"]
                updated["llm_validation"] = llm_result["semantic_caption"]
            if llm_result.get("llm_raw_excerpt"):
                updated["llm_raw_excerpt"] = llm_result["llm_raw_excerpt"]
            if updated["llm_status"] == "used":
                used_count += 1
            elif updated["llm_status"] == "error":
                error_count += 1
            else:
                skipped_count += 1
        except Exception as exc:
            updated = {
                **node,
                "llm_status": "error",
                "llm_reason": str(exc),
                "llm_used": False,
            }
            error_count += 1

        validator_evidence = dict(updated.get("validator_evidence", {}))
        validator_evidence.update(
            {
                "llm_status": updated.get("llm_status"),
                "llm_used": updated.get("llm_used", False),
                "llm_trigger": updated.get("llm_trigger", ""),
            }
        )
        updated["validator_evidence"] = validator_evidence
        reviewed_nodes.append(updated)

    return reviewed_nodes, used_count, skipped_count, error_count


def merge_duplicate_node(existing: Dict[str, Any], duplicate: Dict[str, Any]) -> Dict[str, Any]:
    """只合并强确定重复候选；保留首次出现节点作为主节点。"""
    representative_id = existing.get("representative_node_id") or existing.get("id")
    duplicate_cluster_id = existing.get("duplicate_cluster_id") or duplicate_cluster_id_for(representative_id)
    merged_from = unique_values(
        [
            *existing.get("merged_from", [existing.get("id")]),
            *duplicate.get("merged_from", [duplicate.get("id")]),
        ]
    )
    dedup_keys = unique_values(
        [
            *existing.get("dedup_keys", build_dedup_keys(existing)),
            *duplicate.get("dedup_keys", build_dedup_keys(duplicate)),
        ]
    )
    merged_urls = unique_values(
        [
            *existing.get("merged_urls", [existing.get("url")]),
            *duplicate.get("merged_urls", [duplicate.get("url")]),
        ]
    )
    source_types = unique_values(
        [
            *existing.get("source_types", [existing.get("source_type")]),
            *duplicate.get("source_types", [duplicate.get("source_type")]),
        ]
    )
    topology_hints = unique_values(
        [
            *existing.get("topology_hints", []),
            {
                "relation_type": "exact_duplicate",
                "source_id": representative_id,
                "target_id": duplicate.get("id"),
                "confidence": 1.0,
                "evidence": build_dedup_keys(duplicate),
            },
        ]
    )
    validator_evidence = dict(existing.get("validator_evidence", {}))
    validator_evidence["topology_hints"] = topology_hints
    validator_evidence["duplicate_cluster_id"] = duplicate_cluster_id
    validator_evidence["duplicate_relation"] = "exact_duplicate"
    validator_evidence["representative_node_id"] = representative_id

    return {
        **existing,
        "dedup_keys": [key for key in dedup_keys if key],
        "duplicate_cluster_id": duplicate_cluster_id,
        "duplicate_relation": "exact_duplicate",
        "representative_node_id": representative_id,
        "duplicate_count": len([item for item in merged_from if item]),
        "merged_from": [item for item in merged_from if item],
        "merged_urls": [item for item in merged_urls if item],
        "source_types": [item for item in source_types if item],
        "topology_hints": topology_hints,
        "validator_evidence": validator_evidence,
        "dedup_strategy": "exact_url_or_exact_image_reference",
        "primary_selected_by": "first_seen_exact_duplicate",
    }


def annotate_possible_duplicate(
    node: Dict[str, Any],
    existing_node: Dict[str, Any],
    signals: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    """标记疑似重复但不合并，避免在源头不确定时丢失候选。"""
    representative_id = existing_node.get("representative_node_id") or existing_node.get("id")
    duplicate_cluster_id = existing_node.get("duplicate_cluster_id") or duplicate_cluster_id_for(representative_id)
    confidence = float(
        signals.get("joint_score")
        or signals.get("image_similarity")
        or signals.get("text_similarity")
        or 0.0
    )
    possible_duplicates = list(node.get("possible_duplicates", []))
    possible_duplicates.append(
        {
            "id": existing_node.get("id"),
            "url": existing_node.get("url"),
            "reason": reason,
            "signals": signals,
        }
    )
    topology_hints = unique_values(
        [
            *node.get("topology_hints", []),
            {
                "relation_type": reason,
                "source_id": representative_id,
                "target_id": node.get("id"),
                "confidence": round(confidence, 4),
                "evidence": [reason],
                "signals": signals,
            },
        ]
    )
    validator_evidence = dict(node.get("validator_evidence", {}))
    validator_evidence["topology_hints"] = topology_hints
    validator_evidence["duplicate_cluster_id"] = duplicate_cluster_id
    validator_evidence["duplicate_relation"] = reason
    validator_evidence["duplicate_confidence"] = round(confidence, 4)
    validator_evidence["representative_node_id"] = representative_id
    return {
        **node,
        "possible_duplicate": True,
        "possible_duplicates": possible_duplicates,
        "duplicate_cluster_id": duplicate_cluster_id,
        "duplicate_relation": reason,
        "duplicate_confidence": round(confidence, 4),
        "representative_node_id": representative_id,
        "topology_hints": topology_hints,
        "validator_evidence": validator_evidence,
    }


def deduplicate_validated_nodes(nodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    """只合并强确定重复；弱重复保留并打标，避免破坏传播链。"""
    deduplicated_nodes: List[Dict[str, Any]] = []
    key_to_index: Dict[str, int] = {}
    possible_key_to_index: Dict[str, int] = {}
    merged_count = 0
    possible_duplicate_count = 0

    for original_index, node in enumerate(nodes):
        node = {**node, "_dedup_order": original_index}
        keys = build_dedup_keys(node)
        matched_index = next((key_to_index[key] for key in keys if key in key_to_index), None)

        if matched_index is None:
            possible_keys = build_possible_duplicate_keys(node)
            node_with_dedup = {
                **node,
                "dedup_keys": keys,
                "possible_duplicate_keys": possible_keys,
                "duplicate_cluster_id": duplicate_cluster_id_for(node.get("id")),
                "duplicate_relation": "self",
                "representative_node_id": node.get("id"),
                "duplicate_count": 1,
                "merged_from": [node.get("id")],
                "merged_urls": [node.get("url")] if node.get("url") else [],
                "source_types": [node.get("source_type")] if node.get("source_type") else [],
            }
            possible_match_index = next(
                (possible_key_to_index[key] for key in possible_keys if key in possible_key_to_index),
                None,
            )
            possible_marked = False
            if possible_match_index is not None:
                node_with_dedup["possible_duplicate_group_key"] = possible_keys[0]

            for existing_node in deduplicated_nodes:
                joint_info = joint_duplicate_score(existing_node, node_with_dedup)
                existing_keys = set(existing_node.get("possible_duplicate_keys", []))
                current_keys = set(node_with_dedup.get("possible_duplicate_keys", []))
                shared_group_keys = sorted(existing_keys & current_keys)
                if joint_info["is_duplicate"] or joint_info["same_image_different_text"]:
                    if shared_group_keys:
                        joint_info = {
                            **joint_info,
                            "possible_duplicate_group_keys": shared_group_keys,
                        }
                    node_with_dedup = annotate_possible_duplicate(
                        node_with_dedup,
                        existing_node,
                        joint_info,
                        (
                            "same_image_different_text"
                            if joint_info["same_image_different_text"]
                            else (
                                "same_domain_title_with_joint_similarity"
                                if shared_group_keys
                                else "joint_image_text_similarity"
                            )
                        ),
                    )
                    possible_marked = True
                    break

            if possible_marked:
                possible_duplicate_count += 1

            deduplicated_nodes.append(node_with_dedup)
            current_index = len(deduplicated_nodes) - 1
            for key in keys:
                key_to_index[key] = current_index
            for key in possible_keys:
                possible_key_to_index.setdefault(key, current_index)
            continue

        deduplicated_nodes[matched_index] = merge_duplicate_node(
            deduplicated_nodes[matched_index],
            node,
        )
        for key in deduplicated_nodes[matched_index].get("dedup_keys", []):
            key_to_index[key] = matched_index
        merged_count += 1

    return deduplicated_nodes, merged_count, possible_duplicate_count


def validate_candidate(
    node: Dict[str, Any],
    target_image: Image.Image,
    enable_clip: bool,
    enable_ocr: bool,
    llm_client: Optional[ValidatorMultimodalLLMClient] = None,
    llm_budget_available: bool = True,
    watermark_ocr_budget_available: bool = True,
    target_text_context: str = "",
    target_ocr_text: Optional[str] = None,
    target_ocr_analysis: Optional[Dict[str, Any]] = None,
    target_ocr_error: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool]:
    """校验单个候选节点，并返回更新后的节点和是否保留。"""
    try:
        candidate_image = get_candidate_image(node, target_image)
        signals: Dict[str, Any] = {}
        signals["candidate_image_complexity"] = compute_image_complexity(candidate_image)

        try:
            signals["phash_similarity"] = compute_phash_similarity(target_image, candidate_image)
            signals["candidate_phash"] = compute_image_phash(candidate_image)
        except Exception as exc:
            signals["phash_error"] = str(exc)

        try:
            signals["resized_image_similarity"] = compute_resized_image_similarity(target_image, candidate_image)
        except Exception as exc:
            signals["resized_image_error"] = str(exc)

        try:
            signals["grayscale_similarity"] = compute_grayscale_similarity(target_image, candidate_image)
        except Exception as exc:
            signals["grayscale_error"] = str(exc)

        try:
            signals["color_hist_similarity"] = compute_color_hist_similarity(target_image, candidate_image)
        except Exception as exc:
            signals["color_hist_error"] = str(exc)

        if enable_clip:
            try:
                signals["clip_similarity"] = compute_clip_similarity(target_image, candidate_image)
            except Exception as exc:  # CLIP 是增强层，失败时不阻断 pHash 主链路。
                signals["clip_error"] = str(exc)

        candidate_text_context = collect_text_context(node)
        if tokenize_text(target_text_context) and tokenize_text(candidate_text_context):
            text_overlap, text_details = compute_retrieved_text_similarity(target_text_context, node)
            signals["retrieved_text_overlap"] = text_overlap
            signals.update(text_details)
            if enable_clip:
                try:
                    signals["retrieved_text_semantic_similarity"] = compute_clip_text_similarity(
                        target_text_context,
                        candidate_text_context,
                    )
                except Exception as exc:
                    signals["text_semantic_error"] = str(exc)

        if enable_ocr:
            if target_ocr_error:
                signals["ocr_error"] = target_ocr_error
            else:
                run_ocr, ocr_prefilter_reason = should_run_candidate_ocr(signals)
                run_watermark_ocr = (
                    watermark_ocr_enabled()
                    and watermark_ocr_budget_available
                    and signal_visual_score(signals) >= watermark_ocr_visual_threshold()
                )
                signals["ocr_prefilter_passed"] = run_ocr
                signals["ocr_prefilter_reason"] = ocr_prefilter_reason
                signals["watermark_ocr_prefilter_passed"] = run_watermark_ocr
                signals["watermark_ocr_prefilter_reason"] = (
                    "visual_candidate"
                    if run_watermark_ocr
                    else (
                        "budget_exhausted"
                        if watermark_ocr_enabled() and not watermark_ocr_budget_available
                        else "below_visual_threshold_or_disabled"
                    )
                )
                if run_ocr:
                    try:
                        ocr_overlap, ocr_details = compute_ocr_similarity(
                            target_image,
                            candidate_image,
                            target_text=target_ocr_text,
                            target_analysis=target_ocr_analysis,
                            enable_watermark_ocr=run_watermark_ocr,
                        )
                        signals.update(ocr_details)
                        target_content_text = str(
                            (target_ocr_analysis or analyze_ocr_text(target_ocr_text or "")).get("ocr_content_text") or ""
                        )
                        if tokenize_text(target_content_text) and tokenize_text(ocr_details["candidate_content_text"]):
                            signals["ocr_content_overlap"] = ocr_overlap
                            signals["ocr_text_overlap"] = ocr_overlap
                        else:
                            signals["ocr_text_overlap_skipped"] = "target_or_candidate_has_no_text"
                    except Exception as exc:  # OCR 只作为弱信号，失败不阻断视觉校验。
                        signals["ocr_error"] = str(exc)
                else:
                    signals["ocr_skipped"] = True
                    signals["ocr_skip_reason"] = ocr_prefilter_reason
                    if run_watermark_ocr:
                        try:
                            candidate_watermark_blocks = watermark_ocr_blocks(candidate_image)
                            watermark_summary = detect_watermarks_from_blocks(
                                candidate_watermark_blocks,
                                candidate_image.size,
                            )
                            signals.update(
                                {
                                    **watermark_summary,
                                    "watermark_source": "watermark_ocr",
                                    "watermark_ocr_used": True,
                                    "watermark_ocr_provider_chain": watermark_ocr_provider_chain(),
                                    "watermark_evidence": {
                                        "source": "watermark_ocr",
                                        "confidence": max(
                                            [item.get("confidence", 0.0) for item in watermark_summary["watermarks"]],
                                            default=0.0,
                                        ),
                                    },
                                }
                            )
                        except Exception as exc:
                            signals["watermark_ocr_error"] = str(exc)

        decision = make_validation_decision(signals)
        signals = decision["signals"]
        similarity = float(decision["similarity"])
        is_similar = bool(decision["is_similar"])
        image_variant = str(decision["image_variant"])
        validation_reason = str(decision["validation_reason"])
        validation_decision_type = str(decision["validation_decision_type"])
        validation_decision_stage = str(decision["validation_decision_stage"])
        suspected_tampering = bool(decision["suspected_tampering"])
        tampering_signals = list(decision["tampering_signals"])
        tampering_type = classify_tampering_type(signals, image_variant, tampering_signals)
        signals["tampering_type"] = tampering_type
        topology_hints = build_target_topology_hints(
            node=node,
            signals=signals,
            similarity=similarity,
            is_similar=is_similar,
            decision_type=validation_decision_type,
            decision_stage=validation_decision_stage,
            tampering_type=tampering_type,
            tampering_signals=tampering_signals,
        )
        signals["topology_hints"] = topology_hints
        updated_node = {
            **node,
            "stage": "validated" if is_similar else "rejected",
            "similarity": round(similarity, 4),
            "is_similar": is_similar,
            "reason": node.get("reason") or validation_reason,
            "validation_signals": signals,
            "image_variant": image_variant,
            "validation_reason": validation_reason,
            "validation_decision_type": validation_decision_type,
            "validation_decision_stage": validation_decision_stage,
            "suspected_tampering": suspected_tampering,
            "tampering_signals": tampering_signals,
            "tampering_reason": decision["tampering_reason"],
            "tampering_type": tampering_type,
            "topology_hints": topology_hints,
            "validator_evidence": build_validator_evidence(
                signals=signals,
                similarity=similarity,
                decision_type=validation_decision_type,
                decision_stage=validation_decision_stage,
                image_variant=image_variant,
                suspected_tampering=suspected_tampering,
                tampering_signals=tampering_signals,
            ),
        }
        for watermark_key in (
            "watermark_detected",
            "watermarks",
            "watermark_platforms",
            "watermark_accounts",
            "watermark_text",
            "watermark_evidence",
            "watermark_source",
            "watermark_ocr_used",
            "watermark_ocr_provider_chain",
            "watermark_ocr_error",
        ):
            if watermark_key in signals:
                updated_node[watermark_key] = signals[watermark_key]
        if "candidate_content_text" in signals:
            updated_node["candidate_content_text"] = signals["candidate_content_text"]
        if "ocr_content_overlap" in signals:
            updated_node["ocr_content_overlap"] = signals["ocr_content_overlap"]
        should_run_llm = bool(decision["should_run_llm"])
        llm_trigger = str(decision["llm_trigger"])
        if llm_client is not None:
            updated_node["llm_trigger"] = llm_trigger
            updated_node["llm_status"] = "pending" if should_run_llm and llm_budget_available else "skipped"
            updated_node["llm_reason"] = (
                "waiting for Validator LLM boundary review"
                if should_run_llm and llm_budget_available
                else f"{llm_trigger}，跳过 Validator LLM 以控制成本"
            )
            updated_node["llm_used"] = False

        validator_evidence = dict(updated_node.get("validator_evidence", {}))
        if "llm_status" in updated_node:
            validator_evidence["llm_status"] = updated_node.get("llm_status")
            validator_evidence["llm_used"] = updated_node.get("llm_used", False)
            validator_evidence["llm_trigger"] = updated_node.get("llm_trigger", "")
        updated_node["validator_evidence"] = validator_evidence

        return (updated_node, is_similar)
    except (OSError, UnidentifiedImageError, ValueError, requests.RequestException) as exc:
        return (
            {
                **node,
                "stage": "rejected",
                "similarity": 0.0,
                "is_similar": False,
                "validation_signals": {},
                "image_variant": "无法读取候选图片",
                "validation_reason": str(exc),
                "reason": str(exc),
            },
            False,
        )


def validate_node(state: AgentState) -> AgentState:
    """视觉校验智能体节点：pHash 快筛、CLIP 复核、OCR 和多模态 LLM 增强。"""
    logs = append_log(state, "validate_node: 启动 pHash + CLIP + OCR + 多模态 LLM 视觉校验。")

    try:
        target_image = get_target_image(state)
    except (OSError, UnidentifiedImageError) as exc:
        error_log = make_log_line(f"validate_node: 目标图片读取失败，终止视觉校验：{exc}")
        print(error_log)
        return {"nodes_data": [], "execution_logs": [*logs, error_log]}

    if target_image is None:
        error_log = make_log_line("validate_node: target_image.local_path 为空，无法进行真实视觉校验。")
        print(error_log)
        return {"nodes_data": [], "validated_nodes": [], "rejected_nodes": [], "execution_logs": [*logs, error_log]}

    enable_clip = env_flag("VALIDATOR_ENABLE_CLIP", True)
    enable_ocr = env_flag("VALIDATOR_ENABLE_OCR", True)
    enable_multimodal_llm = env_flag("VALIDATOR_ENABLE_MULTIMODAL_LLM", True)
    llm_client: Optional[ValidatorMultimodalLLMClient] = None
    llm_reason = "disabled by VALIDATOR_ENABLE_MULTIMODAL_LLM"
    if enable_multimodal_llm:
        llm_client = ValidatorMultimodalLLMClient()
        llm_reason = llm_client.reason
        if not llm_client.enabled:
            llm_log = make_log_line(f"validate_node: Validator 多模态 LLM 不可用，降级跳过：{llm_reason}")
            print(llm_log)
            logs = [*logs, llm_log]

    target_text_context = collect_text_context(state.get("target_image", {}))
    target_ocr_text: Optional[str] = None
    target_ocr_analysis: Dict[str, Any] = analyze_ocr_text("")
    target_ocr_error: Optional[str] = None

    if enable_ocr:
        try:
            target_ocr_blocks = ocr_blocks(target_image)
            target_ocr_analysis = analyze_ocr_blocks(target_ocr_blocks, target_image.size)
            if watermark_ocr_enabled():
                try:
                    target_watermark_blocks = watermark_ocr_blocks(target_image)
                    target_watermark_summary = detect_watermarks_from_blocks(
                        target_watermark_blocks,
                        target_image.size,
                    )
                    target_ocr_analysis = merge_watermark_summary(
                        target_ocr_analysis,
                        target_watermark_summary,
                    )
                    target_watermark_source = str(target_ocr_analysis.get("watermark_source") or "none")
                    target_ocr_analysis["watermark_source"] = (
                        "watermark_ocr"
                        if target_watermark_source == "none"
                        else f"{target_watermark_source}+watermark_ocr"
                    )
                    target_ocr_analysis["watermark_ocr_used"] = True
                    target_ocr_analysis["watermark_ocr_provider_chain"] = watermark_ocr_provider_chain()
                except Exception as exc:
                    target_ocr_analysis["watermark_ocr_error"] = str(exc)
            target_ocr_text = str(target_ocr_analysis.get("ocr_text") or "")
            target_text_context = "\n".join(
                part for part in (target_text_context, target_ocr_analysis.get("ocr_content_text")) if part
            )
        except Exception as exc:
            target_ocr_error = str(exc)
            ocr_log = make_log_line(f"validate_node: 目标图 OCR 失败，文本校验降级：{exc}")
            print(ocr_log)
            logs = [*logs, ocr_log]

    validated_nodes: List[Dict[str, Any]] = []
    rejected_nodes: List[Dict[str, Any]] = []
    rejected_count = 0
    input_nodes = state.get("nodes_data", [])
    progress_interval = env_int("VALIDATOR_PROGRESS_INTERVAL", 10, 1)
    progress_log = make_log_line(
        "validate_node: 缓存状态 "
        f"image_cache={env_flag('VALIDATOR_ENABLE_IMAGE_CACHE', True)} "
        f"ocr_cache={env_flag('VALIDATOR_ENABLE_OCR_CACHE', True)} "
        f"cache_dir={cache_root()}"
    )
    print(progress_log)
    logs = [*logs, progress_log]
    progress_callback = state.get("_progress_callback") if isinstance(state, dict) else None
    total_nodes = len(input_nodes)
    watermark_ocr_call_count = 0
    llm_node_limit = max_validator_llm_nodes()
    watermark_ocr_node_limit = max_watermark_ocr_nodes()
    for index, node in enumerate(input_nodes, start=1):
        progress_log = make_log_line(
            f"validate_node: 处理候选 {index}/{total_nodes}，id={node.get('id', 'unknown')}"
        )
        print(progress_log)
        logs = [*logs, progress_log]
        if progress_callback:
            progress_callback("validate_progress", index, total_nodes, 0, progress_log)
        updated_node, is_similar = validate_candidate(
            node=node,
            target_image=target_image,
            enable_clip=enable_clip,
            enable_ocr=enable_ocr,
            llm_client=llm_client if enable_multimodal_llm and llm_client and llm_client.enabled else None,
            llm_budget_available=True,
            watermark_ocr_budget_available=(
                watermark_ocr_enabled()
                and (watermark_ocr_node_limit <= 0 or watermark_ocr_call_count < watermark_ocr_node_limit)
            ),
            target_text_context=target_text_context,
            target_ocr_text=target_ocr_text,
            target_ocr_analysis=target_ocr_analysis,
            target_ocr_error=target_ocr_error,
        )
        if (updated_node.get("validation_signals") or {}).get("watermark_ocr_used"):
            watermark_ocr_call_count += 1
        if is_similar:
            validated_nodes.append(updated_node)
        else:
            rejected_nodes.append(updated_node)
            rejected_count += 1

    deduplicated_nodes, merged_count, possible_duplicate_count = deduplicate_validated_nodes(validated_nodes)
    deduplicated_nodes = sort_nodes_for_analyzer(deduplicated_nodes)
    deduplicated_nodes, global_llm_used_count, global_llm_skipped_count, global_llm_error_count = (
        run_global_validator_llm_reviews(
            nodes=deduplicated_nodes,
            target_image=target_image,
            llm_client=llm_client if enable_multimodal_llm and llm_client and llm_client.enabled else None,
            max_nodes=llm_node_limit,
        )
    )
    processed_nodes = [*deduplicated_nodes, *rejected_nodes]
    llm_used_count = sum(1 for node in processed_nodes if node.get("llm_status") == "used")
    llm_skipped_count = sum(1 for node in processed_nodes if node.get("llm_status") == "skipped")
    llm_error_count = sum(1 for node in processed_nodes if node.get("llm_status") == "error")
    validated_llm_used_count = sum(1 for node in deduplicated_nodes if node.get("llm_status") == "used")
    validated_llm_error_count = sum(1 for node in deduplicated_nodes if node.get("llm_status") == "error")
    validated_llm_skipped_count = sum(1 for node in deduplicated_nodes if node.get("llm_status") == "skipped")
    ocr_candidate_run_count = sum(
        1 for node in processed_nodes if (node.get("validation_signals") or {}).get("ocr_prefilter_passed") is True
    )
    ocr_candidate_skipped_count = sum(
        1 for node in processed_nodes if (node.get("validation_signals") or {}).get("ocr_skipped") is True
    )
    decision_counts: Dict[str, int] = {}
    for node in processed_nodes:
        decision_type = str(node.get("validation_decision_type") or "unknown")
        decision_counts[decision_type] = decision_counts.get(decision_type, 0) + 1
    tampering_type_counts: Dict[str, int] = {}
    topology_hint_count = 0
    for node in deduplicated_nodes:
        tampering_type = str(node.get("tampering_type") or "unknown")
        topology_hint_count += len(node.get("topology_hints", []) or [])
        tampering_type_counts[tampering_type] = tampering_type_counts.get(tampering_type, 0) + 1
    summary_log = make_log_line(
        f"validate_node: 保留 {len(validated_nodes)} 个相似候选，过滤 {rejected_count} 个误召回。"
    )
    dedup_log = make_log_line(
        f"validate_node: 强确定去重合并 {merged_count} 个候选，"
        f"标记并保留 {possible_duplicate_count} 个疑似重复候选，输出 {len(deduplicated_nodes)} 个节点。"
    )
    print(summary_log)
    print(dedup_log)
    logs = [*logs, summary_log, dedup_log]
    analyzer_nodes = [prepare_analyzer_node(node) for node in deduplicated_nodes]
    analyzer_rejected_nodes = [prepare_analyzer_node(node) for node in rejected_nodes]

    # ── 保存 validator 结果到 output ──
    try:
        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_payload = {
            "timestamp": timestamp,
            "summary": {
                "input": len(input_nodes),
                "passed": len(validated_nodes),
                "rejected": rejected_count,
                "final": len(deduplicated_nodes),
                "merged": merged_count,
                "possible_dup": possible_duplicate_count,
            },
            "nodes": [
                {
                    "id": n.get("id"), "url": n.get("url"), "title": n.get("title"),
                    "similarity": n.get("similarity"), "source": n.get("source"),
                    "engine": n.get("engine"), "image_url": n.get("image_url"),
                    "possible_duplicate": n.get("possible_duplicate", False),
                    "reason": n.get("validation_reason"),
                }
                for n in deduplicated_nodes
            ],
        }
        output_path = output_dir / f"validator_result_{timestamp}.json"
        output_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        save_log = make_log_line(f"validator: 结果已保存至 {output_path}")
        print(save_log)
        logs = [*logs, save_log]
    except Exception as exc:
        print(f"validator: 保存结果失败: {exc}")

    return {
        "nodes_data": analyzer_nodes,
        "validated_nodes": analyzer_nodes,
        "rejected_nodes": analyzer_rejected_nodes,
        "validation_summary": {
            "threshold": similarity_threshold(),
            "input_count": len(input_nodes),
            "validated_count": len(validated_nodes),
            "rejected_count": rejected_count,
            "deduplicated_count": len(deduplicated_nodes),
            "merged_duplicate_count": merged_count,
            "possible_duplicate_count": possible_duplicate_count,
            "decision_counts": decision_counts,
            "tampering_type_counts": tampering_type_counts,
            "topology_hint_count": topology_hint_count,
            "clip_enabled": enable_clip,
            "ocr_enabled": enable_ocr,
            "ocr_provider_chain": ocr_provider_chain(),
            "ocr_prefilter_enabled": ocr_prefilter_enabled(),
            "ocr_prefilter_visual_threshold": ocr_prefilter_visual_threshold(),
            "ocr_prefilter_text_threshold": ocr_prefilter_text_threshold(),
            "ocr_candidate_run_count": ocr_candidate_run_count,
            "ocr_candidate_skipped_count": ocr_candidate_skipped_count,
            "watermark_ocr_enabled": watermark_ocr_enabled(),
            "watermark_ocr_provider_chain": watermark_ocr_provider_chain(),
            "watermark_ocr_visual_threshold": watermark_ocr_visual_threshold(),
            "watermark_ocr_max_nodes": watermark_ocr_node_limit,
            "watermark_ocr_used_count": watermark_ocr_call_count,
            "analyzer_debug_signals_included": include_debug_signals(),
            "image_cache_enabled": env_flag("VALIDATOR_ENABLE_IMAGE_CACHE", True),
            "ocr_cache_enabled": env_flag("VALIDATOR_ENABLE_OCR_CACHE", True),
            "clip_feature_cache_enabled": clip_feature_cache_enabled(),
            "clip_feature_cache_image_count": len(CLIP_IMAGE_FEATURE_CACHE),
            "clip_feature_cache_text_count": len(CLIP_TEXT_FEATURE_CACHE),
            "clip_runtime_error": CLIP_RUNTIME_FAILURE["reason"],
            "cache_dir": str(cache_root()),
            "multimodal_llm_enabled": enable_multimodal_llm,
            "multimodal_llm_available": bool(llm_client and llm_client.enabled),
            "multimodal_llm_model": llm_client.model if llm_client else "",
            "multimodal_llm_reason": llm_reason,
            "multimodal_llm_policy": "complex_image_semantic_caption_boundary_gated",
            "multimodal_llm_boundary_similarity_floor": llm_boundary_similarity_floor(),
            "multimodal_llm_complexity_threshold": validator_llm_complexity_threshold(),
            "multimodal_llm_max_nodes": llm_node_limit,
            "multimodal_llm_used_count": llm_used_count,
            "multimodal_llm_skipped_count": llm_skipped_count,
            "multimodal_llm_error_count": llm_error_count,
            "global_llm_used_count": global_llm_used_count,
            "global_llm_skipped_count": global_llm_skipped_count,
            "global_llm_error_count": global_llm_error_count,
            "validated_llm_used_count": validated_llm_used_count,
            "validated_llm_skipped_count": validated_llm_skipped_count,
            "validated_llm_error_count": validated_llm_error_count,
            "target_text_available": bool(tokenize_text(target_text_context)),
            "target_text_excerpt": compact_text(target_text_context),
            "target_ocr_text": compact_text(target_ocr_text or ""),
            "target_content_text": target_ocr_analysis.get("ocr_content_text", ""),
            "target_watermark_detected": target_ocr_analysis.get("watermark_detected", False),
            "target_watermarks": target_ocr_analysis.get("watermarks", []),
            "target_watermark_platforms": target_ocr_analysis.get("watermark_platforms", []),
            "target_watermark_accounts": target_ocr_analysis.get("watermark_accounts", {}),
            "target_watermark_text": target_ocr_analysis.get("watermark_text", []),
            "target_watermark_source": target_ocr_analysis.get("watermark_source", "none"),
            "target_watermark_ocr_used": bool(target_ocr_analysis.get("watermark_ocr_used")),
            "target_watermark_ocr_provider_chain": target_ocr_analysis.get("watermark_ocr_provider_chain", []),
            "target_watermark_ocr_error": target_ocr_analysis.get("watermark_ocr_error", ""),
            "layered_filter": {
                "hash_strong_threshold": hash_strong_threshold(),
                "hash_weak_threshold": hash_weak_threshold(),
                "clip_review_threshold": clip_review_threshold(),
            },
            "joint_dedup": {
                "threshold": joint_dedup_threshold(),
                "text_weight": JOINT_DEDUP_TEXT_WEIGHT,
                "image_weight": JOINT_DEDUP_IMAGE_WEIGHT,
                "same_image_different_text_threshold": same_image_text_different_threshold(),
                "merge_policy": "exact_url_or_exact_image_reference_only",
            },
        },
        "execution_logs": logs,
    }

#仅用于本地调试
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    target_path = project_root / "TEST_DATA" / "input_data" / "TEST.jpg"
    captured_path = project_root / "TEST_DATA" / "input_data" / "_mitmproxy_captured.json"
    nodes_data: List[Dict[str, Any]] = []
    if captured_path.exists():
        captured = json.loads(captured_path.read_text(encoding="utf-8"))
        nodes_data = list(captured.get("results", []))[:5]

    debug_state: AgentState = {
        "target_image": {"local_path": str(target_path)} if target_path.exists() else {},
        "nodes_data": nodes_data,
        "execution_logs": [],
    }

    result = validate_node(debug_state)
    print(json.dumps(result, ensure_ascii=False, indent=2))
