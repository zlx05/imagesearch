"""SerpAPI Google Lens 单独测试"""
import streamlit as st
import sys, os, re, json
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="SerpAPI 测试", layout="wide")
st.title("SerpAPI Google Lens 测试")

uploaded = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "webp"])
if not uploaded:
    st.info("请上传图片")
    st.stop()

saved = ROOT / "data" / "uploads" / f"{uuid4().hex}_{re.sub(r'[^A-Za-z0-9_.-]', '_', uploaded.name)}"
saved.parent.mkdir(parents=True, exist_ok=True)
saved.write_bytes(uploaded.getbuffer())
st.image(str(saved), width=300)

# 加载 .env
from pathlib import Path as _P
_env_file = _P(__file__).resolve().parents[2] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
            if _k not in os.environ:
                os.environ[_k] = _v

if not st.button("测试", type="primary"):
    st.stop()

st.divider()

# ── 步骤1: 上传到 Imgur（TUN 模式直连）──
st.subheader("1. Imgur 上传")
import requests, base64

IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID", "")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", "")

st.write(f"Imgur Client ID: {IMGUR_CLIENT_ID[:8]}..." if IMGUR_CLIENT_ID else "Imgur Client ID: 未设置")
st.write(f"SerpAPI Key: {SERPAPI_KEY[:8]}..." if SERPAPI_KEY else "SerpAPI Key: 未设置")

img_url = None
try:
    with open(saved, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(
        "https://api.imgur.com/3/image",
        headers={"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"},
        data={"image": img_b64, "type": "base64"},
        timeout=30,
    )
    st.write(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        img_url = resp.json()["data"]["link"]
        st.success(f"上传成功: {img_url}")
    else:
        st.error(f"上传失败: {resp.text[:300]}")
except Exception as e:
    st.error(f"异常: {e}")

# ── 步骤2: 搜图 ──
if img_url:
    st.subheader("2. SerpAPI Google Lens")
    try:
        params = {
            "engine": "google_lens",
            "url": img_url,
            "api_key": SERPAPI_KEY,
        }
        resp = requests.get("https://serpapi.com/search", params=params, timeout=60)
        st.write(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            visual = data.get("visual_matches", [])
            st.success(f"搜索结果: {len(visual)} 个视觉匹配")
            if visual:
                st.json(visual[:5])
            else:
                st.json(data)
        else:
            st.error(resp.text[:500])
    except Exception as e:
        st.error(f"异常: {e}")
