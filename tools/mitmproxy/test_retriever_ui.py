"""Retriever 测试 UI — 上传图片 → 多引擎并发搜图（含 mitmproxy 自动化）"""
import streamlit as st
import os, sys, time, re, json
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 加载 .env
_env = ROOT / ".env"
if _env.exists():
    for _l in _env.read_text(encoding="utf-8").splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            _k, _v = _l.split("=", 1)
            os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

from core.state import build_initial_state
from agents.retriever import retrieve_node

st.set_page_config(page_title="Retriever 测试", layout="wide")
st.title("Retriever 搜图测试")
st.caption("上传图片 → 多引擎并发搜图（PicImageSearch + SerpAPI + mitmproxy 自动化）")

# ── 上传 ──
uploaded = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "webp"])
if not uploaded:
    st.info("请上传一张图片")
    st.stop()

saved = ROOT / "data" / "uploads" / f"{uuid4().hex}_{re.sub(r'[^A-Za-z0-9_.-]', '_', uploaded.name)}"
saved.parent.mkdir(parents=True, exist_ok=True)
saved.write_bytes(uploaded.getbuffer())

left, right = st.columns([1, 2])
with left:
    st.image(str(saved), caption="目标图片", use_container_width=True)
with right:
    st.json({"filename": uploaded.name, "type": uploaded.type, "size": uploaded.size})

# ── 引擎选择 ──
st.subheader("搜索引擎")
cols = st.columns(4)
engines_all = {
    "baidu": "百度", "yandex": "Yandex", "bing": "Bing",
    "google": "Google", "tineye": "TinEye", "saucenao": "SauceNAO",
    "ascii2d": "Ascii2D", "serpapi_lens": "Google Lens",
    "mitmproxy": "微博+小红书 (模拟器)",
}
selected = {}
for i, (key, label) in enumerate(engines_all.items()):
    selected[key] = cols[i % 4].checkbox(label, value=key in ("baidu", "mitmproxy"))

if not st.button("开始搜图", type="primary"):
    st.stop()

# ── 运行 ──
selected_engines = [k for k, v in selected.items() if v]
if not selected_engines:
    st.warning("至少选一个引擎")
    st.stop()

st.divider()
log_placeholder = st.empty()
progress_bar = st.progress(0)
status = st.empty()

def log(msg, level="info"):
    with log_placeholder.container():
        st.text(msg)

log("准备检索 ...")
progress_bar.progress(0.1)

# 构建 state
state = build_initial_state({
    "filename": uploaded.name,
    "content_type": uploaded.type,
    "size_bytes": uploaded.size,
    "local_path": str(saved),
})
state["search_engines"] = selected_engines
state["retriever_max_results"] = 999

# 进回调
last_msg = [""]
def on_progress(stage, current=0, total=0, sub_total=0, message=""):
    msg = str(message or "")
    if msg and msg != last_msg[0]:
        last_msg[0] = msg
        log(msg)

log("retrieve_node 启动 ...")
progress_bar.progress(0.2)

import asyncio
async def run():
    result = retrieve_node(state, progress_callback=on_progress)
    return result

# 用 asyncio 跑
loop = asyncio.new_event_loop()
result_state = loop.run_until_complete(run())
loop.close()

state.update(result_state)
progress_bar.progress(0.8)

# ── 结果 ──
st.divider()
st.subheader("检索结果")

summary = state.get("retrieval_summary", {})
cols = st.columns(4)
cols[0].metric("总候选", summary.get("result_count", 0))
cols[1].metric("引擎成功", len(summary.get("per_engine_counts", {})))
cols[2].metric("引擎异常", len(summary.get("errors", [])))
cols[3].metric("节点数", len(state.get("nodes_data", [])))

if summary.get("per_engine_counts"):
    st.subheader("引擎统计")
    for eng, cnt in sorted(summary["per_engine_counts"].items()):
        st.write(f"  {eng}: {cnt} 条")

if summary.get("errors"):
    with st.expander("引擎异常"):
        for e in summary["errors"]:
            st.error(e)

with st.expander("候选节点"):
    st.json(state.get("nodes_data", [])[:30])

with st.expander("retrieval_summary"):
    st.json(summary)

progress_bar.progress(1.0)
st.success("检索完成")
