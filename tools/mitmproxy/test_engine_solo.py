"""单独测试各引擎 — 排查 mitmproxy/serpapi 无结果问题"""
import streamlit as st
import os, sys, re, json
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

from core.state import build_initial_state, append_log

st.set_page_config(page_title="引擎单独测试", layout="wide")
st.title("引擎单独测试")
st.caption("上传图片，逐个测试 mitmproxy / serpapi_lens")

uploaded = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "webp"])
if not uploaded:
    st.info("请上传图片")
    st.stop()

saved = ROOT / "data" / "uploads" / f"{uuid4().hex}_{re.sub(r'[^A-Za-z0-9_.-]', '_', uploaded.name)}"
saved.parent.mkdir(parents=True, exist_ok=True)
saved.write_bytes(uploaded.getbuffer())
st.image(str(saved), width=300)

# ── 引擎选择 ──
eng = st.radio("引擎", ["mitmproxy", "serpapi_lens", "baidu"], horizontal=True)

if not st.button("测试", type="primary"):
    st.stop()

st.divider()

state = build_initial_state({"local_path": str(saved), "filename": uploaded.name,
                              "content_type": uploaded.type, "size_bytes": uploaded.size})
state["search_engines"] = [eng]
state["retriever_max_results"] = 10

import asyncio
from agents.retriever import retrieve_node

with st.spinner(f"正在测试 {eng} ..."):
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(retrieve_node(state))
    loop.close()

state.update(result)

summary = state.get("retrieval_summary", {})
st.subheader("结果")
nodes = state.get("nodes_data", [])
st.metric("候选数", len(nodes))

diag = summary.get("engine_diagnostics", {}).get(eng, {})
st.json(diag)

if nodes:
    st.subheader("前10个节点")
    st.json(nodes[:10])

# ── mitmproxy 额外诊断 ──
if eng == "mitmproxy":
    st.divider()
    st.subheader("mitmproxy 诊断")

    cap_file = ROOT / "output" / "_mitmproxy_captured.json"
    st.write(f"文件: {cap_file}")
    st.write(f"存在: {cap_file.exists()}")

    if cap_file.exists():
        data = json.loads(cap_file.read_text(encoding="utf-8"))
        results = data.get("results", [])
        st.write(f"结果数: {len(results)}")
        by_plat = {}
        for r in results:
            p = r.get("platform", "unknown")
            by_plat[p] = by_plat.get(p, 0) + 1
        st.json(by_plat)
        if results:
            st.json(results[:5])

    # 检查 ADB
    st.subheader("ADB 状态")
    import subprocess
    r = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=5)
    st.code(r.stdout + r.stderr)
