"""自动化搜图测试 UI — 上传图片 → 推模拟器 → 抓包"""
import streamlit as st
import sys, time, json, subprocess, socket, os, re
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "mitmproxy"))

from tools.mitmproxy.adb_automation import ADBClient, PACKAGE_WEIBO, PACKAGE_XIAOHONGSHU

PROXY_PORT = 8888
SCROLL_NO_NEW_LIMIT = 3
SCROLL_MAX = 30
CAPTURE_FILE = ROOT / "_mitmproxy_captured.json"

st.set_page_config(page_title="自动化搜图测试", layout="wide")
st.title("自动化以图搜图测试")
st.caption("上传图片 → ADB 推送到模拟器 → 自动搜索 → mitmdump 抓包")

# ── 上传 ──
uploaded = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "webp"])
if not uploaded:
    st.info("请上传一张图片")
    st.stop()

# 保存到本地
saved = ROOT / "data" / "uploads" / f"{uuid4().hex}_{re.sub(r'[^A-Za-z0-9_.-]', '_', uploaded.name)}"
saved.parent.mkdir(parents=True, exist_ok=True)
saved.write_bytes(uploaded.getbuffer())

left, right = st.columns([1, 2])
with left:
    st.image(str(saved), caption="目标图片", use_container_width=True)
with right:
    st.subheader("图片信息")
    st.json({"filename": uploaded.name, "type": uploaded.type, "size": uploaded.size})

# ── 平台选择 ──
platform_choice = st.radio("搜索平台", ["仅微博", "仅小红书", "微博 + 小红书"], horizontal=True)
PLATFORMS = []
if platform_choice in ("仅微博", "微博 + 小红书"):
    PLATFORMS.append(("weibo", PACKAGE_WEIBO))
if platform_choice in ("仅小红书", "微博 + 小红书"):
    PLATFORMS.append(("xiaohongshu", PACKAGE_XIAOHONGSHU))

if not st.button("开始自动化搜图", type="primary"):
    st.stop()

# ── 执行 ──
log_placeholder = st.empty()
progress_bar = st.progress(0)
total_steps = 3 + len(PLATFORMS) * 3

def log(msg: str):
    with log_placeholder.container():
        st.text(msg)

step = 0
log("连接 ADB ...")

# 1. ADB
adb = ADBClient("127.0.0.1:5555")
if not adb.connect():
    st.error("ADB 连接失败，确认模拟器已启动")
    st.stop()
step += 1; progress_bar.progress(step / total_steps)
log(f"ADB 已连接, 分辨率: {adb.screen_size}")

# 2. 推图
REMOTE = f"/sdcard/DCIM/_search_{saved.name}"
log(f"推送图片: {saved.name} → {REMOTE}")
adb.push_file(str(saved), REMOTE)
adb.shell(f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://{REMOTE}")
time.sleep(3)
step += 1; progress_bar.progress(step / total_steps)

# 3. 启动 mitmdump
log("启动 mitmdump 抓包 ...")
subprocess.run(["taskkill", "/F", "/IM", "mitmdump.exe"], capture_output=True)
CAPTURE_FILE.unlink(missing_ok=True)
time.sleep(1)

addon = str(ROOT / "app_search" / "capture_simple.py")
mitm_proc = subprocess.Popen(
    [os.environ.get("MITMDUMP_PATH", "mitmdump"), "-s", addon, "-p", str(PROXY_PORT), "--ssl-insecure"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(3)
step += 1; progress_bar.progress(step / total_steps)
log("mitmdump 已启动")

# 4. 逐平台搜索
total_results = 0
for plat_name, package in PLATFORMS:
    log(f"===== {plat_name} 自动化搜图 =====")

    # 杀后台
    adb.force_stop(package); time.sleep(1)
    other_pkg = PACKAGE_XIAOHONGSHU if package == PACKAGE_WEIBO else PACKAGE_WEIBO
    adb.force_stop(other_pkg); time.sleep(1)
    adb.launch_app(package)
    log(f"  等待 {plat_name} 启动 (25s) ...")
    for i in range(25, 0, -5):
        log(f"  等待 {plat_name} 启动 ({i}s) ...")
        time.sleep(5)
    step += 1; progress_bar.progress(step / total_steps)

    # 自动搜图
    adb.trigger_reverse_image_search(plat_name, REMOTE, step_delay=3.0)

    time.sleep(10)
    step += 1; progress_bar.progress(step / total_steps)

    # 下滑坐标因平台而异
    if plat_name == "xiaohongshu":
        swipe_cfg = {"x": 800, "y1": 700, "y2": 100, "dur": 150, "times": 1, "interval": 0.5}
    else:
        swipe_cfg = {"x": 450, "y1": 1400, "y2": 100, "dur": 200, "times": 3, "interval": 0.3}
    log(f"  下滑加载更多 ...")
    no_new = 0
    for swipe_round in range(SCROLL_MAX):
        before = len(json.loads(CAPTURE_FILE.read_text(encoding="utf-8")).get("results", [])) if CAPTURE_FILE.exists() else 0
        for _ in range(swipe_cfg["times"]):
            adb.swipe(swipe_cfg["x"], swipe_cfg["y1"], swipe_cfg["x"], swipe_cfg["y2"], swipe_cfg["dur"])
            time.sleep(swipe_cfg["interval"])
        time.sleep(1)
        after = len(json.loads(CAPTURE_FILE.read_text(encoding="utf-8")).get("results", [])) if CAPTURE_FILE.exists() else 0
        if after > before:
            log(f"    第{swipe_round+1}次: +{after-before} (累计 {after})")
            no_new = 0
        else:
            no_new += 1
            if no_new >= SCROLL_NO_NEW_LIMIT:
                break

    gained = len(json.loads(CAPTURE_FILE.read_text(encoding="utf-8")).get("results", [])) if CAPTURE_FILE.exists() else 0
    log(f"  {plat_name} 完成: +{gained - total_results} 条")
    total_results = gained
    step += 1; progress_bar.progress(step / total_steps)

# 5. 收尾
log("清理模拟器图片 ...")
adb.shell(f"rm -f {REMOTE}")
mitm_proc.terminate(); mitm_proc.wait()
progress_bar.progress(1.0)

# ── 结果 ──
st.divider()
if CAPTURE_FILE.exists():
    data = json.loads(CAPTURE_FILE.read_text(encoding="utf-8"))
    results = data.get("results", [])
    by_platform = {}
    for r in results:
        p = r.get("platform", "unknown")
        by_platform[p] = by_platform.get(p, 0) + 1

    col1, col2 = st.columns(2)
    col1.metric("总结果", len(results))
    col2.metric("文件", CAPTURE_FILE.name)

    st.subheader("按平台统计")
    for p, c in sorted(by_platform.items()):
        st.write(f"  {p}: {c} 条")

    with st.expander("查看结果 JSON"):
        st.json(results[:20])
else:
    st.warning("未捕获到结果")
