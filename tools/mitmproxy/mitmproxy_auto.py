"""mitmproxy 自动化搜图模块 — 供 retriever 调用

被 retriever 的 mitmproxy 引擎调用时，自动完成：
1. ADB 连接模拟器
2. 推送目标图片
3. 启动 mitmdump 抓包
4. 自动化操控微博/小红书 App 搜图 + 下滑翻页
5. 返回抓取结果
"""
import json
import os
import subprocess
import sys
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "mitmproxy"))

from adb_automation import ADBClient, PACKAGE_WEIBO, PACKAGE_XIAOHONGSHU

PROXY_PORT = 8888
SCROLL_MAX = 30
SCROLL_NO_NEW_LIMIT = 3
CAPTURE_FILE = ROOT / "output" / "_mitmproxy_captured.json"
# 单平台结果上限，可通过环境变量 MITMPROXY_MAX_PER_PLATFORM 调整
_MAX_PER_PLATFORM = int(os.getenv("MITMPROXY_MAX_PER_PLATFORM", "150"))

# 各平台自动化参数
PLATFORM_CONFIG = {
    "weibo": {
        "package": PACKAGE_WEIBO,
        "launch_wait": 25,
        "swipe_x": 450, "swipe_y1": 1400, "swipe_y2": 100,
        "swipe_dur": 200, "swipe_times": 3, "swipe_interval": 0.3,
        "wait_after_tap": 15,
    },
    "xiaohongshu": {
        "package": PACKAGE_XIAOHONGSHU,
        "launch_wait": 20,
        "swipe_x": 800, "swipe_y1": 700, "swipe_y2": 100,
        "swipe_dur": 150, "swipe_times": 1, "swipe_interval": 0.5,
        "wait_after_tap": 12,
    },
}


def run_auto_search(
    image_path: str,
    platforms: Optional[List[str]] = None,
    adb_serial: str = "127.0.0.1:5555",
    adb_path: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """执行模拟器自动化以图搜图，返回捕获的节点列表。

    Parameters:
        image_path: 本地图片路径（将被推送到模拟器）
        platforms: 要搜索的平台列表，默认 ["weibo", "xiaohongshu"]
        adb_serial: ADB 设备序列号
        adb_path: adb.exe 路径，默认自动查找
        progress_callback: 进度回调 fn(stage: str, message: str)

    Returns:
        {"nodes": [...], "raw_count": int, "normalized_count": int, "errors": [...]}
    """
    if platforms is None:
        platforms = ["weibo", "xiaohongshu"]

    errors: List[str] = []
    nodes: List[Dict[str, Any]] = []
    total_results = 0

    def _progress(stage: str, msg: str = ""):
        if progress_callback:
            progress_callback(stage, msg)

    # ── 1. 检查图片 ──
    img = Path(image_path).resolve()
    if not img.exists():
        return {"nodes": [], "raw_count": 0, "normalized_count": 0, "errors": [f"图片不存在: {img}"]}

    # ── 2. 连接 ADB ──
    _progress("adb_connect", "连接模拟器 ...")
    adb = ADBClient(adb_serial, adb_path=adb_path)
    if not adb.connect():
        errors.append("ADB 连接失败，请确认模拟器已启动")
        return {"nodes": [], "raw_count": 0, "normalized_count": 0, "errors": errors}

    # ── 3. 推图 ──
    remote = f"/sdcard/DCIM/_search_{img.name}"
    _progress("push_image", f"推送图片: {img.name}")
    adb.push_file(str(img), remote)
    adb.shell(f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://{remote}")
    time.sleep(3)

    # ── 4. 启动 mitmdump ──
    _progress("start_mitmproxy", "启动 mitmdump 抓包 ...")
    subprocess.run(["taskkill", "/F", "/IM", "mitmdump.exe"], capture_output=True)
    time.sleep(1)
    time.sleep(1)

    # 使用 capture_simple.py 作为抓包插件
    addon = str(Path(__file__).resolve().parent / "capture_simple.py")
    mitm_cmd = os.environ.get("MITMDUMP_PATH", "mitmdump")
    try:
        mitm_proc = subprocess.Popen(
            [mitm_cmd, "-s", addon, "-p", str(PROXY_PORT), "--ssl-insecure"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        errors.append(f"mitmdump 未找到, 请安装 mitmproxy 或设置 MITMDUMP_PATH 环境变量")
        adb.shell(f"rm -f {remote}")
        return {"nodes": [], "raw_count": 0, "normalized_count": 0, "errors": errors}

    time.sleep(4)
    if mitm_proc.poll() is not None:
        errors.append("mitmdump 启动失败")
        adb.shell(f"rm -f {remote}")
        return {"nodes": [], "raw_count": 0, "normalized_count": 0, "errors": errors}

    # 验证端口
    try:
        s = socket.socket(); s.settimeout(3)
        s.connect(("127.0.0.1", PROXY_PORT)); s.close()
    except Exception:
        errors.append(f"mitmdump 端口 {PROXY_PORT} 不通")
        mitm_proc.terminate()
        adb.shell(f"rm -f {remote}")
        return {"nodes": [], "raw_count": 0, "normalized_count": 0, "errors": errors}

    # ── 5. 逐平台搜索 ──
    try:
        for plat_name in platforms:
            cfg = PLATFORM_CONFIG.get(plat_name)
            if not cfg:
                errors.append(f"未知平台: {plat_name}")
                continue

            _progress(f"search_{plat_name}", f"===== {plat_name} 自动化搜图 =====")

            # 杀后台
            adb.force_stop(cfg["package"])
            time.sleep(1)
            other_pkg = PACKAGE_XIAOHONGSHU if cfg["package"] == PACKAGE_WEIBO else PACKAGE_WEIBO
            adb.force_stop(other_pkg)
            time.sleep(1)

            # 启动 App
            adb.launch_app(cfg["package"])
            _progress(f"wait_{plat_name}", f"等待 {plat_name} 启动 ({cfg['launch_wait']}s)")
            time.sleep(cfg["launch_wait"])

            # 自动化搜图
            _progress(f"tap_{plat_name}", f"{plat_name} 自动化点击 ...")
            ok = adb.trigger_reverse_image_search(plat_name, remote, step_delay=3.0)
            if not ok:
                errors.append(f"{plat_name}: 自动化流程可能有步骤失败")

            time.sleep(cfg["wait_after_tap"])

            # 下滑翻页
            _progress(f"scroll_{plat_name}", f"{plat_name} 下滑加载更多 (上限 {_MAX_PER_PLATFORM}) ...")
            no_new = 0
            for swipe_round in range(SCROLL_MAX):
                before = _count_results()
                plat_count = before - total_results
                if plat_count >= _MAX_PER_PLATFORM:
                    _progress(f"scroll_{plat_name}", f"{plat_name} 已达上限 {_MAX_PER_PLATFORM}，停止下滑")
                    break
                for _ in range(cfg["swipe_times"]):
                    adb.swipe(cfg["swipe_x"], cfg["swipe_y1"],
                              cfg["swipe_x"], cfg["swipe_y2"],
                              cfg["swipe_dur"])
                    time.sleep(cfg["swipe_interval"])
                time.sleep(1)
                after = _count_results()
                if after > before:
                    _progress(f"scroll_{plat_name}", f"{plat_name} 第{swipe_round+1}次: +{after-before} (累计 {after})")
                    no_new = 0
                else:
                    no_new += 1
                    if no_new >= SCROLL_NO_NEW_LIMIT:
                        break

            gained = _count_results() - total_results
            total_results = _count_results()
            _progress(f"done_{plat_name}", f"{plat_name} 完成: +{gained} 条, 累计 {total_results}")

        # ── 6. 读取结果 ──
        results_data = _read_capture_file()
        nodes = results_data.get("results", [])

    finally:
        # ── 7. 清理 ──
        _progress("cleanup", "清理 ...")
        mitm_proc.terminate()
        mitm_proc.wait()
        adb.shell(f"rm -f {remote}")

    return {
        "nodes": nodes,
        "raw_count": total_results,
        "normalized_count": len(nodes),
        "errors": errors,
    }


def _count_results() -> int:
    """读取当前抓包文件的结果数."""
    data = _read_capture_file()
    return len(data.get("results", []))


def _read_capture_file() -> Dict[str, Any]:
    """安全读取抓包文件."""
    if not CAPTURE_FILE.exists():
        return {"results": [], "total": 0}
    try:
        return json.loads(CAPTURE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"results": [], "total": 0}
