"""自动化以图搜图测试脚本
用法: python tools/mitmproxy/test_auto_search.py <图片路径> [平台]
示例: python tools/mitmproxy/test_auto_search.py test.jpg           # 微博+小红书
      python tools/mitmproxy/test_auto_search.py test.jpg weibo     # 仅微博
      python tools/mitmproxy/test_auto_search.py test.jpg xhs       # 仅小红书
"""
import sys, time, json, subprocess, socket
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "mitmproxy"))

from tools.mitmproxy.adb_automation import ADBClient, PACKAGE_WEIBO, PACKAGE_XIAOHONGSHU

# ── 配置 ──
PROXY_PORT = 8888
ADB_SERIAL = "127.0.0.1:5555"
CAPTURE_FILE = ROOT / "output" / "_mitmproxy_captured.json"
SCROLL_MAX = 30          # 最多下滑次数
SCROLL_NO_NEW_LIMIT = 3  # 连续无新结果则停止

# ── 参数 ──
if len(sys.argv) < 2:
    print("用法: python app_search/test_auto_search.py <图片路径> [weibo|xhs|all]")
    sys.exit(1)

IMAGE = Path(sys.argv[1]).resolve()
if not IMAGE.exists():
    print(f"图片不存在: {IMAGE}")
    sys.exit(1)

PLATFORM = sys.argv[2].lower() if len(sys.argv) > 2 else "all"
PLATFORMS = []
if PLATFORM in ("all", "weibo"):
    PLATFORMS.append(("weibo", PACKAGE_WEIBO))
if PLATFORM in ("all", "xhs", "xiaohongshu"):
    PLATFORMS.append(("xiaohongshu", PACKAGE_XIAOHONGSHU))

print(f"=" * 50)
print(f"图片: {IMAGE}")
print(f"平台: {[p[0] for p in PLATFORMS]}")
print(f"=" * 50)

# ── 1. 连接 ADB ──
print("\n[1] 连接模拟器...")
adb = ADBClient(ADB_SERIAL)
if not adb.connect():
    print("ERROR: ADB 连接失败，请确认模拟器已启动")
    print("  尝试: adb connect 127.0.0.1:5555")
    sys.exit(1)
print(f"  连接成功, 分辨率: {adb.screen_size}")

# ── 2. 推图 ──
REMOTE = f"/sdcard/DCIM/_search_{IMAGE.name}"
print(f"\n[2] 推送图片到模拟器: {REMOTE}")
adb.push_file(str(IMAGE), REMOTE)
# 触发媒体扫描
adb.shell(f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://{REMOTE}")
time.sleep(2)
print("  推送完成")

# ── 3. 清理旧数据 + 启动 mitmdump ──
print("\n[3] 清理旧数据 + 启动抓包...")
subprocess.run(["taskkill", "/F", "/IM", "mitmdump.exe"], capture_output=True)
CAPTURE_FILE.unlink(missing_ok=True)
time.sleep(1)

addon = str(Path(__file__).resolve().parent / "capture_simple.py")
mitmdump_cmd = os.environ.get("MITMDUMP_PATH", "mitmdump")
mitm_proc = subprocess.Popen(
    [mitmdump_cmd, "-s", addon, "-p", str(PROXY_PORT), "--ssl-insecure"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
time.sleep(4)

if mitm_proc.poll() is not None:
    print("ERROR: mitmdump 启动失败!")
    sys.exit(1)

# 验证端口
try:
    s = socket.socket(); s.settimeout(3)
    s.connect(("127.0.0.1", PROXY_PORT)); s.close()
    print(f"  mitmdump 运行中 (端口 {PROXY_PORT})")
except Exception:
    print(f"ERROR: 端口 {PROXY_PORT} 无法连接!")
    mitm_proc.terminate()
    sys.exit(1)

# ── 4. 逐平台搜图 ──
results_before = 0
total_results = 0

for plat_name, package in PLATFORMS:
    print(f"\n[4] ===== {plat_name} 自动化搜图 =====")

    # 杀后台
    print(f"  杀后台 + 启动 {plat_name}...")
    adb.force_stop(package)
    time.sleep(1)

    # 清理另一个 App 避免干扰
    other_pkg = PACKAGE_XIAOHONGSHU if package == PACKAGE_WEIBO else PACKAGE_WEIBO
    adb.force_stop(other_pkg)
    time.sleep(1)

    adb.launch_app(package)
    print("  等待 App 启动 (25s)...")
    time.sleep(25)

    # 自动化搜图
    ok = adb.trigger_reverse_image_search(plat_name, REMOTE, step_delay=3.0)
    if not ok:
        print(f"  WARNING: {plat_name} 自动化流程可能有步骤失败")

    print("  等待搜索结果...")
    time.sleep(10)

    # 下滑坐标因平台而异
    if plat_name == "xiaohongshu":
        # 横屏 1600x900
        swipe_cfg = {"x": 800, "y1": 700, "y2": 100, "dur": 150, "times": 1, "interval": 0.5}
    else:
        # 竖屏 900x1600
        swipe_cfg = {"x": 450, "y1": 1400, "y2": 100, "dur": 200, "times": 3, "interval": 0.3}
    print(f"  下滑: ({swipe_cfg['x']},{swipe_cfg['y1']})→({swipe_cfg['x']},{swipe_cfg['y2']}) {swipe_cfg['dur']}ms ×{swipe_cfg['times']}")
    no_new = 0
    for swipe_round in range(SCROLL_MAX):
        before = len(json.loads(CAPTURE_FILE.read_text(encoding="utf-8")).get("results", [])) if CAPTURE_FILE.exists() else 0
        for _ in range(swipe_cfg["times"]):
            adb.swipe(swipe_cfg["x"], swipe_cfg["y1"], swipe_cfg["x"], swipe_cfg["y2"], swipe_cfg["dur"])
            time.sleep(swipe_cfg["interval"])
        time.sleep(1)
        after = len(json.loads(CAPTURE_FILE.read_text(encoding="utf-8")).get("results", [])) if CAPTURE_FILE.exists() else 0

        if after > before:
            print(f"    第{swipe_round+1}次: +{after-before} (累计 {after})")
            no_new = 0
        else:
            no_new += 1
            if no_new >= SCROLL_NO_NEW_LIMIT:
                print(f"    连续{SCROLL_NO_NEW_LIMIT}次无新结果, 停止下滑")
                break

    platform_results = len(json.loads(CAPTURE_FILE.read_text(encoding="utf-8")).get("results", [])) if CAPTURE_FILE.exists() else 0
    gained = platform_results - total_results
    total_results = platform_results
    print(f"  {plat_name} 完成: +{gained} 条, 累计 {total_results} 条")

# ── 5. 收尾 ──
print(f"\n[5] 抓包完成")

# 按平台统计
if CAPTURE_FILE.exists():
    data = json.loads(CAPTURE_FILE.read_text(encoding="utf-8"))
    results = data.get("results", [])
    by_platform = {}
    for r in results:
        p = r.get("platform", "unknown")
        by_platform[p] = by_platform.get(p, 0) + 1
    print(f"  总计: {len(results)} 条")
    for p, c in sorted(by_platform.items()):
        print(f"    {p}: {c} 条")
    print(f"  结果文件: {CAPTURE_FILE}")

# 清理模拟器上的图片
print("\n[6] 清理模拟器图片...")
adb.shell(f"rm -f {REMOTE}")
adb.shell("am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file:///sdcard/DCIM/")

mitm_proc.terminate()
mitm_proc.wait()
print("\ndone")
