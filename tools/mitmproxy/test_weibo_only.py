"""微博以图搜图自动化
用法: python test_weibo_only.py <图片路径>
示例: python test_weibo_only.py test.jpg
"""
import sys, time, json, subprocess, socket, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adb_automation import ADBClient, PACKAGE_WEIBO

if len(sys.argv) < 2:
    print("用法: python test_weibo_only.py <图片路径>")
    sys.exit(1)
IMAGE = Path(sys.argv[1]).resolve()
if not IMAGE.exists():
    print(f"图片不存在: {IMAGE}")
    sys.exit(1)

adb = ADBClient('127.0.0.1:5555')
adb.connect()

CAPTURE = Path('_mitmproxy_captured.json')

# 清理
subprocess.run(['taskkill', '/F', '/IM', 'mitmdump.exe'], capture_output=True)
CAPTURE.unlink(missing_ok=True)
time.sleep(1)

# 推图到模拟器
REMOTE = f'/sdcard/DCIM/_search_{IMAGE.name}'
print(f'[0] 推送图片: {IMAGE} -> {REMOTE}')
adb.push_file(str(IMAGE), REMOTE)
adb.shell(f'am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://{REMOTE}')
time.sleep(2)

# 启动 mitmdump — 跟小红书一模一样
print('[1] 启动 mitmdump...')
mitmdump = os.getenv('MITMDUMP_PATH', 'mitmdump')
addon = str(Path(__file__).resolve().with_name('capture_simple.py'))
proc = subprocess.Popen(
    [mitmdump, '-s', addon, '-p', '8888', '--ssl-insecure']
)
time.sleep(4)
if proc.poll() is not None:
    print('X mitmdump crashed!')
    sys.exit(1)
s = socket.socket(); s.settimeout(2)
try: s.connect(('127.0.0.1', 8888)); s.close(); print('OK mitmdump running')
except: print('X port 8888'); proc.terminate(); sys.exit(1)

# 微博流程 - 竖屏坐标
print('[2] 杀后台 + 启动微博...')
adb.force_stop(PACKAGE_WEIBO); time.sleep(1)
adb.force_stop('com.xingin.xhs'); time.sleep(1)
adb.launch_app(PACKAGE_WEIBO)
time.sleep(25)

print('[3] 点发现 (445,1549)...')
adb.tap(445, 1549); time.sleep(8)

print('[4] 点相机 (740,68)...')
adb.tap(740, 68); time.sleep(8)

print('[5] 点左下照片 (86,1499)...')
adb.tap(86, 1499); time.sleep(15)

# ── 下滑加载 ──────────────────────────────────────────
print('[6] 下滑加载更多...')
no_new = 0
for s in range(50):
    before = len(json.loads(CAPTURE.read_text(encoding='utf-8')).get('results', [])) if CAPTURE.exists() else 0
    for _ in range(3):
        adb.swipe(450, 1400, 450, 100, 200)
        time.sleep(0.3)
    time.sleep(1)
    after = len(json.loads(CAPTURE.read_text(encoding='utf-8')).get('results', [])) if CAPTURE.exists() else 0
    if after > before:
        print(f'  第{s+1}次: +{after-before} ({after}总)')
        no_new = 0
    else:
        no_new += 1
        if no_new >= 3:
            print(f'  连续3次无新结果, 停止')
            break

n = len(json.loads(CAPTURE.read_text(encoding='utf-8')).get('results', [])) if CAPTURE.exists() else 0
print(f'\n最终: {n} 条')

proc.terminate(); proc.wait()
print('done')
