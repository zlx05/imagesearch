"""小红书以图搜图自动化
用法: python test_xhs_only.py <图片路径>
示例: python test_xhs_only.py test.jpg
"""
import sys, time, json, subprocess, socket, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adb_automation import ADBClient, PACKAGE_XIAOHONGSHU

# 图片参数
if len(sys.argv) < 2:
    print("用法: python test_xhs_only.py <图片路径>")
    sys.exit(1)
IMAGE = Path(sys.argv[1]).resolve()
if not IMAGE.exists():
    print(f"图片不存在: {IMAGE}")
    sys.exit(1)

adb = ADBClient('127.0.0.1:5555')
adb.connect()

CAPTURE = Path('_mitmproxy_captured.json')

# 1. 清理
subprocess.run(['taskkill', '/F', '/IM', 'mitmdump.exe'], capture_output=True)
CAPTURE.unlink(missing_ok=True)
time.sleep(1)

# 2. 推图到模拟器 (DCIM 目录会被相册扫描到)
REMOTE = f'/sdcard/DCIM/_search_{IMAGE.name}'
print(f'[0] 推送图片: {IMAGE} -> {REMOTE}')
adb.push_file(str(IMAGE), REMOTE)
# 触发媒体扫描
adb.shell(f'am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://{REMOTE}')
time.sleep(2)

# 3. 启动 mitmdump
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
# double check port is listening
import socket
s = socket.socket(); s.settimeout(2)
try:
    s.connect(('127.0.0.1', 8888)); s.close()
    print('OK mitmdump running (port 8888 confirmed)')
except:
    print('X port 8888 not listening!')
    proc.terminate()
    sys.exit(1)

# 4. 小红书流程
print('[3] 杀后台 + 启动...')
adb.force_stop(PACKAGE_XIAOHONGSHU); time.sleep(1)
adb.force_stop('com.sina.weibo'); time.sleep(1)
adb.launch_app(PACKAGE_XIAOHONGSHU)
print('  等待20秒加载首页...')
time.sleep(20)

print('[4] 点搜索 (1308,61)...')
adb.tap(1308, 61)
time.sleep(12)

print('[5] 点相机 (1463,72)...')
adb.tap(1463, 72)
time.sleep(8)

print('[6] 点左下照片 (150,750)...')
adb.tap(150, 750)
time.sleep(12)

# ── 下滑加载 ──────────────────────────────────────────
print('[7] 下滑加载更多...')
no_new = 0
for s in range(50):
    before = len(json.loads(CAPTURE.read_text(encoding='utf-8')).get('results', [])) if CAPTURE.exists() else 0
    adb.swipe(800, 700, 800, 100, 150)
    time.sleep(0.5)
    after = len(json.loads(CAPTURE.read_text(encoding='utf-8')).get('results', [])) if CAPTURE.exists() else 0
    if after > before:
        print(f'  第{s+1}次: +{after-before} ({after}总)')
        no_new = 0
    else:
        no_new += 1
        if no_new >= 3:
            print(f'  连续3次无新结果, 停止 (共{after}条)')
            break

# ── 输出 ──────────────────────────────────────────────
if CAPTURE.exists():
    d = json.loads(CAPTURE.read_text(encoding='utf-8'))
    xhs = [r for r in d.get('results', []) if r.get('engine') == 'xiaohongshu']
    print(f'\n最终: {len(xhs)} 条')
    for r in xhs[:10]:
        print(f"  {r.get('title','?')[:60]} | {r.get('author','?')}")
else:
    print('\n无结果')

proc.terminate(); proc.wait()
print('done')
