"""Fix Xiaohongshu data in _mitmproxy_captured.json

Workflow:
  1. Backup current file + read weibo entries into memory
  2. Start a FRESH mitmdump capture
  3. You manually do Xiaohongshu image search
  4. Stop capture, extract XHS-only results
  5. Merge: old weibo + new xiaohongshu

Usage:
  python tools/mitmproxy/fix_xhs_data.py
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "output"
CAPTURED_FILE = OUTPUT / "_mitmproxy_captured.json"
BACKUP_FILE = OUTPUT / "_mitmproxy_captured.json.bak"
CAPTURE_SCRIPT = Path(__file__).resolve().parent / "capture_simple.py"
MITMDUMP = os.getenv("MITMDUMP_PATH", "mitmdump")
PROXY_PORT = int(os.getenv("MITMPROXY_PORT", "8888"))


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def step(msg: str) -> None:
    print(f"  - {msg}")


# ---- step 1: backup + read weibo into memory ----

def backup_and_read_weibo():
    """Backup the existing file, extract weibo entries into memory."""
    if not CAPTURED_FILE.exists():
        step("no existing capture file")
        return []

    # Save backup (so we have a recovery point)
    shutil.copy2(CAPTURED_FILE, BACKUP_FILE)
    step(f"backup saved: {BACKUP_FILE.name}")

    data = json.loads(CAPTURED_FILE.read_text(encoding="utf-8"))
    results = data.get("results", [])

    weibo = [r for r in results if r.get("platform") == "weibo"]
    xhs = [r for r in results if r.get("platform") == "xiaohongshu"]
    other = [r for r in results if r.get("platform") not in ("weibo", "xiaohongshu")]

    step(f"backup contains: weibo={len(weibo)}  xhs={len(xhs)}  other={len(other)}")

    # Remove captured file so capture_simple starts clean
    CAPTURED_FILE.unlink()

    return weibo


# ---- step 2: launch capture ----

def launch_capture():
    cmd = [
        MITMDUMP,
        "-s", str(CAPTURE_SCRIPT),
        "--set", "block_global=false",
        "-p", str(PROXY_PORT),
        "--ssl-insecure",
    ]
    step(f"launch: mitmdump -s capture_simple.py -p {PROXY_PORT}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(PROJECT),
    )
    time.sleep(2.5)
    if proc.poll() is not None:
        out = ""
        try:
            out = proc.stdout.read()[:500] if proc.stdout else ""
        except Exception:
            pass
        step(f"FAILED to start mitmdump (exit={proc.returncode})")
        if out:
            step(f"output: {out}")
        return None
    step(f"mitmdump running (pid={proc.pid})")
    return proc


def stop_capture(proc):
    if proc is None:
        return ""
    step("stopping mitmdump ...")
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate(timeout=3)
    except Exception:
        out = ""
    step("stopped")
    return out or ""


# ---- step 3: read new XHS results ----

def read_new_xhs():
    if not CAPTURED_FILE.exists():
        step("no capture file produced")
        return []

    try:
        data = json.loads(CAPTURED_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        step(f"capture file corrupt: {e}")
        return []

    results = data.get("results", [])
    new_xhs = [r for r in results if r.get("platform") == "xiaohongshu"]
    new_weibo = [r for r in results if r.get("platform") == "weibo"]
    step(f"new capture: {len(results)} total ({len(new_xhs)} xhs, {len(new_weibo)} weibo)")
    return new_xhs


# ---- step 4: merge ----

def merge_and_write(weibo: list, new_xhs: list):
    all_results = []
    seen_urls = set()
    rank = 0

    def add(item):
        nonlocal rank
        url = item.get("url", "") or item.get("image_url", "")
        if url and url in seen_urls:
            return
        if url:
            seen_urls.add(url)
        rank += 1
        item["rank"] = rank
        all_results.append(item)

    # Weibo first (original order)
    for item in weibo:
        add(item)
    weibo_end = rank

    # Xiaohongshu second
    for item in new_xhs:
        add(item)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    CAPTURED_FILE.write_text(
        json.dumps({
            "results": all_results,
            "total": len(all_results),
            "weibo_count": weibo_end,
            "xiaohongshu_count": len(all_results) - weibo_end,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    step(f"written: {len(all_results)} total (weibo={weibo_end} + xhs={len(all_results)-weibo_end})")


# ---- main ----

def main():
    section("Fix Xiaohongshu Capture Data")

    # 1. Backup + extract weibo
    weibo = backup_and_read_weibo()
    if not weibo:
        print("\n  WARNING: no weibo entries found in backup!")
        print("  If this is unexpected, restore from git:")
        print("    git checkout -- output/_mitmproxy_captured.json")
        yn = input("\n  Continue anyway? (y/n): ").strip().lower()
        if yn != "y":
            # Restore backup
            if BACKUP_FILE.exists():
                shutil.copy2(BACKUP_FILE, CAPTURED_FILE)
            return 1

    # 2. Launch capture
    section("Start Xiaohongshu Capture")
    proc = launch_capture()
    if proc is None:
        # Restore backup
        if BACKUP_FILE.exists():
            shutil.copy2(BACKUP_FILE, CAPTURED_FILE)
            step("restored original file from backup")
        return 1

    # 3. User instructions
    print(f"""
  +-------------------------------------------------------+
  |  Proxy running on port {PROXY_PORT}. Now:                  |
  |                                                        |
  |  1. Open Xiaohongshu (小红书) on emulator               |
  |  2. Tap search bar -> camera icon -> select photo       |
  |  3. Wait for image search results to load               |
  |  4. Scroll down several times to load more              |
  |  5. Press Enter here when done                         |
  +-------------------------------------------------------+
""")

    try:
        input("  Press Enter to stop capture ...")
    except (KeyboardInterrupt, EOFError):
        print()

    # 4. Stop + read
    mitm_out = stop_capture(proc)

    for line in mitm_out.splitlines():
        if any(k in line.lower() for k in ("[xhs]", "total=", "+", "error", "warn")):
            print(f"  mitm: {line[:160]}")

    # 5. Extract XHS
    section("New Results")
    new_xhs = read_new_xhs()

    if not new_xhs:
        print("\n  No Xiaohongshu results captured!")
        shutil.copy2(BACKUP_FILE, CAPTURED_FILE)
        step("restored original file (unchanged)")
        return 1

    print(f"\n  Captured {len(new_xhs)} Xiaohongshu entries:")
    for item in new_xhs[:5]:
        print(f"    - {(item.get('title') or 'no title')[:70]}")
        print(f"      author={item.get('author','?')}  "
              f"img={'OK' if item.get('image_url','').startswith('http') else 'MISSING'}")

    # 6. Merge
    section("Merge")
    merge_and_write(weibo, new_xhs)
    print(f"\n  Done: {CAPTURED_FILE}")
    print(f"  Backup: {BACKUP_FILE} (delete when confirmed)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
