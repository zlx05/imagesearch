"""
ADB client and UI automation for LDPlayer 9 emulator.

Provides ``ADBClient`` to connect to an Android emulator/device, push files,
launch apps, and automate UI interactions — specifically reverse image search
on 小红书 and 微博.

Usage (standalone test)::

    python agents/adb_automation.py
"""

from __future__ import annotations

import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Platform package / activity info
# ---------------------------------------------------------------------------
PACKAGE_XIAOHONGSHU = "com.xingin.xhs"
PACKAGE_WEIBO = "com.sina.weibo"

PLATFORM_PACKAGES: Dict[str, str] = {
    "xiaohongshu": PACKAGE_XIAOHONGSHU,
    "weibo": PACKAGE_WEIBO,
}

# 正确的启动 Activity (从 pm resolve-activity 获取)
PLATFORM_ACTIVITIES: Dict[str, str] = {
    "xiaohongshu": "com.xingin.xhs/.index.v2.IndexActivityV2",
    "weibo": "com.sina.weibo/.SplashActivity",
}

# ---------------------------------------------------------------------------
# Fallback coordinate maps — all defined for base resolution 1280×720.
# They will be auto-scaled to the actual device resolution at runtime.
# ---------------------------------------------------------------------------
_BASE_W = 1280
_BASE_H = 720

COORD_XIAOHONGSHU = {
    "search_icon": (1046, 49),         # 实测 1600x900: (1308,61)
    "camera_in_search": (1170, 58),    # 实测 1600x900: (1463,72)
    "bottom_left_photo": (120, 600),   # 实测 1600x900: (150,750) 相机页左下最近照片
    "back": (80, 80),                  # 左上返回
}

COORD_WEIBO = {
    "discover_tab": (1150, 680),
    "search_bar": (640, 200),
    "camera_icon": (1200, 200),
    "from_album": (640, 500),
    "album_first_image": (180, 400),
    "confirm": (640, 80),
    "back": (80, 80),
}

GALLERY_COORDS = {
    "photos_folder": (200, 300),
    "images_folder": (600, 300),
    "first_image_4col": (160, 280),
}

# ---------------------------------------------------------------------------
# ADB executable paths to try (in order)
# ---------------------------------------------------------------------------
_ADB_CANDIDATES = [
    "adb.exe",
    "adb",
    r"D:\leidian\LDPlayer9\adb.exe",
    r"C:\LDPlayer\LDPlayer9\adb.exe",
    r"D:\LDPlayer\LDPlayer9\adb.exe",
    r"C:\Program Files\LDPlayer\LDPlayer9\adb.exe",
]


@dataclass
class ADBResult:
    """Returned by most ADB commands."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int


class ADBError(RuntimeError):
    """Raised when an ADB command fails in an unrecoverable way."""


class ADBClient:
    """Wraps ``adb`` CLI with convenience methods for Android UI automation.

    Can optionally use ``pure-python-adb`` (``ppadb``) if installed, but
    falls back to shelling out to the ``adb`` executable — which is more
    reliable on Windows.

    Parameters:
        serial: ADB device serial (e.g. ``"127.0.0.1:5555"``).
        adb_path: Path to the ``adb.exe`` binary. Auto-detected if omitted.
    """

    def __init__(
        self,
        serial: str = "127.0.0.1:5555",
        adb_path: Optional[str] = None,
    ):
        self.serial = serial
        self._adb = self._resolve_adb(adb_path)
        self._ppadb_device: Any = None
        self._screen_w: int = 1280
        self._screen_h: int = 720

    # -- public API ----------------------------------------------------------

    def connect(self) -> bool:
        """Ensure we can talk to the target device.  Returns True on success."""
        # Try native ppadb first
        try:
            from ppadb.client import Client as AdbPClient  # type: ignore
            host, port = self._split_serial()
            client = AdbPClient(host=host, port=5037)
            devices = client.devices()
            for d in devices:
                if d.serial == self.serial:
                    self._ppadb_device = d
                    self._detect_resolution()
                    return True
        except Exception:
            pass

        # Fallback: shell out
        result = self._run_adb(["connect", self.serial], timeout=8)
        ok = "connected" in result.stdout.lower() or "already" in result.stdout.lower() or self._check_devices()
        if ok:
            self._detect_resolution()
        return ok

    def is_connected(self) -> bool:
        """Quick connectivity check."""
        if self._ppadb_device is not None:
            try:
                self._ppadb_device.shell("echo ok")
                return True
            except Exception:
                self._ppadb_device = None
        return self._check_devices()

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Return current (width, height)."""
        return (self._screen_w, self._screen_h)

    def _detect_resolution(self):
        """Read actual screen resolution from the device and store it."""
        try:
            result = self._run_adb(["shell", "wm", "size"], timeout=5)
            m = re.search(r"(\d+)\s*[xX]\s*(\d+)", result.stdout or "")
            if m:
                self._screen_w = int(m[1])
                self._screen_h = int(m[2])
        except Exception:
            pass

    def _scale(self, x: int, y: int) -> Tuple[int, int]:
        """Scale base-1280×720 coordinates to the device's actual resolution."""
        return (
            int(x * self._screen_w / _BASE_W),
            int(y * self._screen_h / _BASE_H),
        )

    def push_file(self, local_path: str, remote_path: str) -> ADBResult:
        """Push a local file to the device.  Creates remote dirs as needed."""
        self._run_adb(["shell", "mkdir", "-p", str(Path(remote_path).parent)])
        return self._run_adb(["push", local_path, remote_path])

    def shell(self, command: str, timeout: float = 10) -> ADBResult:
        """Run an arbitrary shell command on the device."""
        return self._run_adb(["shell", command], timeout=timeout)

    def launch_app(self, package: str, activity: Optional[str] = None) -> ADBResult:
        """Start an app.  Uses known-good activity if available, else monkey."""
        # Look up the correct activity from our platform map
        for plat, pkg in PLATFORM_PACKAGES.items():
            if pkg == package and plat in PLATFORM_ACTIVITIES:
                return self._run_adb([
                    "shell", "am", "start", "-n", PLATFORM_ACTIVITIES[plat]
                ])
        if activity:
            return self._run_adb([
                "shell", "am", "start", "-n", f"{package}/{activity}"
            ])
        return self._run_adb([
            "shell", "monkey", "-p", package,
            "-c", "android.intent.category.LAUNCHER", "1",
        ])

    def force_stop(self, package: str) -> ADBResult:
        return self._run_adb(["shell", "am", "force-stop", package])

    def tap(self, x: int, y: int) -> ADBResult:
        return self._run_adb(["shell", "input", "tap", str(x), str(y)])

    def swipe(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: int = 300,
    ) -> ADBResult:
        return self._run_adb([
            "shell", "input", "swipe",
            str(x1), str(y1), str(x2), str(y2), str(duration_ms),
        ])

    def keyevent(self, keycode: int) -> ADBResult:
        return self._run_adb(["shell", "input", "keyevent", str(keycode)])

    def get_ui_hierarchy(self) -> Optional[str]:
        """Dump current UI hierarchy as XML string. Returns None on failure."""
        tmp = "/sdcard/window_dump.xml"
        result = self._run_adb(["shell", "uiautomator", "dump", tmp], timeout=12)
        if result.exit_code != 0 and "dumped" not in result.stdout:
            return None
        # Wait a tick for file to be fully written
        time.sleep(0.3)
        pull = self._run_adb(["exec-out", "cat", tmp], timeout=5)
        if pull.exit_code != 0 or not pull.stdout.strip():
            return None
        return pull.stdout

    def find_element_by_text(self, text: str) -> Optional[Tuple[int, int]]:
        """Return the center (x, y) of the first UI element whose ``text`` or
        ``content-desc`` attribute matches *text*.  Case-insensitive."""
        xml_str = self.get_ui_hierarchy()
        if not xml_str:
            return None
        try:
            # uiautomator dump sometimes produces non-XML preamble; skip it
            start = xml_str.find("<")
            if start > 0:
                xml_str = xml_str[start:]
            root = ET.fromstring(xml_str)
        except ET.ParseError:
            return None

        return self._search_element(root, text.lower())

    def wait_for_element(
        self, text: str, timeout: float = 10.0, interval: float = 1.0
    ) -> Optional[Tuple[int, int]]:
        """Poll for a UI element until it appears or *timeout* expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            coords = self.find_element_by_text(text)
            if coords is not None:
                return coords
            time.sleep(interval)
        return None

    def is_app_installed(self, package: str) -> bool:
        """Check if *package* exists on the device."""
        result = self._run_adb(["shell", "pm", "list", "packages", package])
        return package in result.stdout

    def screencap(self) -> Optional[bytes]:
        """Capture screenshot as PNG bytes. Returns None on failure."""
        cmd = [self._adb, "-s", self.serial, "exec-out", "screencap", "-p"]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
            )
            if proc.returncode != 0 and not proc.stdout:
                return None
            return proc.stdout
        except Exception:
            return None

    # -- UI automation -------------------------------------------------------

    def trigger_reverse_image_search(
        self,
        platform: str,
        image_remote_path: str,
        *,
        step_delay: float = 2.0,
    ) -> bool:
        """Run the UI sequence to perform a reverse-image search on *platform*.

        Parameters:
            platform: ``"xiaohongshu"`` or ``"weibo"``.
            image_remote_path: Device path of the image to search (e.g.
                               ``/sdcard/Pictures/search_img.jpg``).
            step_delay: Seconds to pause between each UI action.
        """
        platform_lower = platform.lower()
        if platform_lower == "xiaohongshu":
            return self._search_xiaohongshu(image_remote_path, step_delay)
        if platform_lower == "weibo":
            return self._search_weibo(image_remote_path, step_delay)
        return False

    # -- internal: xiaohongshu search flow -----------------------------------

    def _search_xiaohongshu(self, image_path: str, delay: float) -> bool:
        """小红书以图搜图 (横屏 1600x900): 首页(20s) -> 搜索(12s) -> 相机(8s) -> 选图(12s)."""
        steps: List[ADBResult] = []

        # 1. Launch app
        steps.append(self.launch_app(PACKAGE_XIAOHONGSHU))
        time.sleep(20)

        # 2. 右上角搜索 (1308,61)
        coord = self._find_or_fallback("搜索", (1308, 61))
        steps.append(self.tap(*coord))
        time.sleep(12)

        # 3. 相机图标 (1463,72)
        coord = self._find_or_fallback("扫一扫", (1463, 72))
        steps.append(self.tap(*coord))
        time.sleep(8)

        # 4. 左下角照片 -> 自动搜图 (150,750)
        self.tap(150, 750)
        time.sleep(12)

        return any(step.exit_code == 0 for step in steps)

    # -- internal: weibo search flow -----------------------------------------

    def _search_weibo(self, image_path: str, delay: float) -> bool:
        """微博以图搜图 (竖屏 900x1600): 启动(25s) -> 发现(8s) -> 相机(8s) -> 选图(15s)."""
        steps: List[ADBResult] = []

        # 1. Launch
        steps.append(self.launch_app(PACKAGE_WEIBO))
        time.sleep(25)

        # 2. 发现 (445,1549)
        coord = self._find_or_fallback("发现", (445, 1549))
        steps.append(self.tap(*coord))
        time.sleep(8)

        # 3. 相机 (740,68)
        steps.append(self.tap(740, 68))
        time.sleep(8)

        # 4. 左下照片 (86,1499)
        self.tap(86, 1499)
        time.sleep(15)

        return any(step.exit_code == 0 for step in steps)

    # -- internal helpers ----------------------------------------------------

    def _select_image_from_gallery(self, delay: float):
        """Attempt to select the first image in the system gallery."""
        self.tap(*self._scale(*GALLERY_COORDS["first_image_4col"]))
        time.sleep(delay * 0.6)

    def _find_or_fallback(self, text: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
        """Try to locate an element by text; fall back to coordinates."""
        coords = self.find_element_by_text(text)
        if coords is not None:
            return coords
        return fallback

    def _search_element(
        self, element: ET.Element, target: str
    ) -> Optional[Tuple[int, int]]:
        """DFS through the UI XML tree looking for an element whose ``text``
        or ``content-desc`` contains *target*."""
        for attr in ("text", "content-desc"):
            value = (element.get(attr) or "").lower()
            if target in value:
                bounds = element.get("bounds") or ""
                return _bounds_to_center(bounds)
        for child in element:
            result = self._search_element(child, target)
            if result is not None:
                return result
        return None

    def _run_adb(
        self, args: List[str], timeout: float = 10
    ) -> ADBResult:
        """Execute adb with -s <serial> prefix."""
        cmd = [self._adb, "-s", self.serial] + args
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            return ADBResult(
                success=proc.returncode == 0,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                exit_code=proc.returncode,
            )
        except FileNotFoundError:
            return ADBResult(
                success=False,
                stdout="",
                stderr=f"adb executable not found: {self._adb}",
                exit_code=-1,
            )
        except subprocess.TimeoutExpired:
            return ADBResult(
                success=False,
                stdout="",
                stderr=f"adb command timed out after {timeout}s",
                exit_code=-2,
            )

    def _check_devices(self) -> bool:
        result = self._run_adb(["devices"], timeout=5)
        return self.serial in result.stdout

    def _split_serial(self) -> Tuple[str, int]:
        parts = self.serial.split(":")
        host = parts[0] if len(parts) > 0 else "127.0.0.1"
        port = int(parts[1]) if len(parts) > 1 else 5555
        return host, port

    @staticmethod
    def _resolve_adb(path: Optional[str]) -> str:
        if path:
            return path
        for candidate in _ADB_CANDIDATES:
            try:
                subprocess.run(
                    [candidate, "version"],
                    capture_output=True,
                    timeout=3,
                )
                return candidate
            except Exception:
                continue
        return "adb.exe"  # fallback; will fail with clear error later


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


def _bounds_to_center(bounds: str) -> Optional[Tuple[int, int]]:
    """Convert a ``[x1,y1][x2,y2]`` bounds string to center coordinates."""
    m = _BOUNDS_RE.match(bounds)
    if not m:
        return None
    x1, y1, x2, y2 = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return ((x1 + x2) // 2, (y1 + y2) // 2)


# ---------------------------------------------------------------------------
# Standalone test (run with: python agents/adb_automation.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    host = os.getenv("ADB_HOST", "127.0.0.1")
    port = int(os.getenv("ADB_PORT", "5555"))
    serial = f"{host}:{port}"
    client = ADBClient(serial=serial)

    print(f"Connecting to {serial} ...")
    if not client.connect():
        print("ERROR: Failed to connect. Is the emulator running?")
        print("  Try: adb connect 127.0.0.1:5555")
        sys.exit(1)

    print("Connected OK")

    # Device info
    result = client.shell("getprop ro.product.model")
    print(f"Device model: {result.stdout.strip() or 'unknown'}")

    result = client.shell("getprop ro.build.version.release")
    print(f"Android version: {result.stdout.strip() or 'unknown'}")

    # Screen resolution
    w = client.shell("wm size")
    print(f"Screen: {w.stdout.strip()}")

    # App checks
    for platform, pkg in PLATFORM_PACKAGES.items():
        installed = client.is_app_installed(pkg)
        status = "OK" if installed else "MISSING"
        print(f"  {platform:15s} ({pkg:30s}) -> {status}")

    # Screenshot
    img_data = client.screencap()
    if img_data:
        out_path = Path(__file__).resolve().parents[1] / "data" / "adb_screenshot.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(img_data)
        print(f"\nScreenshot saved to: {out_path}")

    print("\nADBClient is ready.")
