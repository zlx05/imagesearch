"""小红书/微博抓包插件"""
import json, time, threading, re
from pathlib import Path
from mitmproxy import http, ctx

OUT = Path(__file__).resolve().parents[2] / "output" / "_mitmproxy_captured.json"

class CaptureAddon:
    def __init__(self):
        self._lock = threading.Lock()
        self._results = []
        self._rank = 0
        self._seen = set()
        self._fallback = None  # old cached data, discarded on first new capture

    def load(self, loader):
        loader.add_option("mitmproxy_targets", str, "", "")
        loader.add_option("mitmproxy_output_file", str, "", "")
        loader.add_option("mitmproxy_max_results_per_platform", int, 0, "")
        loader.add_option("block_global", bool, False, "")
        self._load_existing()  # keep old data as fallback

    def _load_existing(self):
        """Load previous cached data as fallback.
        Once a new capture produces results, we replace (not append) old data.
        """
        try:
            if OUT.exists():
                data = json.loads(OUT.read_text(encoding="utf-8"))
                self._fallback = data.get("results", []) if isinstance(data, dict) else []
        except Exception:
            self._fallback = []

    def _add(self, node):
        # First new result: discard fallback, start fresh
        if self._fallback is not None:
            self._results = []
            self._seen = set()
            self._rank = 0
            self._fallback = None

        key = node.get("url") or node.get("image_url") or ""
        if key and key in self._seen:
            return
        if key:
            self._seen.add(key)
        self._rank += 1
        node["rank"] = self._rank
        self._results.append(node)

    def response(self, flow: http.HTTPFlow):
        url = flow.request.pretty_url
        host = flow.request.pretty_host or ""
        try:
            # ── 小红书 ──
            if "edith.xiaohongshu.com" in host and "/search/images" in url:
                data = json.loads(flow.response.text)
                items = data.get("data", {}).get("items", [])
                if not items:
                    return
                for item in items:
                    user = item.get("user") or {}
                    # 图片: image_info.url_size_large > images_list[0].original > image_info.url
                    image_info = item.get("image_info", {}) or {}
                    img = image_info.get("url_size_large") or image_info.get("url") or ""
                    if not img:
                        imgs = item.get("images_list", [])
                        if imgs and isinstance(imgs[0], dict):
                            img = imgs[0].get("original") or imgs[0].get("url") or ""
                    title = item.get("display_title", "")
                    note_id = item.get("id", "")
                    with self._lock:
                        self._add({
                            "title": title or item.get("desc", ""),
                            "url": f"https://www.xiaohongshu.com/explore/{note_id}" if note_id else "",
                            "image_url": img,
                            "author": user.get("nickname", ""),
                            "likes": item.get("likes", 0),
                            "note_id": note_id,
                            "platform": "xiaohongshu",
                            "engine": "mitmproxy",
                            "source": "xiaohongshu.com",
                        })
                        self._write()
                ctx.log.info(f"[xhs] +{len(items)} total={len(self._results)}")

            # ── 微博 (从 mitmproxy_addon.py 移植) ──
            if "weibo" in host and ("/search" in url or "/wis/" in url):
                data = json.loads(flow.response.text)
                items = data.get("cards") or data.get("data", {}).get("cards") or []
                if not items:
                    items = data.get("items") or data.get("data", {}).get("items") or []

                for card in items:
                    cat = card.get("category", "")
                    if cat == "card":
                        d = card.get("data", {})
                        if not isinstance(d, dict): continue
                        ct = str(d.get("card_type", ""))
                        if ct not in ("127", "11"): continue
                        title = d.get("title") or d.get("text") or ""
                        scheme = d.get("scheme", "")
                        author = ""
                        created = d.get("created_at", "")
                        # 图片: mblog.bmiddle_pic > page_pic > pic_infos
                        img = d.get("page_pic") or d.get("pic") or ""
                        for nk in ("user", "weibo", "mblog"):
                            nd = d.get(nk)
                            if isinstance(nd, dict):
                                title = title or (nd.get("text") or "")
                                author = author or str(nd.get("screen_name") or nd.get("nickname") or "")
                                if not img:
                                    img = nd.get("bmiddle_pic") or nd.get("original_pic") or ""
                                    if not img:
                                        pics = nd.get("pics") or []
                                        if pics:
                                            p0 = pics[0]
                                            img = p0.get("url","") if isinstance(p0,dict) else str(p0)
                                created = created or str(nd.get("created_at") or "")
                        # pic_infos fallback
                        if not img:
                            pinfos = d.get("pic_infos") or {}
                            if pinfos:
                                first_key = next(iter(pinfos), None)
                                if first_key:
                                    pi = pinfos[first_key]
                                    img = pi.get("large",{}).get("url") or pi.get("bmiddle",{}).get("url") or "" if isinstance(pi, dict) else ""
                        m = re.search(r'mix_mid=(\d+)', scheme)
                        page_url = f"https://m.weibo.cn/status/{m.group(1)}" if m else (d.get("url") or "")
                        if img and not img.startswith("http"):
                            img = f"https://wx1.sinaimg.cn/large/{img}"
                    else:
                        mblog = card.get("mblog") or card
                        user = mblog.get("user") or {}
                        img = mblog.get("bmiddle_pic") or mblog.get("original_pic") or ""
                        if not img:
                            pics = mblog.get("pics") or []
                            img = pics[0].get("url","") if pics else ""
                        title = re.sub(r"<[^>]*>","",mblog.get("text",""))[:80]
                        page_url = f"https://weibo.com/{user.get('id','')}/{mblog.get('mid','')}"
                        author = user.get("screen_name","")
                        created = mblog.get("created_at","")
                        scheme = ""

                    if not page_url and not img: continue
                    with self._lock:
                        self._add({
                            "title": title or page_url, "url": page_url or img,
                            "image_url": img, "author": author,
                            "published_at": created, "platform": "weibo",
                            "engine": "mitmproxy", "source": "weibo.com",
                        })
                        self._write()
                if items:
                    ctx.log.info(f"[wb] +{len(items)} cards total={len(self._results)}")
        except Exception:
            pass

    def _write(self, clear=False):
        if clear:
            self._results = []
        # If no new data captured, keep old fallback data
        results = self._results if self._results else (self._fallback or [])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "results": results, "total": len(results),
            "updated_at": time.time()
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def done(self):
        self._write()

addons = [CaptureAddon()]
