# -*- coding: utf-8 -*-
"""Round 6 -- try to actually parse Chinese results out of CNKI-search and Wanfang."""
import io
import json
import os
import re
import ssl
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch(url, timeout=30):
    hdr = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"}
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers=hdr)
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read().decode("utf-8", "replace")


QUERIES = ["转子 临界转速 神经网络", "转子 生成式设计", "临界转速 代理模型",
           "转子动力学 机器学习 优化设计"]

out = {}
for q in QUERIES:
    print("\n" + "=" * 72)
    print("QUERY:", q)
    # ---- CNKI old search
    try:
        html = fetch("https://search.cnki.com.cn/Search/Result?content=" + urllib.parse.quote(q))
        titles = re.findall(r'<a[^>]*class="[^"]*fz14[^"]*"[^>]*>(.*?)</a>', html, re.S)
        if not titles:
            titles = re.findall(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>', html, re.S)
        clean = [re.sub(r"<[^>]+>", "", t).strip() for t in titles]
        clean = [t for t in clean if len(t) > 6]
        print("  CNKI-search: %d titles" % len(clean))
        for t in clean[:10]:
            print("     -", t[:95])
        out.setdefault("cnki", {})[q] = clean[:25]
    except Exception as exc:                          # noqa: BLE001
        print("  CNKI-search FAILED", type(exc).__name__, str(exc)[:80])
    # ---- Wanfang
    try:
        html = fetch("https://s.wanfangdata.com.cn/paper?q=" + urllib.parse.quote(q))
        titles = re.findall(r'class="title"[^>]*>\s*<a[^>]*>(.*?)</a>', html, re.S)
        if not titles:
            titles = re.findall(r'<a[^>]*href="/paper/periodical[^"]*"[^>]*>(.*?)</a>', html, re.S)
        clean = [re.sub(r"<[^>]+>", "", t).strip() for t in titles]
        clean = [t for t in clean if len(t) > 6]
        print("  Wanfang: %d titles" % len(clean))
        for t in clean[:10]:
            print("     -", t[:95])
        out.setdefault("wanfang", {})[q] = clean[:25]
    except Exception as exc:                          # noqa: BLE001
        print("  Wanfang FAILED", type(exc).__name__, str(exc)[:80])

with io.open(os.path.join(HERE, "chinese_search_20261002.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print("\n[wrote] chinese_search_20261002.json")