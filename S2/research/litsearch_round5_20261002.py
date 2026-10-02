# -*- coding: utf-8 -*-
"""Round 5 -- Chinese databases, arXiv lookups, and OA PDF download."""
import io
import json
import os
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(HERE, "pdf_new")
os.makedirs(DL, exist_ok=True)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


def fetch(url, data=None, timeout=25, headers=None):
    hdr = {"User-Agent": UA, "Accept": "*/*", "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
    hdr.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdr)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.status, r.read()


print("=" * 70)
print("A. Chinese databases -- can they be reached from a script at all?")
CH = [
    ("CNKI search page", "https://kns.cnki.net/kns8s/defaultresult/index?kw=" +
     urllib.parse.quote("转子 临界转速 代理模型")),
    ("CNKI old search", "https://search.cnki.com.cn/Search/Result?content=" +
     urllib.parse.quote("转子 生成式设计")),
    ("Baidu Xueshu", "https://xueshu.baidu.com/s?wd=" +
     urllib.parse.quote("转子 临界转速 神经网络")),
    ("Wanfang", "https://s.wanfangdata.com.cn/paper?q=" +
     urllib.parse.quote("转子 临界转速 代理模型")),
    ("CQVIP", "http://qikan.cqvip.com/Qikan/Search/Index?key=" +
     urllib.parse.quote("转子 临界转速")),
    ("Chinese J Mech Eng (学报)", "https://www.cjmenet.com.cn/CN/volumn/current.shtml"),
    ("Semantic Scholar", "https://api.semanticscholar.org/graph/v1/paper/search?query=" +
     urllib.parse.quote("rotor critical speed generative design") + "&limit=5&fields=title,year"),
]
for name, url in CH:
    try:
        status, body = fetch(url, timeout=20)
        text = body.decode("utf-8", "replace")
        print("  %-26s HTTP %s  %6d bytes  hits=%s" % (
            name, status, len(body),
            len([1 for kw in ("转子", "临界转速", "rotor", "title") if kw in text])))
    except Exception as exc:                          # noqa: BLE001
        print("  %-26s FAILED %s: %s" % (name, type(exc).__name__, str(exc)[:90]))

print("=" * 70)
print("B. arXiv lookups for specific known-good items")
AX = [
    "ti:\"ARRID\"",
    "all:\"performance-to-design\" AND cat:eess.SY",
    "abs:\"rotor\" AND abs:\"surrogate model\" AND abs:\"design optimization\"",
    "abs:\"parametric design\" AND abs:\"diffusion\" AND abs:\"constraint\"",
]
for q in AX:
    url = ("http://export.arxiv.org/api/query?search_query=%s&start=0&max_results=6"
           % urllib.parse.quote(q))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=40) as r:
            raw = r.read()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(raw)
        entries = root.findall("a:entry", ns)
        print("\n  query %s -> %d" % (q, len(entries)))
        for e in entries:
            print("    %s | %s | %s" % (e.find("a:published", ns).text[:10],
                                        e.find("a:id", ns).text.split("/abs/")[-1],
                                        " ".join(e.find("a:title", ns).text.split())[:80]))
    except Exception as exc:                          # noqa: BLE001
        print("  query FAILED %s: %s" % (q, str(exc)[:90]))

print("=" * 70)
print("C. download the open-access PDFs that matter for the review")
TARGETS = [
    ("2026_RePaint_Conditional_Diffusion_Performance_Constraints_arXiv.pdf",
     "https://arxiv.org/pdf/2602.00384"),
    ("2024_CcDPM_Continuous_Conditional_Diffusion_Inverse_Design_AAAI.pdf",
     "https://ojs.aaai.org/index.php/AAAI/article/download/29647/31235"),
    ("2024_Bagazinski_CShipGen_arXiv.pdf",
     "https://arxiv.org/pdf/2407.03333"),
    ("2022_Maze_Diffusion_Topology_Optimization_arXiv.pdf",
     "https://arxiv.org/pdf/2208.09591"),
]
for name, url in TARGETS:
    path = os.path.join(DL, name)
    try:
        status, body = fetch(url, timeout=90)
        with open(path, "wb") as fh:
            fh.write(body)
        print("  %-62s %7d bytes  %s" % (name[:62], len(body), body[:5]))
    except Exception as exc:                          # noqa: BLE001
        print("  %-62s FAILED %s" % (name[:62], str(exc)[:70]))

print("\n[done] %s" % DL)