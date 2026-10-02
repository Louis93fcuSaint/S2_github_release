# -*- coding: utf-8 -*-
"""Round 3 -- Chinese-language search, arXiv, and full metadata for the keepers."""
import io
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
UA = "rotor-s2-litreview/1.0 (mailto:914419810@qq.com)"

ZH = [
    ("Z1", '"临界转速" AND "代理模型"'),
    ("Z2", '"转子" AND "生成式设计"'),
    ("Z3", '"临界转速" AND "神经网络" AND 设计'),
    ("Z4", '"转子" AND "扩散模型"'),
    ("Z5", '"区间" AND "临界转速"'),
    ("Z6", '"转子动力学" AND "机器学习"'),
    ("Z7", '"转子" AND "深度学习" AND "优化设计"'),
]

ARXIV = [
    ("X1", 'all:"rotordynamics" AND all:"machine learning"'),
    ("X2", 'abs:"critical speed" AND abs:"generative"'),
    ("X3", 'abs:"diffusion model" AND abs:"engineering design" AND abs:"constraint"'),
    ("X4", 'abs:"inverse design" AND abs:"conditional diffusion"'),
    ("X5", 'abs:"range conditioning" OR abs:"interval conditioning"'),
]


def get_json(url, tries=3, headers=None):
    hdr = {"User-Agent": UA}
    hdr.update(headers or {})
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3.0 * (attempt + 1))


def openalex(phrase, per_page=12, extra=""):
    url = ("https://api.openalex.org/works?filter=title_and_abstract.search:%s%s"
           "&per-page=%d&sort=relevance_score:desc"
           % (urllib.parse.quote(phrase), extra, per_page))
    data = get_json(url)
    out = []
    for w in data.get("results", []):
        auth = [a["author"]["display_name"] for a in (w.get("authorships") or [])][:3]
        out.append({"title": w.get("title") or "", "year": w.get("publication_year"),
                    "doi": w.get("doi"), "cites": w.get("cited_by_count"),
                    "lang": w.get("language"),
                    "authors": auth,
                    "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "oa": ((w.get("open_access") or {}).get("oa_url"))})
    return data["meta"]["count"], out


def arxiv(query, max_results=8):
    url = ("http://export.arxiv.org/api/query?search_query=%s&start=0&max_results=%d"
           "&sortBy=relevance" % (urllib.parse.quote(query), max_results))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        raw = r.read()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    out = []
    for e in root.findall("a:entry", ns):
        out.append({"title": " ".join(e.find("a:title", ns).text.split()),
                    "id": e.find("a:id", ns).text,
                    "published": e.find("a:published", ns).text[:10],
                    "summary": " ".join(e.find("a:summary", ns).text.split())[:400]})
    return out


def main():
    res = {"zh": {}, "arxiv": {}, "details": {}}
    for tag, phrase in ZH:
        try:
            total, hits = openalex(phrase, extra=",language:zh")
        except Exception as exc:                      # noqa: BLE001
            print("%-3s FAILED %s" % (tag, exc))
            continue
        res["zh"][tag] = {"phrase": phrase, "total": total, "hits": hits}
        print("\n=== %s (%d) %s" % (tag, total, phrase))
        for h in hits[:10]:
            print("   %-5s %-34s %s" % (h["year"], (h["venue"] or "-")[:34], h["title"][:80]))
    for tag, q in ARXIV:
        try:
            hits = arxiv(q)
        except Exception as exc:                      # noqa: BLE001
            print("%-3s FAILED %s" % (tag, exc))
            continue
        res["arxiv"][tag] = {"query": q, "hits": hits}
        print("\n=== %s %s" % (tag, q))
        for h in hits:
            print("   %s  %s" % (h["published"], h["title"][:95]))

    # full records for the keepers (by DOI) so the review can cite them exactly
    DOIS = [
        "10.1016/j.jmps.2024.105893",
        "10.1016/j.cma.2023.116270",
        "10.1016/j.ress.2025.111214",
        "10.1177/09544062251334612",
        "10.1016/j.advengsoft.2024.103745",
        "10.1080/17452759.2024.2411824",
        "10.1016/j.oceaneng.2023.116588",
        "10.1016/j.engstruct.2025.120269",
        "10.1016/j.apm.2025.116310",
        "10.1016/j.ymssp.2025.112975",
    ]
    for doi in DOIS:
        try:
            w = get_json("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi))
        except Exception as exc:                      # noqa: BLE001
            print("detail FAILED %s %s" % (doi, exc))
            continue
        res["details"][doi] = {"title": w.get("title"), "year": w.get("publication_year"),
                               "doi": w.get("doi"),
                               "authors": [a["author"]["display_name"] for a in (w.get("authorships") or [])][:5],
                               "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                               "oa": (w.get("open_access") or {}).get("oa_url")}
        print("\ndetail %s -> %s" % (doi, (w.get("title") or "")[:90]))

    with io.open(os.path.join(HERE, "litsearch_round3_20261002.json"), "w",
                 encoding="utf-8") as handle:
        json.dump(res, handle, ensure_ascii=False, indent=1)
    print("\n[wrote] litsearch_round3_20261002.json")


if __name__ == "__main__":
    main()