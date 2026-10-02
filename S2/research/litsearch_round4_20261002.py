# -*- coding: utf-8 -*-
"""Round 4 -- exact records for the keepers (looked up by OpenAlex ID, never by a
guessed DOI), Chinese retry without the language filter, and OA PDF download."""
import io
import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = "rotor-s2-litreview/1.0 (mailto:914419810@qq.com)"
STORE = os.path.join(HERE, "litsearch_round2_20261002.json")


def get_json(url, tries=3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3.0 * (attempt + 1))


KEY = ["rotor", "rotordynamic", "critical speed", "bearing", "turbomachinery",
       "diffusion", "generative", "inverse design", "topology optimization",
       "ship hull", "metamaterial", "interval", "constraint", "surrogate",
       "performance", "gearbox", "wind turbine", "compressor", "shaft"]


def score(title):
    low = title.lower()
    return sum(1 for k in KEY if k in low)


def main():
    store = json.load(io.open(STORE, encoding="utf-8"))
    keep = {}
    for tag, block in store.get("openalex", {}).items():
        for h in block.get("hits", []):
            if not h.get("title") or not h.get("doi"):
                continue
            s = score(h["title"])
            if s >= 2:
                keep[h["doi"]] = dict(h, query=tag, score=s)
    print("keepers with a DOI:", len(keep))

    ids = {}
    for doi, h in keep.items():
        try:
            w = get_json("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi))
        except Exception as exc:                      # noqa: BLE001
            print("  detail failed", doi, exc)
            continue
        ids[doi] = {
            "id": w.get("id"), "doi": w.get("doi"), "title": w.get("title"),
            "year": w.get("publication_year"),
            "authors": [a["author"]["display_name"] for a in (w.get("authorships") or [])][:6],
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "cites": w.get("cited_by_count"), "lang": w.get("language"),
            "oa_url": (w.get("open_access") or {}).get("oa_url"),
            "oa_status": (w.get("open_access") or {}).get("oa_status"),
            "abstract": "", "query": h["query"]}
        print("  %-4s %-30s %s" % (ids[doi]["year"], (ids[doi]["venue"] or "-")[:30],
                                   (ids[doi]["title"] or "")[:78]))

    # Chinese retry without the language filter
    zh = {}
    for phrase in ['"临界转速" AND "神经网络"', '"转子" AND "生成式"', '"临界转速" AND "优化"',
                   '"转子动力学" AND "代理模型"', '"参数化" AND "转子" AND "设计"']:
        try:
            url = ("https://api.openalex.org/works?filter=title_and_abstract.search:%s"
                   "&per-page=8&sort=relevance_score:desc" % urllib.parse.quote(phrase))
            data = get_json(url)
        except Exception as exc:                      # noqa: BLE001
            print("zh failed", phrase, exc)
            continue
        zh[phrase] = {"total": data["meta"]["count"], "hits": [
            {"title": w.get("title"), "year": w.get("publication_year"),
             "doi": w.get("doi"), "lang": w.get("language"),
             "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name")}
            for w in data.get("results", [])]}
        print("\nzh %s -> %d" % (phrase, data["meta"]["count"]))
        for h in zh[phrase]["hits"]:
            print("   %s | %s | %s" % (h["year"], (h["venue"] or "-")[:28], (h["title"] or "")[:70]))

    with io.open(os.path.join(HERE, "lit_keepers_20261002.json"), "w", encoding="utf-8") as fh:
        json.dump({"keepers": ids, "chinese_retry": zh}, fh, ensure_ascii=False, indent=1)
    print("\n[wrote] lit_keepers_20261002.json")


if __name__ == "__main__":
    main()