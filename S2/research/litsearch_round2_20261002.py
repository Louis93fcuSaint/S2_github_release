# -*- coding: utf-8 -*-
"""Round 2 -- phrase-precise search (OpenAlex title/abstract + Semantic Scholar)."""
import io
import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "litsearch_round2_20261002.json")
UA = "rotor-s2-litreview/1.0 (mailto:914419810@qq.com)"

PHRASES = [
    ("T1", '"critical speed" AND rotor AND diffusion'),
    ("T2", 'rotor AND "generative design"'),
    ("T3", '"critical speed" AND "inverse design"'),
    ("T4", '"diffusion model" AND "design constraints"'),
    ("T5", '"conditional diffusion" AND "engineering design"'),
    ("T6", '"rotor" AND "conditional generative"'),
    ("T7", 'rotordynamics AND "machine learning"'),
    ("T8", '"critical speed" AND "design optimization" AND "neural network"'),
    ("T9", '"interval" AND "generative design"'),
    ("T10", '"performance target" AND "diffusion model" AND design'),
    ("T11", '"rotor-bearing" AND "deep learning"'),
    ("T12", '"latent diffusion" AND "mechanical design"'),
]

SS = [
    "generative design of rotor geometry conditioned on critical speed",
    "conditional diffusion model for rotor design",
    "interval conditioned generative model engineering design",
    "surrogate model screening then simulation verification design",
]


def get_json(url, headers=None, tries=3):
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
            time.sleep(2.0 * (attempt + 1))


def openalex(phrase, per_page=15):
    url = ("https://api.openalex.org/works?filter=title_and_abstract.search:%s"
           "&per-page=%d&sort=relevance_score:desc" % (urllib.parse.quote(phrase), per_page))
    data = get_json(url)
    out = []
    for w in data.get("results", []):
        out.append({"title": w.get("title") or "", "year": w.get("publication_year"),
                    "doi": w.get("doi"), "cites": w.get("cited_by_count"),
                    "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "lang": w.get("language")})
    return data["meta"]["count"], out


def semanticscholar(query, limit=12):
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?query=%s&limit=%d"
           "&fields=title,year,externalIds,abstract,venue,citationCount" %
           (urllib.parse.quote(query), limit))
    data = get_json(url)
    return data.get("total", 0), data.get("data", [])


def main():
    results = {"openalex": {}, "semanticscholar": {}}
    for tag, phrase in PHRASES:
        try:
            total, hits = openalex(phrase)
        except Exception as exc:                      # noqa: BLE001
            print("%-4s FAILED %s %s" % (tag, type(exc).__name__, exc))
            continue
        results["openalex"][tag] = {"phrase": phrase, "total": total, "hits": hits}
        print("\n=== %s (%d) %s" % (tag, total, phrase))
        for h in hits[:8]:
            print("   %-4s c=%-5s %s" % (h["year"], h["cites"], h["title"][:105]))
    for i, q in enumerate(SS):
        try:
            total, hits = semanticscholar(q)
        except Exception as exc:                      # noqa: BLE001
            print("\n=== SS%d FAILED %s %s" % (i, type(exc).__name__, exc))
            continue
        results["semanticscholar"]["SS%d" % i] = {"query": q, "total": total, "hits": hits}
        print("\n=== SS%d (%s) %s" % (i, total, q))
        for h in hits[:8]:
            print("   %-4s c=%-5s %s" % (h.get("year"), h.get("citationCount"),
                                         (h.get("title") or "")[:105]))
    with io.open(OUT, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    print("\n[wrote] %s" % OUT)


if __name__ == "__main__":
    main()