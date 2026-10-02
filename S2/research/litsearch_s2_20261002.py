# -*- coding: utf-8 -*-
"""Targeted novelty search for the S2 idea, 2026-10-02.

The question the report has to answer is not "is diffusion used in engineering"
(obviously yes) but three sharper ones:

  A. has anyone generated ROTOR designs conditioned on CRITICAL SPEEDS?
  B. has anyone conditioned a diffusion model on an INTERVAL / RANGE rather than
     a point value?
  C. has anyone done "surrogate shortlist + simulator gate" for rotors?

Writes raw hits to outputs/litsearch_20261002.json and prints a compact table.
"""
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "litsearch_20261002.json")
UA = "rotor-s2-litreview/1.0 (mailto:914419810@qq.com)"

QUERIES = [
    ("A1", "rotor critical speed generative design diffusion model"),
    ("A2", "rotordynamics design generation deep learning inverse design"),
    ("A3", "rotor bearing system optimization critical speed machine learning surrogate"),
    ("A4", "turbomachinery rotor geometry generation neural network"),
    ("B1", "diffusion model conditioned on interval range constraint engineering"),
    ("B2", "generative design performance target interval inverse problem"),
    ("B3", "conditional diffusion model engineering design performance constraints"),
    ("B4", "latent diffusion parametric design generation performance target"),
    ("C1", "surrogate model shortlist simulator verification design screening"),
    ("C2", "Bayesian optimization versus genetic algorithm rotor design"),
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except Exception as exc:                      # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))


def search(text, per_page=25):
    url = ("https://api.openalex.org/works?search=%s&per-page=%d"
           "&sort=relevance_score:desc" % (urllib.parse.quote(text), per_page))
    data = get(url)
    out = []
    for w in data.get("results", []):
        out.append({
            "id": w.get("id"),
            "doi": w.get("doi"),
            "title": w.get("title") or "",
            "year": w.get("publication_year"),
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "citations": w.get("cited_by_count"),
            "language": w.get("language"),
            "type": w.get("type"),
            "abstract": (w.get("abstract_inverted_index") and
                         " ".join(sorted(w["abstract_inverted_index"],
                                         key=lambda k: min(w["abstract_inverted_index"][k]))) or "")[:1200],
        })
    return data["meta"]["count"], out


def main():
    results = {}
    for tag, text in QUERIES:
        try:
            total, hits = search(text)
        except Exception as exc:                      # noqa: BLE001
            print("%-4s FAILED %s: %s" % (tag, type(exc).__name__, exc))
            continue
        results[tag] = {"query": text, "total": total, "hits": hits}
        print("\n=== %s  (%d hits)  %s" % (tag, total, text))
        for h in hits[:12]:
            print("  %-4s %-4s c=%-5s %s" % (h["year"], (h["language"] or "?")[:2],
                                             h["citations"], h["title"][:110]))

    with io.open(OUT, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)
    print("\n[wrote] %s" % OUT)


if __name__ == "__main__":
    main()