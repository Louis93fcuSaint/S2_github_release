# -*- coding: utf-8 -*-
"""Round 8 -- exact DOIs for the named items + Chinese-journal sweeps via Crossref."""
import io
import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = "rotor-s2-litreview/1.0 (mailto:914419810@qq.com)"


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


def bib(query, rows=6, extra=""):
    url = ("https://api.crossref.org/works?query.bibliographic=%s&rows=%d%s"
           % (urllib.parse.quote(query), rows, extra))
    d = get_json(url)
    out = []
    for it in d["message"]["items"]:
        out.append({"title": (it.get("title") or ["-"])[0],
                    "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
                    "doi": it.get("DOI"),
                    "container": (it.get("container-title") or ["-"])[0],
                    "authors": ["%s %s" % (a.get("given", ""), a.get("family", ""))
                                for a in (it.get("author") or [])][:4]})
    return out


NAMED = [
    "Intelligent Generative Design A New Mechanical Design Concept",
    "Interval optimization of rotor-bearing systems with dynamic behavior constraints",
    "Robust Optimization of Flexible Rotor Systems With Uncertain Parameters via Interval",
    "Interval Stability Analysis of a Flexible 8DOF Rotor-Bearing System",
    "Data-driven structural generative design based on diffusion model for flexible structures",
    "Inverse design of composite materials with desired mechanical behaviors based on diffusion",
    "Stochastic model updating using conditional diffusion-based probabilistic generative model",
    "Deep Analogical Generative Design and Evaluation Integration of Stable Diffusion",
]
out = {}
for q in NAMED:
    try:
        hits = bib(q, rows=3)
    except Exception as exc:                          # noqa: BLE001
        print("FAILED", q, exc)
        continue
    out[q] = hits
    print("\n== %s" % q)
    for h in hits:
        print("   %s | %-34s | %s | %s" % (h["year"], (h["container"] or "-")[:34], h["doi"], h["title"][:72]))

print("\n" + "=" * 70)
print("Chinese journals that deposit DOIs (English metadata)")
for journal, extra_q in [("Journal of Mechanical Engineering", "rotor critical speed"),
                         ("Journal of Mechanical Engineering", "surrogate model optimization design"),
                         ("Journal of Vibration Engineering", "rotor critical speed"),
                         ("Journal of Aerospace Power", "rotor critical speed")]:
    try:
        hits = bib(extra_q, rows=6, extra="&query.container-title=" + urllib.parse.quote(journal))
    except Exception as exc:                          # noqa: BLE001
        print("FAILED", journal, exc)
        continue
    out[journal + " :: " + extra_q] = hits
    print("\n== %s :: %s" % (journal, extra_q))
    for h in hits:
        print("   %s | %s | %s" % (h["year"], h["doi"], h["title"][:78]))

with io.open(os.path.join(HERE, "lit_round8_20261002.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print("\n[wrote] lit_round8_20261002.json")