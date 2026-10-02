# -*- coding: utf-8 -*-
"""Round 7 -- Crossref (incl. Chinese journals that deposit DOIs) + abstracts for keepers."""
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


def crossref(query, rows=12, extra=""):
    url = ("https://api.crossref.org/works?query.bibliographic=%s&rows=%d%s"
           % (urllib.parse.quote(query), rows, extra))
    d = get_json(url)
    out = []
    for it in d["message"]["items"]:
        out.append({
            "title": (it.get("title") or ["-"])[0],
            "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "doi": it.get("DOI"),
            "container": (it.get("container-title") or ["-"])[0],
            "publisher": it.get("publisher"),
            "type": it.get("type"),
        })
    return d["message"]["total-results"], out


QUERIES = [
    ("R1", "rotor critical speed machine learning prediction"),
    ("R2", "rotordynamics surrogate model deep learning design"),
    ("R3", "critical speed optimization rotor bearing genetic algorithm"),
    ("R4", "generative design diffusion model mechanical engineering"),
    ("R5", "interval optimization rotor bearing system uncertain"),
    ("R6", "parametric design generation neural network shaft rotor"),
]
CH_JOURNALS = ["机械工程学报", "振动工程学报", "航空动力学报"]

res = {"crossref": {}}
for tag, q in QUERIES:
    try:
        total, hits = crossref(q)
    except Exception as exc:                          # noqa: BLE001
        print("%-3s FAILED %s" % (tag, exc))
        continue
    res["crossref"][tag] = {"query": q, "total": total, "hits": hits}
    print("\n=== %s (%d) %s" % (tag, total, q))
    for h in hits[:8]:
        print("   %s | %-38s | %s" % (h["year"], (h["container"] or "-")[:38], h["title"][:80]))

# ---- abstracts for the keepers, so the review can describe them accurately
kp = json.load(io.open(os.path.join(HERE, "lit_keepers_20261002.json"), encoding="utf-8"))
PICK = [
    "10.1007/s00158-025-03984-2", "10.1609/aaai.v38i15.29647",
    "10.1016/j.tws.2025.113466", "10.1115/detc2025-168995",
    "10.48550/arxiv.2602.00384", "10.1115/1.4071943",
    "10.2139/ssrn.7123508", "10.1016/j.cja.2026.104109",
    "10.1115/gt2024-121854", "10.26678/abcm.cobem2025.cob2025-1583",
    "10.3390/wind4020009", "10.1016/j.ijnonlinmec.2025.105218",
    "10.1063/5.0246189", "10.59490/imdc.2024.841",
]
abstracts = {}
for doi in PICK:
    try:
        w = get_json("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi))
    except Exception as exc:                          # noqa: BLE001
        print("abs FAILED", doi, exc)
        continue
    inv = w.get("abstract_inverted_index")
    text = ""
    if inv:
        pos = {}
        for word, idxs in inv.items():
            for i in idxs:
                pos[i] = word
        text = " ".join(pos[k] for k in sorted(pos))
    abstract = w.get("abstract") if not text else text
    abstracts[doi] = {"title": w.get("title"), "year": w.get("publication_year"),
                      "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                      "abstract": (abstract or "")[:1500]}
    print("\n--- %s\n%s" % (doi, (abstracts[doi]["abstract"] or "(no abstract)")[:400]))

res["abstracts"] = abstracts
with io.open(os.path.join(HERE, "lit_round7_20261002.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, ensure_ascii=False, indent=1)
print("\n[wrote] lit_round7_20261002.json")