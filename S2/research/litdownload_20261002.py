# -*- coding: utf-8 -*-
"""Round 9 -- download whatever is openly available, into references/06_ (new)."""
import io
import os
import ssl
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))           # 大学生创新创业训练计划
REF = os.path.join(PROJ, "references")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

TARGETS = [
    ("01_rotordynamics_simulation",
     "2022_ARRID_ANN_Based_Rotordynamics_Robust_Integrated_Design_arXiv.pdf",
     "https://arxiv.org/pdf/2208.12640"),
    ("03_generative_design_diffusion",
     "2022_Maze_Geometric_Deep_Learning_Diffusion_Topology_Optimization_arXiv.pdf",
     "https://arxiv.org/pdf/2206.04617"),
    ("03_generative_design_diffusion",
     "2024_Chen_CcDPM_Continuous_Conditional_Diffusion_Inverse_Design_AAAI.pdf",
     "https://ojs.aaai.org/index.php/AAAI/article/download/29647/31235"),
]


def fetch(url, timeout=120):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()


for folder, name, url in TARGETS:
    dest = os.path.join(REF, folder, name)
    try:
        body = fetch(url)
        if not body[:5].startswith(b"%PDF"):
            print("  %-70s NOT A PDF (%r)" % (name[:70], body[:12]))
            continue
        with open(dest, "wb") as fh:
            fh.write(body)
        print("  OK  %-66s %8.2f MB" % (name[:66], len(body) / 1e6))
    except Exception as exc:                          # noqa: BLE001
        print("  FAIL %-66s %s" % (name[:66], str(exc)[:60]))

# also move the newly downloaded ones from research/pdf_new
new = os.path.join(HERE, "pdf_new")
if os.path.isdir(new):
    for f in os.listdir(new):
        src = os.path.join(new, f)
        if not os.path.isfile(src) or not f.lower().endswith(".pdf"):
            continue
        dest = os.path.join(REF, "03_generative_design_diffusion", f)
        if os.path.exists(dest):
            print("  (already archived) %s" % f)
            continue
        with open(src, "rb") as a, open(dest, "wb") as b:
            b.write(a.read())
        print("  moved %s -> 03_generative_design_diffusion" % f)
print("\nreferences updated at %s" % REF)