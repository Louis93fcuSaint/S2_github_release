# -*- coding: utf-8 -*-
"""Convert the reviewed Chapter 4 LaTeX source into Markdown.

The prose is converted from chapter4_revised.tex. Table bodies are repeated in
Markdown here so that the review copy does not depend on a LaTeX converter.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "chapter4_revised.tex"
OUT = ROOT / "chapter4_revised_en.md"

TABLES = {
    "tab:ch4_protocol": """**Table 1.** Frozen specifications and evaluation protocol for the TST comparison at sampling seed 42.

| ID | Material / family | Orders and form | Frozen requirement (rpm) | Library support |
|---|---|---|---:|---:|
| TST01 | Steel / free | 1, one-sided | cs1 ≥ 15,966.825 | 3,319 |
| TST02 | Aluminum / 3d–3b | 1, interval | cs1 ∈ [5,282.079, 6,451.546] | 337 |
| TST04 | Steel / 6d–4b | 1–2, joint point | cs1 ∈ [3,105.122, 3,431.977]; cs2 ∈ [5,186.288, 5,732.212] | 14 |
| TST07 | Steel / free | 1–3, joint point | cs1 ∈ [3,460.280, 3,824.520]; cs2 ∈ [7,254.153, 8,017.748]; cs3 ∈ [19,981.825, 22,085.175] | 10 |
| TST09 | Titanium / free | 1–2, joint interval | cs1 ∈ [4,008.079, 4,703.524]; cs2 ∈ [8,389.189, 9,844.805] | 555 |

*Note:* every method generated 5,000 attempts per specification. A uniform sample of 100 attempt IDs estimated raw-pool behaviour; a separate 200-design shortlist was used for ROSS verification. All lists used sampling seed 42. DDPM denotes the separately trained interval-conditioned checkpoint, sampled for 25 steps with classifier-free guidance 1.0 and no surrogate-gradient guidance. CVAE is a separately trained conditional variational autoencoder with the same 21-dimensional interval condition. Retrieval+jitter uses stored labels in the frozen reference library to select source designs. The shortlist uses the common surrogate gate, uniform selection among predicted in-band unique designs, and a deterministic out-of-band fallback ordered by violation and candidate ID. ROSS 1.6.1 results are cached only under an exact design, material, unit, solver-version, and acceptance-protocol key.""",
    "tab:ch4_cv": """**Table 2.** Five-fold GroupKFold validation of the deployed surrogate. MAPE values are percentages.

| Fold | Test rows | Unadjusted MAPE | Adjusted MAPE | Median APE | P90 APE |
|---:|---:|---:|---:|---:|---:|
| 0 | 38,018 | 4.596 | 4.596 | 2.553 | 10.424 |
| 1 | 40,442 | 4.512 | 4.511 | 2.478 | 10.318 |
| 2 | 40,453 | 4.865 | 4.865 | 2.745 | 11.049 |
| 3 | 40,442 | 11.857 | 4.537 | 2.514 | 10.391 |
| 4 | 40,460 | 4.388 | 4.388 | 2.405 | 10.000 |
| Mean | — | 6.044 | 4.579 | 2.539 | 10.436 |

*Note:* adjusted scores remove cs1 < 100 rpm labels from first-order scoring; other-order labels are unchanged.""",
    "tab:ch4_tst": """**Table 3.** TST ROSS specification satisfaction at sampling seed 42. Raw-pool values use 100 uniformly sampled attempts; shortlist values use 200 surrogate-gated candidates.

| Specification | DDPM raw | CVAE raw | Retrieval raw | DDPM shortlist | CVAE shortlist | Retrieval shortlist |
|---|---:|---:|---:|---:|---:|---:|
| TST01 | 55 (55.0%) | 96 (96.0%) | 99 (99.0%) | 192 (96.0%) | 199 (99.5%) | 198 (99.0%) |
| TST02 | 7 (7.0%) | 43 (43.0%) | 74 (74.0%) | 183 (91.5%) | 189 (94.5%) | 176 (88.0%) |
| TST04 | 3 (3.0%) | 2 (2.0%) | 55 (55.0%) | 77 (38.5%) | 111 (55.5%) | 160 (80.0%) |
| TST07 | 0 (0.0%) | 0 (0.0%) | 20 (20.0%) | 4 (2.0%) | 16 (8.0%) | 133 (66.5%) |
| TST09 | 5 (5.0%) | 16 (16.0%) | 51 (51.0%) | 136 (68.0%) | 163 (81.5%) | 165 (82.5%) |
| Macro | 70 (14.0%) | 157 (31.4%) | 299 (59.8%) | 592 (59.2%) | 678 (67.8%) | 832 (83.2%) |

*Note:* denominators remain fixed at 100 and 200. Five solver errors and one timeout occurred among the 4,500 logical slots; all remain in their denominators. Rates are ROSS outcomes, not surrogate in-band counts.""",
    "tab:ch4_geometry": """**Table 4.** Geometry screening of the TST shortlists. Entries are numbers of designs; all denominators are 200.

| Specification | DDPM ROSS | CVAE ROSS | Retrieval ROSS | DDPM edge | CVAE edge | Retrieval edge | DDPM joint | CVAE joint | Retrieval joint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TST01 | 192 | 199 | 198 | 67 | 56 | 81 | 64 | 55 | 79 |
| TST02 | 183 | 189 | 176 | 80 | 89 | 66 | 74 | 80 | 59 |
| TST04 | 77 | 111 | 160 | 27 | 23 | 16 | 12 | 10 | 12 |
| TST07 | 4 | 16 | 133 | 85 | 72 | 91 | 0 | 5 | 63 |
| TST09 | 136 | 163 | 165 | 77 | 102 | 69 | 46 | 85 | 57 |
| Total | 592 | 678 | 832 | 336 | 342 | 323 | 196 | 235 | 270 |
| Rate | 59.2% | 67.8% | 83.2% | 33.6% | 34.2% | 32.3% | 19.6% | 23.5% | 27.0% |

*Note:* H4 passed for all 200 submitted designs in every pool because it is enforced by the decoder. The edge criterion is an axial bearing-centre-to-disk-end-face distance of at least 10 mm and is not a complete assembly or manufacturing check.""",
    "tab:ch4_caseC": """**Table 5.** Case C geometry and critical-speed outcomes.

| Quantity | Value |
|---|---:|
| Shaft length | 1,576.93 mm |
| Disks / bearings | 6 / 3 |
| Nearest disk d5 centre / width | 1,364.699 / 115.912 mm |
| Disk d5 right end face | 1,422.655 mm |
| Bearing b2 centre | 1,419.237 mm |
| H4 bearing–disk centre distance | 54.538 mm |
| Minimum disk-edge clearance | −3.418 mm |
| Shortfall from 10 mm edge clearance | 13.418 mm |
| ROSS cs1 / cs2 | 4,156.59 / 7,607.86 rpm |
| Two-order specification | satisfied |""",
    "tab:ch4_diversity": """**Table 6.** Accepted-set composition and parameter dispersion after joint ROSS and 10 mm edge acceptance.

| Specification | Method | Accepted | Families | Family entropy | Within-family NN | Reference NN |
|---|---|---:|---:|---:|---:|---:|
| TST01 | DDPM | 64 | 14 | 2.472 | 3.771 | 2.610 |
| TST01 | CVAE | 55 | 12 | 2.202 | 2.482 | 1.982 |
| TST01 | Retrieval | 79 | 9 | 1.904 | 3.242 | 0.236 |
| TST02 | DDPM | 74 | 1 | 0.000 | 2.430 | 2.009 |
| TST02 | CVAE | 80 | 1 | 0.000 | 2.086 | 2.044 |
| TST02 | Retrieval | 59 | 1 | 0.000 | 2.895 | 0.282 |
| TST04 | DDPM | 12 | 1 | 0.000 | 3.507 | 2.953 |
| TST04 | CVAE | 10 | 1 | 0.000 | 3.262 | 3.146 |
| TST04 | Retrieval | 12 | 1 | 0.000 | 0.462 | 0.366 |
| TST07 | DDPM | 0 | 0 | — | — | — |
| TST07 | CVAE | 5 | 5 | 1.609 | — | — |
| TST07 | Retrieval | 63 | 4 | 1.132 | — | — |
| TST09 | DDPM | 46 | 14 | 2.386 | 2.901 | 1.907 |
| TST09 | CVAE | 85 | 16 | 2.520 | 2.762 | 1.878 |
| TST09 | Retrieval | 57 | 13 | 2.414 | 3.580 | 0.262 |

*Note:* distances are in standardized active-parameter units and are estimated from 50 equal-size subsamples per specification; the common counts are 55, 59, 10, 0, and 46 for TST01, TST02, TST04, TST07, and TST09. Entropy is computed from configuration-family counts. Larger distances indicate greater parameter dispersion, not higher engineering quality.""",
}

FIGURES = {
    "fig:tst_rates": """![Figure 1](figures/fig_ch4_tst.png)

**Figure 1.** TST results at seed 42: (a) ROSS satisfaction in the 100-design uniform sample; (b) ROSS satisfaction in the 200-design surrogate-gated shortlist; (c) joint dynamic–geometric acceptance after the 10 mm disk-edge screen. Bars are specification-level rates; macro averages are reported in Tables 3 and 4 and are not drawn.""",
}

SECTION_MAP = {
    "sec:setup": "Section 4.1",
    "sec:surrogate_analysis": "Section 4.2",
    "sec:accuracy": "Section 4.2",
    "sec:specresults": "Section 4.3",
    "sec:tstsubset": "Section 4.3",
    "sec:baselines": "Section 4.3",
    "sec:cases": "Section 4.4",
    "sec:ranking": "Section 4.5",
    "sec:prescreen": "Section 4.5",
    "sec:ablation": "Section 4.5",
    "sec:ablation_params": "Section 4.5",
}

TABLE_MAP = {
    "tab:ch4_protocol": "Table 1",
    "tab:ch4_cv": "Table 2",
    "tab:ch4_tst": "Table 3",
    "tab:ch4_geometry": "Table 4",
    "tab:ch4_caseC": "Table 5",
    "tab:ch4_diversity": "Table 6",
}


def inline(text: str) -> str:
    text = re.sub(r"Table~\\ref\{([^}]+)\}", lambda m: TABLE_MAP.get(m.group(1), m.group(1)), text)
    text = re.sub(r"Secs\.~\\ref\{([^}]+)\}\s*and~\\ref\{([^}]+)\}", lambda m: SECTION_MAP.get(m.group(1), m.group(1)) + " and " + SECTION_MAP.get(m.group(2), m.group(2)), text)
    text = re.sub(r"Secs\.~\\ref\{([^}]+)\}", lambda m: SECTION_MAP.get(m.group(1), m.group(1)), text)
    text = re.sub(r"Sec\.~\\ref\{([^}]+)\}", lambda m: SECTION_MAP.get(m.group(1), m.group(1)), text)
    text = text.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_")
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
    text = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\pm", "±").replace("\\ge", "≥").replace("\\leq", "≤")
    text = text.replace("\\in", "∈").replace("\\times", "×").replace("\\approx", "≈")
    text = text.replace("\\,", " ").replace("{,}", ",").replace("\\ ", " ")
    text = text.replace("$P_{90}$", "P90").replace("P_{90}", "P90")
    text = re.sub(r"cs_\{(\d)\}", r"cs\1", text)
    text = re.sub(r"\$\\mathrm\{cs\}_\{(\d)\}\$", lambda m: f"cs{m.group(1)}", text)
    text = re.sub(r"\$\\mathrm\{cs\}_\{(\d)\}\s*\\in\s*\[([^\]]+)\]\$", lambda m: f"cs{m.group(1)} ∈ [{m.group(2)}]", text)
    text = re.sub(r"\$([^$]*)\$", lambda m: m.group(1), text)
    text = text.replace("~", " ")
    text = text.replace("---", "—").replace("--", "–")
    text = re.sub(r"\\ref\{([^}]+)\}", lambda m: TABLE_MAP.get(m.group(1), FIGURES.get(m.group(1), SECTION_MAP.get(m.group(1), m.group(1)))), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    src = TEX.read_text(encoding="utf-8")
    src = re.sub(r"(?s)\\begin\{table\*?\}.*?\\end\{table\*?\}", lambda m: "\n\n@@TABLE@@\n\n", src)
    src = re.sub(r"(?s)\\begin\{figure\}.*?\\end\{figure\}", lambda m: "\n\n@@FIGURE@@\n\n", src)
    src = re.sub(r"\\label\{[^}]*\}", "", src)
    src = re.sub(r"\\section\{([^}]*)\}", r"# \1", src)
    src = re.sub(r"\\subsection\{([^}]*)\}", r"## \1", src)
    src = re.sub(r"\\paragraph\{([^}]*)\}", r"**\1.**", src)
    paragraphs = []
    for block in re.split(r"\n\s*\n", src):
        block = block.strip()
        if not block:
            continue
        if block in ("@@TABLE@@", "@@FIGURE@@"):
            paragraphs.append(block)
        elif block.startswith("#"):
            paragraphs.append(inline(block))
        else:
            paragraphs.append(inline(block))
    table_blocks = re.findall(r"(?s)\\begin\{table\*?\}.*?\\end\{table\*?\}", TEX.read_text(encoding="utf-8"))
    fig_blocks = re.findall(r"(?s)\\begin\{figure\}.*?\\end\{figure\}", TEX.read_text(encoding="utf-8"))
    table_labels = [re.search(r"\\label\{([^}]+)\}", b).group(1) for b in table_blocks]
    fig_labels = [re.search(r"\\label\{([^}]+)\}", b).group(1) for b in fig_blocks]
    ti = fi = 0
    out = []
    for block in paragraphs:
        if block == "@@TABLE@@":
            out.append(TABLES[table_labels[ti]])
            ti += 1
        elif block == "@@FIGURE@@":
            out.append(FIGURES[fig_labels[fi]])
            fi += 1
        else:
            out.append(block)
    text = "\n\n".join(out) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
