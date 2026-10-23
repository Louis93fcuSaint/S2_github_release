# -*- coding: utf-8 -*-
"""Apply the verified round-2 edits to the manuscript copy (never the master MSSP files).

'@' is used as a stand-in for a LaTeX backslash so that this file survives shell/JSON escaping.
"""
from pathlib import Path

BS = chr(92)
def tex(s: str) -> str:
    return s.replace("@", BS)

MS = Path(__file__).resolve().parents[1] / "manuscript"
changed = []

# ---------------------------------------------------------------- sec01: research gap
p = MS / "sec01.tex"
s = p.read_text(encoding="utf-8")
start = s.find("Second, related generative studies")
end = s.find(" Third, predicted performance is not equivalent", start)
assert start >= 0 and end > start
new_gap = tex(
    "Second, related generative studies mainly use point-valued performance conditions. Interval or range "
    "conditioning has been studied in general generative models @cite{r31,r32}, but no rotor-specific study "
    "was found that uses user-specified critical-speed intervals, especially joint intervals over several "
    "orders, as generation conditions; the intervals in rotor interval finite-element and robust-optimization "
    "studies @cite{r25,r26,r27,r28} describe input-parameter uncertainty instead "
    "(Sec.~@ref{sec:conditions}).")
s = s[:start] + new_gap + s[end:]

# ---------------------------------------------------------------- sec01: interval-conditioning prior art
anchor = ("Different from the above studies, this paper addresses specification-driven rotor inverse design")
assert anchor in s
insert = tex(
    "Range or interval conditions have also been studied outside rotor design. Range-GAN @cite{r31} conditions "
    "a generative adversarial network on ranges of target performance, and Simformer @cite{r32} trains a "
    "transformer-based diffusion model on the joint distribution of parameters and observations, supports "
    "arbitrary conditional queries including missing data, and explicitly discusses conditioning on "
    "observation intervals through diffusion guidance. These methods show that interval conditioning is not, "
    "by itself, a new capability. They are not concerned with rotor parameterisations, with joint constraints "
    "on several critical-speed orders, or with per-design numerical verification under a fixed re-computation "
    "budget.\n\n"
    "Different from these studies, this paper addresses specification-driven rotor inverse design")
s = s.replace(anchor, insert)
p.write_text(s, encoding="utf-8")
changed.append(p.name)

# ---------------------------------------------------------------- refs
p = MS / "refs.tex"
s = p.read_text(encoding="utf-8")
assert "thebibliography}{30}" in s
s = s.replace("thebibliography}{30}", "thebibliography}{33}")
end_bib = tex("@end{thebibliography}")
assert s.rstrip().endswith(end_bib)
bib = tex(
    "@bibitem{r31} A. Heyrani Nobari, W. Chen, F. Ahmed, Range-GAN: Range-Constrained Generative Adversarial "
    "Network for Conditioned Design Synthesis, in: Proceedings of the ASME International Design Engineering "
    "Technical Conferences and Computers and Information in Engineering Conference (IDETC-CIE), 2021, "
    "DETC2021-69963. https://doi.org/10.1115/DETC2021-69963.\n\n"
    "@bibitem{r32} M. Gloeckler, M. Deistler, C. Weilbach, F. Wood, J.H. Macke, All-in-one simulation-based "
    "inference, in: Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235, "
    "2024, pp. 15735--15766. arXiv:2404.09636.\n\n"
    "@bibitem{r33} B. Zhuang, A. Gallet, D. Smyl, Inverse structural design with generative and probabilistic "
    "autoencoders and diffusion models, Eng. Appl. Artif. Intell. 161 (2025) 112143. "
    "https://doi.org/10.1016/j.engappai.2025.112143.\n\n")
s = s.rstrip()[: -len(end_bib)] + bib + end_bib + "\n"
p.write_text(s, encoding="utf-8")
changed.append(p.name)

# ---------------------------------------------------------------- sec04b: Case C
p = MS / "sec04b.tex"
s = p.read_text(encoding="utf-8")
anchor = ("This result shows that the proposed method can generate candidates satisfying several "
          "critical-speed constraints simultaneously.")
assert anchor in s
add = tex(
    "\n\nAll three designs pass H1--H6 and their critical-speed specifications, but H4 constrains only the "
    "axial distance between a bearing centre and a disk centre. In Case C, the sixth disk (centre 1364.7~mm, "
    "width 115.9~mm) extends from 1306.7 to 1422.7~mm, while bearing b2 is centred at 1419.2~mm. The H4 "
    "centre distance is 54.5~mm, yet the disk body overlaps the bearing centre by 3.4~mm; the minimum "
    "disk-edge clearance is $-3.4$~mm, i.e.@ 13.4~mm short of a 10~mm edge-clearance requirement. Case C is "
    "retained here to make this distinction explicit and is not presented as an assembly-ready design. In "
    "the same 200-design submission, 174 designs pass ROSS and 70 of them also satisfy the 10~mm disk-edge "
    "clearance.")
s = s.replace(anchor, anchor + add)
p.write_text(s, encoding="utf-8")
changed.append(p.name)

# ---------------------------------------------------------------- appendix note
p = MS / "appendix_a.tex"
s = p.read_text(encoding="utf-8")
old = ("H4 checks only the axial distance between bearing centers and disk centers and does not guarantee "
       "assembly clearance between disk bodies and bearing solids.")
new = old + " For Case C the minimum disk-edge clearance is $-3.4$~mm."
if old in s:
    s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    changed.append(p.name)
else:
    print("WARNING: appendix note not replaced")

print("updated:", ", ".join(changed))
