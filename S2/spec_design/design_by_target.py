# -*- coding: utf-8 -*-
"""One command: a target first critical speed in, a readable design sheet out.

  python design_by_target.py --target 5000 --material Steel
  python design_by_target.py --target 5000 --material Steel --spec-mode lower --tol 0.02
  python design_by_target.py --target 16000 --material Steel --disks 3-3 --bearings 2-4

A wrapper, not a new model: the same latent DDPM, the same frozen six-order
surrogate and the same protocol as `05_latent_ddpm.py sample`, with the tuned
sampling settings as defaults, plus a sheet in engineering units next to the raw
CSV.  ROSS is SI (metres, N/m), so the sheet prints mm and MN/m.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spec_common as S
import spec_eval as E

OUT_DIR = os.path.join(E.OUT, "latent_ddpm")
MM = 1000.0
MN = 1e6


def generator():
    path = os.path.join(HERE, "05_latent_ddpm.py")
    spec = importlib.util.spec_from_file_location("gen_05_latent_ddpm", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_05_latent_ddpm"] = module
    spec.loader.exec_module(module)
    return module


def verifier():
    path = os.path.join(HERE, "05_ross_verify.py")
    spec = importlib.util.spec_from_file_location("gen_05_ross_verify", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_05_ross_verify"] = module
    spec.loader.exec_module(module)
    return module


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def disk_cell(row, i):
    od, width, pos = row["d%d_OD" % i], row["d%d_W" % i], row["d%d_pos" % i]
    if od <= 0.0 and width <= 0.0:
        return "-"
    return "%.1f/%.1f@%.1f" % (od * MM, width * MM, pos * MM)


def bearing_cell(row, i):
    stiff, damp, pos = row["b%d_K" % i], row["b%d_C" % i], row["b%d_pos" % i]
    if stiff <= 0.0:
        return "-"
    return "%.1f/%.0f@%.1f" % (stiff / MN, damp, pos * MM)


def write_sheet(frame, spec, funnel, path, top):
    rows = frame.head(top) if top else frame
    aim = float(spec.get("aim") or spec.get("target") or 0.0)
    entry = (funnel or {}).get(spec["name"], {})
    verified = "cs1_ross" in frame
    if spec.get("mode") == "lower":
        title = "一阶临界转速 >= %.0f rpm" % float(spec["lower"])
        aim_note = "瞄准 %.0f rpm" % aim
    else:
        title = "一阶临界转速 %.0f rpm +-%.1f%%" % (float(spec["target"]),
                                                   100.0 * float(spec["tol"]))
        aim_note = "带内 %.0f 到 %.0f rpm" % (float(spec["lower"]), float(spec["upper"]))
    lines = []
    add = lines.append
    add("# 设计清单：%s" % spec["name"])
    add("")
    add("- 目标：%s（%s）" % (title, aim_note))
    add("- 材料：%s" % spec["material"])
    band = entry.get("family_band")
    add("- 家族约束：%s" % (", ".join(str(f).replace("_", "x") for f in band)
                          if band else "不限，18 个家族均可"))
    add("- 生成 %s 台，交付 %d 台，本清单列前 %d 台" % (entry.get("generated", "?"),
                                                    len(frame), len(rows)))
    if entry.get("support_pool_band_family") is not None:
        add("- 池子里同时满足该目标带与该家族的真实设计：%d 台（为 0 表示外推）"
            % entry["support_pool_band_family"])
    gate = entry.get("gate", {})
    if verified:
        add("- ROSS 复验：送验 %d 台，实测在带 %d 台（%.0f%%），%.2f 秒交付一台"
            % (gate.get("gate_verified", len(frame)), gate.get("gate_passed", len(frame)),
               100.0 * gate.get("gate_rate", 1.0),
               gate.get("gate_seconds_per_pass", float("nan"))))
    elif "rel_err_pct" in frame:
        add("- 交付批的代理中位相对误差（整体口径，本次清单列前 %d 台）：%.2f%%，代理在带率 %.3f"
            % (len(frame), np.median(np.abs(frame["rel_err_pct"])),
               entry.get("in_band_top200", float("nan"))))
    add("- %s单位：长度 mm，刚度 MN/m，阻尼 N s/m。"
        % ("每条都由 ROSS 复验过，cs1_ross 是实测值。" if verified
           else "这些是代理预测值，交付前应过 ROSS 复验。"))
    add("")
    header = ["#", "家族", "L", "OD", "ID"]
    header += ["盘%d 外径/宽@位置" % (i + 1) for i in range(S.MAX_DISKS)]
    header += ["轴承%d 刚度/阻尼@位置" % (i + 1) for i in range(S.MAX_BEARINGS)]
    header += (["代理 cs1", "ROSS 实测", "实测偏差"] if verified
               else ["代理 cs1", "与目标差"])
    add("| " + " | ".join(header) + " |")
    add("|" + "---|" * len(header))
    for n, (_, row) in enumerate(rows.iterrows(), start=1):
        cells = [str(n), "%dx%d" % (int(row["n_disks"]), int(row["n_bearings"])),
                 "%.0f" % (row["L"] * MM), "%.1f" % (row["OD"] * MM),
                 "%.1f" % (row["ID"] * MM)]
        cells += [disk_cell(row, i) for i in range(S.MAX_DISKS)]
        cells += [bearing_cell(row, i) for i in range(S.MAX_BEARINGS)]
        if verified:
            cells += ["%.1f" % row["cs1_pred"], "%.1f" % row["cs1_ross"],
                      "%+.2f%%" % (100.0 * (row["cs1_ross"] - aim) / aim)]
        else:
            cells += ["%.1f" % row["cs1_pred"] if "cs1_pred" in frame else "?",
                      "%+.2f%%" % row["rel_err_pct"] if "rel_err_pct" in frame else "?"]
        add("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, required=True,
                        help="target cs1 in rpm; with --spec-mode lower this is the "
                             "requirement the design has to meet")
    parser.add_argument("--material", required=True, choices=list(S.MATERIAL_ORDER))
    parser.add_argument("--tol", type=float, default=0.05,
                        help="band spec: relative half width; lower spec: aiming margin")
    parser.add_argument("--spec-mode", dest="spec_mode", choices=["band", "lower"],
                        default="band")
    parser.add_argument("--disks", default="", help="disk count range, e.g. 3-3 or 2-4")
    parser.add_argument("--bearings", default="", help="bearing count range, e.g. 2-4")
    parser.add_argument("--tag", default="armV_ema")
    parser.add_argument("--out-tag", default="")
    parser.add_argument("--n-generate", type=int, default=5000)
    parser.add_argument("--n-submit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--guide-lambda", type=float, default=0.0,
                        help="differentiable-surrogate step applied to every "
                             "predicted x0; 0 disables it")
    parser.add_argument("--fam-policy", default="uniform",
                        choices=["uniform", "learned", "pinned"])
    parser.add_argument("--sheet-top", type=int, default=50,
                        help="rows in the readable sheet, 0 for all")
    parser.add_argument("--no-sheet", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="solve the shortlist with ROSS and deliver only the "
                             "designs whose true cs1 is in band; costs roughly "
                             "one solve per submitted design")
    parser.add_argument("--verify-top", type=int, default=0,
                        help="verify only the first N rows, 0 for the whole batch")
    parser.add_argument("--verify-workers", type=int, default=18)
    parser.add_argument("--verify-timeout", type=float, default=180.0)
    parser.add_argument("--verify-n-base", type=int, default=15)
    args = parser.parse_args()

    allowed = ""
    if args.disks or args.bearings:
        allowed = "%s/%s" % (args.disks or "%d-%d" % (min(S.ND_CHOICES), max(S.ND_CHOICES)),
                             args.bearings or "%d-%d" % (min(S.NB_CHOICES), max(S.NB_CHOICES)))
    out_tag = args.out_tag or ("t%g%s" % (args.target, "lb" if args.spec_mode == "lower" else ""))

    namespace = argparse.Namespace(
        mode="sample", tag=args.tag, out_tag=out_tag, specs="",
        spec_mode=args.spec_mode, target=args.target, material=args.material,
        tol=args.tol, target_upper=0.0, spec_name="",
        fam_policy=args.fam_policy, fam_allowed=allowed, fam_floor=0.05,
        n_generate=args.n_generate, n_submit=args.n_submit, seed=args.seed,
        dump_batch=0,
        threads=args.threads, guidance=args.guidance, sample_steps=args.sample_steps,
        guide_lambda=args.guide_lambda, use_ema=1)
    generator().sample(namespace)

    name = E.adhoc_spec(args.material, args.target, args.tol,
                        name=None, mode=args.spec_mode)["name"]
    spec = next((sp for sp in E.adhoc_specs() if sp["name"] == name), None)
    csv_path = os.path.join(OUT_DIR, "%s__%s.csv" % (out_tag, name))
    if spec is None or not os.path.exists(csv_path):
        print("[sheet] skipped, no %s" % csv_path)
        return
    frame = pd.read_csv(csv_path)
    funnel = read_json(os.path.join(OUT_DIR, "funnel_%s.json" % out_tag), {})
    if args.verify:
        rows = frame.head(args.verify_top) if args.verify_top else frame
        print("[verify] %d designs through ROSS, %d workers"
              % (len(rows), args.verify_workers), flush=True)
        entry = verifier().gate_deliverable(
            rows, spec, workers=args.verify_workers, timeout=args.verify_timeout,
            n_base=args.verify_n_base, method=out_tag, spec_name=name)
        entry["mode"] = "gate"
        funnel.setdefault(name, {})["gate"] = entry
        S.save_json(os.path.join(OUT_DIR, "funnel_%s.json" % out_tag), funnel)
        gate_csv = os.path.join(OUT_DIR, entry.get("gate_file", ""))
        if not entry.get("gate_passed"):
            print("[verify] nothing passed, no deliverable")
            return
        frame = pd.read_csv(gate_csv)
        print("[verify] %d/%d designs verified in band -> %s"
              % (entry["gate_passed"], entry["gate_verified"], gate_csv))
    if args.no_sheet:
        print("[sheet] disabled")
        return
    sheet_path = os.path.join(OUT_DIR, "%s__%s_sheet.md" % (out_tag, name))
    if args.verify:
        sheet_path = os.path.join(OUT_DIR, "%s__%s_verified_sheet.md" % (out_tag, name))
    print("[sheet] %s" % write_sheet(frame, spec, funnel, sheet_path, args.sheet_top))


if __name__ == "__main__":
    main()