"""Promote a completed 48-mode relabel into the canonical v4.7 file set.

The 48-mode relabel that is currently running was launched from the v4.6
scripts, so it writes ``relabeled_v4.6_*.csv`` style names and its label table
only carries cs_1..cs_3 (the v4.6 column set).  v4.7 renames the outputs and
widens the label table to cs_1..cs_6.

This script does the rename *without* touching ROSS: it re-reads the long-form
modal table that the relabel already wrote, rebuilds the forward-whirl list per
rotor, and emits the v4.7 files.  No FEM is repeated.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

N_ORDERS = 6
TARGETS = ["cs_{}_rpm".format(i) for i in range(1, N_ORDERS + 1)]


def forward_lists(modes_path: Path):
    forward = collections.defaultdict(list)
    with modes_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("whirl_direction") != "Forward":
                continue
            try:
                wd = float(row["wd_rpm"])
                log_dec = float(row["log_dec"])
            except (TypeError, ValueError):
                continue
            if wd > 0 and log_dec > 0:
                forward[int(row["source_row_id"])].append(wd)
    return {key: sorted(values) for key, values in forward.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True,
                        help="directory holding relabel_*_v4.6_all.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stem", default="v4.6", help="source file stem")
    args = parser.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    old = args.stem
    new = "v4.7"

    modes_in = src / "relabel_modes_{}_all.csv".format(old)
    labels_in = src / "relabeled_{}_all.csv".format(old)
    audit_in = src / "relabel_audit_{}_all.csv".format(old)
    summary_in = src / "relabel_summary_{}_all.json".format(old)

    print("[promote] reading forward modes from", modes_in, flush=True)
    forward = forward_lists(modes_in)
    print("[promote] rotors with forward modes:", len(forward), flush=True)

    labels_out = out / "relabeled_{}_all.csv".format(new)
    with labels_in.open(newline="", encoding="utf-8") as fin, \
            labels_out.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        for target in TARGETS:
            if target not in fieldnames:
                fieldnames.append(target)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        n_gained = 0
        for row in reader:
            values = forward.get(int(row["source_row_id"]), [])[:N_ORDERS]
            for index, target in enumerate(TARGETS):
                if target not in row or row.get(target) in ("", None):
                    if index < len(values):
                        row[target] = values[index]
            n_gained += 1
        print("[promote] label rows written:", n_gained, flush=True)

    for name in ("relabel_modes", "relabel_audit"):
        src_path = src / "{}_{}_all.csv".format(name, old)
        dst_path = out / "{}_{}_all.csv".format(name, new)
        with src_path.open(newline="", encoding="utf-8") as fin, \
                dst_path.open("w", newline="", encoding="utf-8") as fout:
            while True:
                chunk = fin.read(1 << 20)
                if not chunk:
                    break
                fout.write(chunk)
        print("[promote] copied", dst_path.name, flush=True)

    summary = json.loads(summary_in.read_text(encoding="utf-8"))
    summary["version"] = new
    summary["promoted_from"] = str(src)
    summary["num_modes"] = 48
    summary["label_columns"] = TARGETS
    summary["note"] = (
        "48-mode relabel promoted from the v4.6-named chunk run; "
        "cs_4..cs_6 recovered from the long-form modal table, cs_1..cs_3 untouched."
    )
    summary["outputs"] = {
        "labels": str(labels_out),
        "modes": str(out / "relabel_modes_{}_all.csv".format(new)),
        "audit": str(out / "relabel_audit_{}_all.csv".format(new)),
    }
    summary_out = out / "relabel_summary_{}_all.json".format(new)
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print("[done]", summary_out, flush=True)


if __name__ == "__main__":
    main()