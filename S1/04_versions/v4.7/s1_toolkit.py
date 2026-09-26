"""Single entry point for the S1 v4.7 datatset toolkit.

    python s1_toolkit.py diagnose --features f.csv [--dataset d.csv]
    python s1_toolkit.py sample   --n 50000 --workers 20 -o output_v4.7
    python s1_toolkit.py repair   --features f.csv [--dataset d.csv] -o repaired_v4.7

Commands
--------
diagnose  Read any feature table (and optional label table) and report schema
          problems, label sanity and whether the mode budget was too small.
          Needs only numpy, never ROSS, so it is safe to run anywhere.
repair    Re-derive labels for an existing feature table with v4.7 semantics.
          The geometry is preserved exactly; only the critical speeds are
          recomputed, at 48 modes with positive-damped forward-whirl
          filtering.  Resumable, chunked, multi-process.
sample    Generate a brand new dataset from scratch with sequential
          conditional LHS plus the H1-H6 hard constraints.

repair and sample both need ROSS 2.3.0; run them with the ROSS interpreter.

Everything is dispatched to the sibling scripts using the current
interpreter, so whatever environment runs this file also runs the tools.
Additional arguments are forwarded verbatim, for example:

    python s1_toolkit.py repair --features f.csv --dataset d.csv \
        -o repaired --workers 20 --chunk-size 2000 --diagnose-only
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = {
    "sample": "s1_run_pipeline_v4_7.py",
    "repair": "s1_repair_dataset.py",
    "diagnose": "s1_diagnose_dataset.py",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0

    command = sys.argv[1]
    script = SCRIPTS.get(command)
    if script is None:
        print("unknown command: %s" % command)
        print("available: %s" % ", ".join(sorted(SCRIPTS)))
        print("")
        print(__doc__)
        return 2

    target = HERE / script
    if not target.exists():
        print("missing script: %s" % target)
        return 2

    command_line = [sys.executable, str(target)] + sys.argv[2:]
    print("[toolkit] %s -> %s" % (command, target.name))
    print("[toolkit] %s" % " ".join(command_line))
    print("")
    return subprocess.run(command_line, cwd=str(HERE)).returncode


if __name__ == "__main__":
    sys.exit(main())