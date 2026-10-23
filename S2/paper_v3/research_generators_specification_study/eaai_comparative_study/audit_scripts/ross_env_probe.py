# -*- coding: utf-8 -*-
"""Record ROSS version and source-level identities without importing project code.

Usage: python ross_env_probe.py OUTPUT_JSON
"""
import hashlib
import inspect
import json
import os
import sys


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    out_path = sys.argv[1]
    import ross
    import ross.rotor_assembly as rotor_mod
    from ross.rotor_assembly import Rotor
    from ross.results import ModalResults
    out = {
        "python": sys.version,
        "executable": sys.executable,
        "ross_version": getattr(ross, "__version__", "unknown"),
        "ross_package": os.path.dirname(os.path.abspath(ross.__file__)),
        "ross_init_file": os.path.abspath(ross.__file__),
        "ross_init_sha256": hashlib.sha256(open(ross.__file__, "rb").read()).hexdigest(),
        "rotor_module_file": os.path.abspath(rotor_mod.__file__),
        "rotor_module_sha256": hashlib.sha256(open(rotor_mod.__file__, "rb").read()).hexdigest(),
    }
    whirl = ModalResults.whirl_direction
    if isinstance(whirl, property):
        whirl = whirl.fget
    funcs = {}
    for name, obj in [
        ("Rotor._eigen", Rotor._eigen),
        ("Rotor.run_modal", Rotor.run_modal),
        ("Rotor.run_critical_speed", Rotor.run_critical_speed),
        ("ModalResults.whirl_direction", whirl),
    ]:
        try:
            src = inspect.getsource(obj)
            funcs[name] = {"signature": str(inspect.signature(obj)),
                           "source_sha256": sha(src), "lines": len(src.splitlines()),
                           "source": src}
        except Exception as exc:
            funcs[name] = {"error": repr(exc)}
    out["functions"] = funcs
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
