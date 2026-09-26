# -*- coding: utf-8 -*-
"""One rotor, one process.  Kept tiny so a pathological design cannot block the
whole verification run -- the caller enforces the timeout."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    payload = json.loads(sys.argv[1])
    import spec_common as S
    x = np.asarray(payload["x"], dtype=np.float64)
    forward = S.run_ross(x, payload["material"], int(payload["nd"]),
                         int(payload["nb"]), n_base=int(payload.get("n_base", 15)))
    return {"ok": True, "forward": [float(v) for v in forward]}


if __name__ == "__main__":
    try:
        print(json.dumps(main()))
    except Exception as exc:                                  # noqa: BLE001
        print(json.dumps({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}))
