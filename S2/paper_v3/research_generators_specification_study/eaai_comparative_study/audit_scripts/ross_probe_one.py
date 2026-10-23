# -*- coding: utf-8 -*-
"""Diagnostic ROSS worker: forward speeds plus the full raw modal table.

This worker deliberately imports only spec_common, so it can run under both
the ROSS 1.6.1 (Python 3.8) and ROSS 2.3.0 (Python 3.12) environments.
"""
import json
import os
import sys

import numpy as np

S2 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
SPEC_DESIGN = os.path.join(S2, "spec_design")
if SPEC_DESIGN not in sys.path:
    sys.path.insert(0, SPEC_DESIGN)
import spec_common as S
S.os.environ['ROSS_FAST_RELABEL'] = '1'


def main():
    payload = json.loads(sys.argv[1])
    module = S.ross_module()
    x = np.asarray(payload["x"], dtype=np.float64)
    rotor = module.build_ross_rotor(
        S.sample_from_canonical(x, payload["material"], int(payload["nd"]), int(payload["nb"])),
        n_base=int(payload.get("n_base", 15)))
    forward, raw = module.critical_speed_result(rotor, num_modes=int(payload.get("num_modes", 48)))
    print(json.dumps({"ok": True, "forward": [float(v) for v in forward], "raw": raw},
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}))

