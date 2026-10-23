import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
EA = HERE.parents[1] / "research_generators_specification_study" / "eaai_comparative_study"
sys.path.insert(0, str(EA / "audit_scripts"))
import eaai_common as A
b = A.load_cvae()
sc = b["scalers"]
for k in ("layout", "condition_dim", "mask_policy", "h_max", "point_prob", "orders"):
    print(k, "=", sc.get(k))
print("mean shape", getattr(sc.get("mean"), "shape", None), "std shape", getattr(sc.get("std"), "shape", None))
d = A.load_ddpm()
print("ddpm payload keys", sorted(d["payload"].keys()))
print("ddpm cond layout", d["scalers"].get("layout"), d["scalers"].get("condition_dim"))
