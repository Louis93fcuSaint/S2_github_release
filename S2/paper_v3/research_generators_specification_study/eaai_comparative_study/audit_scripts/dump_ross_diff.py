import json, difflib, os, sys
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pilot_runs", "00_env")
a = json.load(open(os.path.join(base, "ross_env_161.json"), encoding="utf-8"))
b = json.load(open(os.path.join(base, "ross_env_230.json"), encoding="utf-8"))
for name in a["functions"]:
    sa = a["functions"][name].get("source", "")
    sb = b["functions"][name].get("source", "")
    print("\n###", name)
    print("".join(difflib.unified_diff(sa.splitlines(True), sb.splitlines(True), fromfile="1.6.1", tofile="2.3.0", n=3)))
