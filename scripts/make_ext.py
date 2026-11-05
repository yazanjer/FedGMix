"""Checkpoint-2b step: compare tune_ext selections with the grid's tuned values (jobs/tuned_after_grid.yaml) and write
configs/suites/grid_ext.yaml with the baselines whose selection changed (FedLC also re-runs FedLC+SFS)."""
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs_v2")
before = yaml.safe_load(Path(sys.argv[2]).read_text())["methods"]
sel = json.loads((out / "index" / "tune_ext_selection.json").read_text())["selected"]
changed = []
for m, s in sel.items():
    new = {k: v for k, v in s.items() if "." in k}
    old = {k: before.get(m, {}).get(k) for k in new}
    print(m, "old", old, "new", new, "val", s["val_macro_f1_mean"], "edge", s.get("on_grid_edge"))
    if new != old:
        changed.append(m)
methods = changed + (["fedlc_sfs"] if "fedlc" in changed else [])
(REPO / "configs/suites/grid_ext.yaml").write_text(yaml.safe_dump({"suite": "grid_ext", "groups": [
    {"datasets": ["ciciot2023", "edgeiiot", "toniot"], "labels": ["fine", "grouped"], "methods": methods,
     "seeds": [10, 11, 12, 13, 14]}]}, sort_keys=False) if methods else "suite: grid_ext\ngroups: []\n")
print("re-run:", methods)
