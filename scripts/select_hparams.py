"""Pick hyperparameters by mean validation macro-F1 on tuning seed 100 and write configs/tuned.yaml."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True, choices=["tune_lr", "tune_methods", "tune_v2", "tune_v3", "tune_v4", "tune_grid", "tune_ext"])
ap.add_argument("--out", default="outputs")
a = ap.parse_args()
out = Path(a.out)
index = json.loads((out / "index" / f"{a.suite}.json").read_text())
score = defaultdict(list)
for e in index:
    r = out / "runs" / e["run_id"] / "results.json"
    if not r.exists():
        raise SystemExit(f"missing result {e['run_id']}; finish the suite first")
    _r = json.loads(r.read_text())
    # CCVR re-fits the head after training: select on the calibrated validation score when present.
    v = _r.get("val_macro_f1_final", _r.get("val_macro_f1_calibrated", _r["best_val_macro_f1"]))
    keys = tuple(sorted((k, e[k]) for k in e if "." in k))
    score[(e["method"], keys)].append(v)
tuned = yaml.safe_load((REPO / "configs/tuned.yaml").read_text()) or {}
tuned.setdefault("all", {}); tuned.setdefault("methods", {})
best = {}
for (m, keys), vs in score.items():
    mean = sum(vs) / len(vs)
    if m not in best or mean > best[m][0]:
        best[m] = (mean, keys)
report = {}
for m, (mean, keys) in best.items():
    report[m] = {"val_macro_f1_mean": round(mean, 4), **dict(keys)}
    # Grid-extension rule: flag any selected value that sits on an edge of the values tried.
    edges = []
    for k, v in keys:
        tried = sorted({dict(kk)[k] for (mm, kk) in score if mm == m})
        if len(tried) > 1 and v in (tried[0], tried[-1]):
            edges.append(k)
    if edges:
        report[m]["on_grid_edge"] = edges
    if a.suite == "tune_lr":
        tuned["all"].update(dict(keys))
    else:
        tuned["methods"][m] = dict(keys)
(REPO / "configs/tuned.yaml").write_text("# Written by scripts/select_hparams.py (validation macro-F1, tuning seed 100)\n" + yaml.safe_dump(tuned, sort_keys=True))
(out / "index" / f"{a.suite}_selection.json").write_text(json.dumps({"all_scores": {f"{m}|{dict(k)}": sum(v)/len(v) for (m, k), v in score.items()}, "selected": report}, indent=1))
print(json.dumps(report, indent=1))
