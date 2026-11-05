"""Independent re-derivation of the SFS-PCL v2 pilot (checkpoint 1).

Reads only stored per-run files (outputs/runs/*/results.json and outputs/index/pilot_v2.json); it does
not import the training code. Uses the Python standard library and numpy.

Go rule (fixed in the agent brief before any run), per candidate:
  (a) fine labels: closes >= 50 % of the SFS-PCL -> SCAFFOLD gap on BOTH datasets,
      closure = (cand - SFS-PCL) / (SCAFFOLD - SFS-PCL), on 2-seed means;
  (b) grouped labels: cand - SFS-PCL >= -0.5 pp on each dataset (2-seed means).
If SCAFFOLD does not lead SFS-PCL on a dataset in this pilot, closure is undefined there and is
reported as such (the rule is then judged on the remaining condition and flagged).

Usage: python analysis/v2_pilot_rederive.py --out outputs_v2 [--report report/v2_pilot.md]
       combination test: --index pilot_v2 pilot_v2_combo --candidates sfspcl_ccvr sfspcl_m sfspcl_ccvr_m --tag v2_combo
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CANDIDATES = ["sfspcl_m", "sfspcl_adam", "sfspcl_etf", "sfspcl_ccvr", "sfspcl_decorr"]
REFS = ["sfspcl", "scaffold"]
LABEL = {"sfspcl_m": "C1a SFS-PCL + server momentum", "sfspcl_adam": "C1b SFS-PCL + FedAdam",
         "sfspcl_etf": "C2 SFS-PCL + simplex-ETF head", "sfspcl_ccvr": "C3 SFS-PCL + CCVR head re-calibration",
         "sfspcl_decorr": "C4 SFS-PCL + FedDecorr", "sfspcl": "SFS-PCL", "scaffold": "SCAFFOLD",
         "sfspcl_ccvr_m": "C3+C1a CCVR + server momentum",
         "sfspcl_m_ntd": "P1 SFS-PCL + momentum + FedNTD", "sfspcl_cm": "P2 SFS-PCL + FedCM",
         "sfspcl_m_fa": "P3 SFS-PCL + momentum + FedFA anchors", "sfspcl_v3x_ccvr": "P4 best(P1-P3) + guarded CCVR",
         "sfspcl_v": "V SFS-PCL + mixture virtual samples", "sfspcl_v_ccvr": "V + guarded CCVR"}


def macro_f1_from_vectors(t: dict) -> float:
    f1 = np.asarray(t["per_class_f1"], float)
    sup = np.asarray(t["per_class_support"], float)
    return float(f1[sup > 0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs_v2")
    ap.add_argument("--report", default=None)
    ap.add_argument("--index", nargs="*", default=["pilot_v2"], help="suite indexes to merge (first entry wins on duplicates)")
    ap.add_argument("--candidates", nargs="*", default=None)
    ap.add_argument("--tag", default="v2_pilot")
    a = ap.parse_args()
    global CANDIDATES
    if a.candidates:
        CANDIDATES = a.candidates
    out = Path(a.out)
    index, seen = [], set()
    for name in a.index:
        for e in json.loads((out / "index" / f"{name}.json").read_text()):
            k = (e["dataset"], e["labels"], e["method"], e["seed"])
            if k not in seen:
                seen.add(k); index.append(e)
    vals = defaultdict(dict)          # (dataset, labels, method) -> {seed: macro_f1}
    extra = defaultdict(dict)         # tail recall, uncalibrated CCVR
    gpus, missing, mismatches = set(), [], 0
    for e in index:
        f = out / "runs" / e["run_id"] / "results.json"
        if not f.exists():
            missing.append(e["run_id"]); continue
        r = json.loads(f.read_text())
        mf = macro_f1_from_vectors(r["test"])
        if abs(mf - r["test"]["macro_f1"]) > 1e-9:
            mismatches += 1
        hw = r["provenance"]["hardware"]
        gpus.add(json.dumps(hw.get("gpu", hw), sort_keys=True) if isinstance(hw, dict) else str(hw))
        key = (e["dataset"], e["labels"], e["method"])
        vals[key][e["seed"]] = mf
        extra[key][e["seed"]] = {"tail_recall": r["test"]["tail_recall"],
                                 "uncal": macro_f1_from_vectors(r["test_uncalibrated"]) if "test_uncalibrated" in r else None}
    datasets = sorted({k[0] for k in vals})
    mean = lambda k: float(np.mean(list(vals[k].values()))) if vals.get(k) else float("nan")

    res = {"n_runs": len(index), "missing": missing, "macro_f1_mismatches": mismatches,
           "gpu_types": sorted(gpus), "table": {}, "go_rule": {}}
    for k, v in vals.items():
        res["table"]["|".join(k)] = {"per_seed": v, "mean": mean(k),
                                     "tail_recall_mean": float(np.mean([x["tail_recall"] for x in extra[k].values()]))}
    for c in CANDIDATES:
        g = {"fine": {}, "grouped": {}}
        for ds in datasets:
            fs, sc, cd = mean((ds, "fine", "sfspcl")), mean((ds, "fine", "scaffold")), mean((ds, "fine", c))
            gap = sc - fs
            g["fine"][ds] = {"cand": cd, "sfspcl": fs, "scaffold": sc, "gap_pp": 100 * gap,
                             "closure": (cd - fs) / gap if gap > 0 else None,
                             "per_seed_cand_minus_sfspcl_pp": {s: 100 * (vals[(ds, "fine", c)][s] - vals[(ds, "fine", "sfspcl")][s])
                                                                for s in vals.get((ds, "fine", c), {}) if s in vals.get((ds, "fine", "sfspcl"), {})}}
            gf, gc = mean((ds, "grouped", "sfspcl")), mean((ds, "grouped", c))
            g["grouped"][ds] = {"cand_minus_sfspcl_pp": 100 * (gc - gf), "cand_minus_scaffold_pp": 100 * (gc - mean((ds, "grouped", "scaffold")))}
        closures = [g["fine"][ds]["closure"] for ds in datasets]
        defined = [x for x in closures if x is not None]
        a_ok = bool(defined) and all(x >= 0.5 for x in defined)
        b_ok = all(g["grouped"][ds]["cand_minus_sfspcl_pp"] >= -0.5 for ds in datasets)
        g["fine_closure_ok"], g["grouped_ok"] = a_ok, b_ok
        g["undefined_closure_on"] = [ds for ds, x in zip(datasets, closures) if x is None]
        g["GO"] = a_ok and b_ok and not g["undefined_closure_on"]
        res["go_rule"][c] = g

    js = out / "evidence" / f"{a.tag}_rederive.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(res, indent=1, default=str))

    L = ["# SFS-PCL v2 pilot: independent re-derivation", "",
         f"Runs indexed: {len(index)}; missing: {len(missing)}; macro-F1 recompute mismatches: {mismatches}; "
         f"GPU types: {len(gpus)}. **Two seeds: directions only.**", "",
         f"## Test macro-F1 (mean of seeds {', '.join(str(s) for s in sorted({e['seed'] for e in index}))}; per seed in brackets)", "",
         "| Method | " + " | ".join(f"{ds} {lab}" for ds in datasets for lab in ("fine", "grouped")) + " |",
         "|---|" + "---|" * (2 * len(datasets))]
    for m in REFS + CANDIDATES:
        cells = []
        for ds in datasets:
            for lab in ("fine", "grouped"):
                k = (ds, lab, m)
                v = vals.get(k, {})
                cells.append(f"{100*mean(k):.2f} [{', '.join(f'{100*v[s]:.2f}' for s in sorted(v))}]" if v else "—")
        L.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    L += ["", "## Go rule (fixed before the run)", "",
          "| Candidate | " + " | ".join(f"closure {ds}" for ds in datasets) + " | " +
          " | ".join(f"grouped Δ vs SFS-PCL {ds} (pp)" for ds in datasets) + " | GO |",
          "|---|" + "---|" * (2 * len(datasets) + 1)]
    for c in CANDIDATES:
        g = res["go_rule"][c]
        cl = [("n/a" if g["fine"][ds]["closure"] is None else f"{100*g['fine'][ds]['closure']:.0f} %") for ds in datasets]
        gr = [f"{g['grouped'][ds]['cand_minus_sfspcl_pp']:+.2f}" for ds in datasets]
        L.append(f"| {LABEL[c]} | " + " | ".join(cl + gr) + f" | {'**GO**' if g['GO'] else 'no'} |")
    text = "\n".join(L) + "\n"
    print(text)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(text)


if __name__ == "__main__":
    main()
