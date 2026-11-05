"""Pre-build the SFS-PCL-V / FedMiX mixture cache for every (dataset, labels, seed, partition) of a suite, in parallel
processes, so that the runs of the suite do not fit identical mixtures concurrently. Sweep keys under `federation.`
change the partition and are included. Usage:
python scripts/warm_vmix.py --suite sweep_main --out outputs_v2 --data data/prepared [--workers 4]
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True)
ap.add_argument("--out", default="outputs_v2")
ap.add_argument("--data", default="data/prepared")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
a = ap.parse_args()
if a.one:
    sys.path.insert(0, str(REPO / "src"))
    from hydra import compose, initialize_config_dir
    from fedsim.run import prepare_simulator
    ds, lab, seed, meth, extra = json.loads(a.one)
    with initialize_config_dir(str(REPO / "configs"), version_base="1.3"):
        cfg = compose("main", overrides=[f"dataset={ds}", f"task.labels={lab}", f"seed={seed}", f"method={meth}",
                                         f"paths.out_root={a.out}", f"paths.data_root={a.data}", *extra])
    sim = prepare_simulator(cfg)
    print(ds, lab, seed, extra, "components", sim.vmix_n_components, flush=True)
    sys.exit(0)


def vkey(m):
    c = yaml.safe_load((REPO / "configs/method" / f"{m}.yaml").read_text())
    return (c.get("norm"), c.get("vmix_k", 8), c.get("vmix_min", 50), c.get("vmix_cap", 5000)) if c.get("virtual", 0) > 0 else None


suite = yaml.safe_load((REPO / "configs/suites" / f"{a.suite}.yaml").read_text())
jobs = set()
for g in suite["groups"]:
    reps = {}
    for m in g["methods"]:
        k = vkey(m)
        if k is not None and k not in reps:
            reps[k] = m
    sweep = {k: v for k, v in (g.get("sweep") or {}).items() if k.startswith("federation.")}
    keys = list(sweep)
    combos = list(itertools.product(*[sweep[k] for k in keys])) if keys else [()]
    for m in reps.values():
        for ds in g["datasets"]:
            for lab in g["labels"]:
                for s in g["seeds"]:
                    for vals in combos:
                        jobs.add((ds, lab, s, m, tuple(f"{k}={v}" for k, v in zip(keys, vals))))
jobs = sorted(jobs)
print(len(jobs), "mixture caches", flush=True)
procs, rc = [], 0
for ds, lab, s, m, extra in jobs:
    procs.append(subprocess.Popen([sys.executable, __file__, "--out", a.out, "--data", a.data, "--suite", a.suite,
                                   "--one", json.dumps([ds, lab, s, m, list(extra)])], cwd=REPO))
    if len(procs) >= a.workers:
        rc |= procs.pop(0).wait()
for p in procs:
    rc |= p.wait()
sys.exit(rc)
