"""Run-identity regression guard for the 738 v1 runs.

`--write` snapshots stable_hash(run_identity(cfg)) for every entry of every v1 suite, with the
prepared-data hash replaced by the fixed string "unprepared" (the data is not needed). The check
mode recomputes the same hashes with the current code and fails on any difference, so a v2 change
that would alter an existing run_id is caught without the data or a GPU.
"""
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_suite import expand  # noqa: E402

V1_SUITES = ["main", "ablation", "sensitivity", "plugin", "timing", "tune_lr", "tune_methods", "diag_scaffold_sfs"]
FIX = REPO / "tests/fixtures/identity_v1.json"


def hashes() -> dict:
    from hydra import compose, initialize_config_dir
    from fedsim.run import run_identity
    from fedsim.utils.io import stable_hash
    out = {}
    with initialize_config_dir(str(REPO / "configs"), version_base="1.3"):
        for s in V1_SUITES:
            suite = yaml.safe_load((REPO / "configs/suites" / f"{s}.yaml").read_text())
            for r in expand(suite, ["paths.data_root=/nonexistent"]):
                cfg = compose("main", overrides=r["overrides"])
                key = s + "|" + json.dumps(r["label"], sort_keys=True)
                out[key] = stable_hash(run_identity(cfg))
    return out


if __name__ == "__main__" and "--real" not in sys.argv:
    h = hashes()
    if "--write" in sys.argv:
        FIX.write_text(json.dumps(h, indent=0, sort_keys=True))
        print(f"wrote {len(h)} identities")
    else:
        ref = json.loads(FIX.read_text())
        bad = [k for k in ref if h.get(k) != ref[k]]
        assert not bad and len(h) == len(ref), f"{len(bad)} v1 run identities changed, e.g. {bad[:3]}"
        print(f"v1 run identities unchanged: {len(ref)} entries, {len(set(ref.values()))} unique")


# ---- real run_ids: the prepared-data hashes from data/prepared/*/meta.json (Drive copies, 2026-09-20) ----
PREPARED_SHA256 = {
    "ciciot2023": "dcad9db348d5e307681b4528598c25048fa46412356b05166e74117af53e2927",
    "edgeiiot": "9be2aed2356fdd1324abdd2ca6d77460d2ca60358ef8bec38ffa54a4a188f24e",
}
# Two timing-suite SCAFFOLD entries do not reproduce from the committed configs with the ORIGINAL v1 code
# either (checked at c6fc7c0); they predate the v2 branch and are listed here so the check stays strict.
KNOWN_V1_EXCEPTIONS = {"bd5ab0dae06c", "9e4df34c889a"}


def check_real_index_ids():
    import tempfile
    from hydra import compose, initialize_config_dir
    from fedsim.run import run_identity
    from fedsim.utils.io import stable_hash
    root = Path(tempfile.mkdtemp())
    for ds, sha in PREPARED_SHA256.items():
        (root / ds).mkdir()
        (root / ds / "meta.json").write_text(json.dumps({"prepared_sha256": sha}))
    ok, bad = 0, []
    with initialize_config_dir(str(REPO / "configs"), version_base="1.3"):
        for f in sorted((REPO / "results/index").glob("*.json")):
            if "selection" in f.name:
                continue
            idx = json.loads(f.read_text())
            suite = yaml.safe_load((REPO / "configs/suites" / f"{f.stem}.yaml").read_text())
            for r, e in zip(expand(suite, [f"paths.data_root={root}"]), idx):
                h = stable_hash(run_identity(compose("main", overrides=r["overrides"])))
                if h == e["run_id"]:
                    ok += 1
                elif e["run_id"] not in KNOWN_V1_EXCEPTIONS:
                    bad.append(e)
    assert not bad, f"{len(bad)} real v1 run_ids changed: {bad[:3]}"
    print(f"real v1 run_ids reproduced: {ok} index entries (+{len(KNOWN_V1_EXCEPTIONS)} known pre-v2 exceptions)")


if __name__ == "__main__" and "--real" in sys.argv:
    check_real_index_ids()
