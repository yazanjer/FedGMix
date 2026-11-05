"""CPU checks for SFS-PCL-V (pilot 3). Run: PYTHONPATH=src python tests/test_v4.py"""
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import flat  # noqa: E402
from test_v2 import SFSPCL, TMP, run, sim_for  # noqa: E402
from fedsim.fl.simulator import _gen  # noqa: E402

V = replace(SFSPCL, virtual=0.5, vmix_k=3, vmix_min=20, vmix_cap=500)


def test_defaults_unchanged():
    a = run(SFSPCL)
    b = run(replace(SFSPCL, virtual=0.0))
    assert torch.equal(flat(a.global_sd), flat(b.global_sd))
    print("virtual=0 reproduces SFS-PCL bit-exactly: OK")


def test_mixture_guard_and_accounting():
    s = sim_for(V)
    base = sim_for(SFSPCL)
    assert s.vmix and s.vmix_n_components > 0
    for c, (w, mus, Ls) in s.vmix.items():
        assert abs(float(w.sum()) - 1) < 1e-5 and (w > 0).all()
        assert torch.allclose(Ls, torch.tril(Ls))
    d = s.c[0].xtr.shape[1]
    assert s.comm_up_bytes - base.comm_up_bytes == s.vmix_n_components * (1 + d + d * (d + 1) // 2) * 4
    print(f"mixtures: {s.vmix_n_components} guarded components, weights normalised, upload accounted: OK")


def test_virtual_sampling():
    s = sim_for(V)
    x1, y1 = s._virtual_batch(4000, _gen(0, 1, 2, 77))
    x2, y2 = s._virtual_batch(4000, _gen(0, 1, 2, 77))
    assert torch.equal(x1, x2) and torch.equal(y1, y2), "virtual samples not deterministic"
    counts = torch.bincount(y1, minlength=s.C).float()
    present = counts[s.vmix_classes]
    assert (present > 0).all() and present.max() / present.min() < 1.5, "classes not balanced"
    for c, (w, mus, Ls) in s.vmix.items():
        mu = (w[:, None] * mus).sum(0)
        xc = x1[y1 == c]
        se = xc.std(0) / len(xc) ** 0.5
        assert ((xc.mean(0) - mu).abs() <= 6 * se + 1e-4).all(), c
    print("virtual samples: deterministic, class-balanced, centred on the mixtures: OK")


def test_changes_training_and_resume():
    a = run(SFSPCL)
    b = run(V)
    assert not torch.allclose(flat(a.global_sd), flat(b.global_sd), atol=1e-5)
    c = run(V)
    assert torch.equal(flat(b.global_sd), flat(c.global_sd)), "not deterministic"
    tmp = TMP / "ck4.pt"
    run(V, stop_at=4, ckpt=tmp)
    r = run(V, resume_from=tmp)
    assert torch.equal(flat(b.global_sd), flat(r.global_sd)), "resume differs"
    print("SFS-PCL-V: changes training, deterministic, resumes exactly: OK")


if __name__ == "__main__":
    test_defaults_unchanged()
    test_mixture_guard_and_accounting()
    test_virtual_sampling()
    test_changes_training_and_resume()
