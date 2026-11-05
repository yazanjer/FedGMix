"""CPU checks for checkpoint 4: DP mixture release and the corrected guard. Run: PYTHONPATH=src python tests/test_v6.py"""
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import flat  # noqa: E402
from test_v2 import SFSPCL, run, sim_for  # noqa: E402
from fedsim.fl.simulator import dp_gmm_em, dp_rho  # noqa: E402
from fedsim.run import _strip_v2_defaults  # noqa: E402

V = replace(SFSPCL, virtual=0.5, vmix_k=3, vmix_min=20, vmix_cap=500)


def test_identity_unchanged():
    base = {"name": "fedmix", "virtual": 1.0, "vmix_k": 8, "vmix_min": 50, "vmix_cap": 5000, "ccvr": True}
    ext = dict(base, vmix_cover_fit=False, vmix_dp_eps=0.0, vmix_dp_delta=1e-5, vmix_dp_iters=5, vmix_dp_k=2, vmix_dp_clip=1.0)
    assert _strip_v2_defaults(ext) == _strip_v2_defaults(base)
    a = run(V); b = run(replace(V, vmix_cover_fit=False, vmix_dp_eps=0.0))
    assert torch.equal(flat(a.global_sd), flat(b.global_sd))
    print("new defaults keep run identities and results bit-exact: OK")


def test_accounting():
    for eps in (1.0, 4.0, 16.0, 64.0):
        r = dp_rho(eps, 1e-5)
        assert abs(r + 2 * math.sqrt(r * math.log(1e5)) - eps) < 1e-9
    print("zCDP -> (eps, delta) conversion exact: OK")


def test_dp_em():
    g0 = torch.Generator().manual_seed(0)
    X = torch.cat([torch.randn(2000, 6, generator=g0, dtype=torch.float64) * 0.3 + 1,
                   torch.randn(2000, 6, generator=g0, dtype=torch.float64) * 0.3 - 1])
    N, mu, cov = dp_gmm_em(X, 2, torch.Generator().manual_seed(1), 1e6)
    assert torch.isfinite(cov).all() and (torch.linalg.eigvalsh(cov) > 0).all()
    assert all(1900 <= float(x) <= 2100 for x in N), N
    N2, mu2, cov2 = dp_gmm_em(X, 2, torch.Generator().manual_seed(1), 1.0)
    assert torch.isfinite(cov2).all() and (torch.linalg.eigvalsh(cov2) > 0).all()
    N3, mu3, cov3 = dp_gmm_em(X, 2, torch.Generator().manual_seed(1), 1.0)
    assert torch.equal(mu2, mu3), "not deterministic"
    print("DP-EM: PSD covariances, recovers clusters at huge eps, deterministic: OK")


def test_sim_dp_and_guard():
    a = sim_for(replace(V, vmix_dp_eps=4.0, vmix_dp_k=2))
    assert a.vmix_n_components >= 0
    b = run(replace(V, vmix_dp_eps=4.0, vmix_dp_k=2)); c = run(replace(V, vmix_dp_eps=4.0, vmix_dp_k=2))
    assert torch.equal(flat(b.global_sd), flat(c.global_sd)), "DP run not deterministic"
    f = sim_for(replace(V, vmix_cover_fit=True, vmix_cap=60))
    o = sim_for(replace(V, vmix_cap=60))
    assert f.vmix_n_components <= o.vmix_n_components
    print(f"simulator: DP release runs deterministically ({a.vmix_n_components} components); fit-sample guard is stricter ({f.vmix_n_components} <= {o.vmix_n_components}): OK")


if __name__ == "__main__":
    test_identity_unchanged()
    test_accounting()
    test_dp_em()
    test_sim_dp_and_guard()
