"""CPU checks for the pilot-2 components (FedNTD, FedCM, FedFA anchors) on synthetic data.

Run: PYTHONPATH=src python tests/test_v3.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import flat  # noqa: E402
from test_v2 import SFSPCL, TMP, run, sim_for  # noqa: E402
from fedsim.fl.simulator import ntd_loss  # noqa: E402


def test_defaults_unchanged():
    a = run(SFSPCL)
    b = run(replace(SFSPCL, ntd=0.0, fedcm_alpha=1.0, anchor=0.0, ccvr_guard=False))
    assert torch.equal(flat(a.global_sd), flat(b.global_sd))
    print("v3 defaults reproduce SFS-PCL bit-exactly: OK")


def test_ntd_loss():
    g = torch.Generator().manual_seed(0)
    s = torch.randn(64, 5, generator=g); t = torch.randn(64, 5, generator=g); y = torch.randint(0, 5, (64,), generator=g)
    assert ntd_loss(s, s.clone(), y, 1.0).abs() < 1e-6
    assert ntd_loss(s, t, y, 1.0) > 0
    s2 = s.clone(); s2[torch.arange(64), y] += 7.0          # the true-class logit is ignored
    assert torch.allclose(ntd_loss(s, t, y, 1.0), ntd_loss(s2, t, y, 1.0), atol=1e-6)
    s3 = s + 3.0                                             # shift invariance of the softmax
    assert torch.allclose(ntd_loss(s, t, y, 2.0), ntd_loss(s3, t, y, 2.0), atol=1e-6)
    print("NTD loss: zero at equality, ignores true class, shift-invariant: OK")


def test_components_change_training():
    a = run(SFSPCL)
    for m in [replace(SFSPCL, ntd=1.0), replace(SFSPCL, fedcm_alpha=0.1), replace(SFSPCL, anchor=0.1)]:
        b = run(m)
        assert not torch.allclose(flat(a.global_sd), flat(b.global_sd), atol=1e-5), m
    print("NTD / FedCM / anchors each change training: OK")


def test_fedcm_first_round_is_fedavg():
    # Delta_0 = 0 and no delta is applied before it exists, so round 1 equals SFS-PCL exactly.
    a = sim_for(SFSPCL, rounds=1); a.run(None)
    b = sim_for(replace(SFSPCL, fedcm_alpha=0.1), rounds=1); b.run(None)
    assert torch.equal(flat(a.global_sd), flat(b.global_sd))
    assert set(b.server_state["fedcm_delta"]) == set(b.param_keys)
    print("FedCM: round 1 == FedAvg, Delta covers all trainable parameters: OK")


def test_anchor_state_and_upload():
    s = sim_for(replace(SFSPCL, anchor=0.1), rounds=2)
    base = sim_for(SFSPCL, rounds=2)
    A0 = s._anchors().clone()
    assert torch.allclose(A0 @ A0.T, torch.eye(5)), "initial anchors orthonormal"
    s.run(None); base.run(None)
    A = s.server_state["anchors"]
    assert A.shape == (5, 16) and not torch.allclose(A, A0)
    m_sel = max(1, int(round(s.fl.participation * s.K)))
    assert abs((s.comm_up_bytes - base.comm_up_bytes) - 2 * m_sel * 5 * (16 + 1) * 4) < 1e-6
    print("anchors: orthonormal init, updated by the server, upload accounted: OK")


def test_determinism_and_resume():
    tmp = TMP / "ck3.pt"
    for m in [replace(SFSPCL, ntd=1.0, server_opt="momentum", server_beta1=0.7),
              replace(SFSPCL, fedcm_alpha=0.1),
              replace(SFSPCL, anchor=0.1, server_opt="momentum", server_beta1=0.7)]:
        a = run(m); b = run(m)
        assert torch.equal(flat(a.global_sd), flat(b.global_sd)), f"{m} not deterministic"
        run(m, stop_at=4, ckpt=tmp)
        c = run(m, resume_from=tmp)
        assert torch.equal(flat(a.global_sd), flat(c.global_sd)), f"{m} resume differs"
    print("v3 determinism + resume (FedCM delta and anchors checkpointed): OK")


if __name__ == "__main__":
    test_defaults_unchanged()
    test_ntd_loss()
    test_components_change_training()
    test_fedcm_first_round_is_fedavg()
    test_anchor_state_and_upload()
    test_determinism_and_resume()
