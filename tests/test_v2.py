"""CPU checks for the v2 candidate mechanisms (synthetic data).

Run: PYTHONPATH=src python tests/test_v2.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import flat, make_clients  # noqa: E402
from fedsim.fl.simulator import FLCfg, MethodCfg, Simulator, decorr_loss  # noqa: E402
from fedsim.models.mlp import MLP, simplex_etf  # noqa: E402
from fedsim.utils.seed import set_seed  # noqa: E402

TMP = Path("/tmp/fedgmix-tests/v2tests"); TMP.mkdir(parents=True, exist_ok=True)
SFSPCL = MethodCfg("sfspcl", norm="global", loss="la", tau=1.0)


def sim_for(method, rounds=6):
    set_seed(0, 1)
    clients, *_ = make_clients()
    fl = FLCfg(rounds=rounds, eval_every=2, ckpt_every=2, hidden=(32, 16))
    return Simulator(clients, 5, 8, method, fl, 0, "cpu", TMP)


def run(method, rounds=6, ckpt=None, resume_from=None, stop_at=None):
    sim = sim_for(method, rounds)
    if resume_from:
        sim.load_state(torch.load(resume_from, weights_only=False))
    if stop_at:
        sim.fl = replace(sim.fl, rounds=stop_at)
    sim.run(ckpt)
    return sim


def test_etf_geometry():
    for C, d in [(5, 16), (15, 64), (34, 64), (8, 7)]:
        W = simplex_etf(C, d).double()
        G = W @ W.T
        assert torch.allclose(G.diagonal(), torch.ones(C, dtype=G.dtype), atol=1e-6)
        off = G[~torch.eye(C, dtype=bool)]
        assert torch.allclose(off, torch.full_like(off, -1 / (C - 1)), atol=1e-6), (C, d)
    print("simplex ETF geometry: OK")


def test_v1_defaults_unchanged():
    a = run(SFSPCL)
    b = run(replace(SFSPCL, server_opt="none", head="linear", decorr=0.0, ccvr=False))
    assert torch.equal(flat(a.global_sd), flat(b.global_sd))
    print("v2 defaults reproduce v1 SFS-PCL bit-exactly: OK")


def test_server_momentum_limits():
    a = run(SFSPCL)
    b = run(replace(SFSPCL, server_opt="momentum", server_lr=1.0, server_beta1=0.0))
    assert torch.allclose(flat(a.global_sd), flat(b.global_sd), atol=1e-5), "beta=0, lr=1 must equal FedAvg"
    c = run(replace(SFSPCL, server_opt="momentum", server_lr=1.0, server_beta1=0.9))
    assert not torch.allclose(flat(a.global_sd), flat(c.global_sd), atol=1e-4)
    print("server momentum: beta=0 == FedAvg, beta=0.9 differs: OK")


def test_adam_one_step():
    m = replace(SFSPCL, server_opt="adam", server_lr=0.05, server_beta1=0.9, server_beta2=0.99, server_eps=1e-3)
    s = sim_for(m, rounds=1)
    x0 = {k: v.clone() for k, v in s.global_sd.items()}
    ref = sim_for(SFSPCL, rounds=1)
    ref.run(None)
    avg = ref.global_sd
    s.run(None)
    for k in s.param_keys:
        g = x0[k] - avg[k]
        mt = 0.1 * g
        vt = 0.99 * 1e-6 + 0.01 * g * g
        want = x0[k] - 0.05 * mt / (vt.sqrt() + 1e-3)
        assert torch.allclose(s.global_sd[k], want, atol=1e-5), k
    for k in s.bn_buffer_keys:
        assert torch.allclose(s.global_sd[k].float(), avg[k].float(), atol=1e-6), k
    print("FedAdam single step matches the closed form; BN buffers averaged: OK")


def test_etf_frozen():
    m = replace(SFSPCL, head="etf", head_scale=10.0)
    s = run(m)
    W = simplex_etf(5, 16)
    assert torch.equal(s.global_sd["head.weight"], W), "ETF head must stay fixed"
    assert "head.weight" not in s.param_keys and "proj.weight" in s.param_keys
    lin = sim_for(SFSPCL)
    assert s.n_params == lin.n_params - 5 * 16 - 5 + 16 * 16 + 16, "upload counts trainable parameters only"
    f = run(replace(SFSPCL, head="frozen"))
    f0 = sim_for(replace(SFSPCL, head="frozen"))
    assert torch.equal(f.global_sd["head.weight"], f0.global_sd["head.weight"])
    print("ETF / frozen heads stay fixed and are not uploaded: OK")


def test_determinism_and_resume():
    tmp = TMP / "ck.pt"
    for m in [replace(SFSPCL, server_opt="momentum"), replace(SFSPCL, server_opt="yogi", server_lr=0.01),
              replace(SFSPCL, head="etf"), replace(SFSPCL, decorr=0.1)]:
        a = run(m)
        b = run(m)
        assert torch.equal(flat(a.global_sd), flat(b.global_sd)), f"{m} not deterministic"
        run(m, stop_at=4, ckpt=tmp)
        c = run(m, resume_from=tmp)
        assert torch.equal(flat(a.global_sd), flat(c.global_sd)), f"{m} resume differs"
    print("v2 determinism + resume (server state checkpointed): OK")


def test_decorr_loss():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4096, 8, generator=g)
    low = decorr_loss(z).item()
    z2 = torch.cat([z[:, :4], z[:, :4]], 1)
    high = decorr_loss(z2).item()
    assert abs(low - 1 / 8) < 0.01 and abs(high - 0.25) < 0.01, (low, high)  # duplicates add 8 unit off-diagonals
    print("decorrelation loss: independent ~1/d, duplicated columns higher: OK")


def test_ccvr():
    m = replace(SFSPCL, ccvr=True, ccvr_samples=500, ccvr_epochs=3, ccvr_lr=0.05)
    s = run(m)
    n, sm, ss = s.class_feature_moments(s.best["sd"])
    s.model.load_state_dict(s.best["sd"]); s.model.eval()
    with torch.no_grad():
        F_ = torch.cat([s.model.features(c.xtr) for c in s.c]).double()
        Y = torch.cat([c.ytr for c in s.c])
    for c in range(5):
        fc = F_[Y == c]
        assert int(n[c]) == len(fc)
        assert torch.allclose(sm[c], fc.sum(0), atol=1e-6)
        assert torch.allclose(ss[c], fc.T @ fc, atol=1e-5)
    before = s.comm_up_bytes
    cal1 = s.calibrate_head(s.best["sd"])
    assert s.comm_up_bytes - before == 6 * 5 * (1 + 16 + 16 * 17 // 2) * 4
    cal2 = run(m).calibrate_head(s.best["sd"])
    changed = {k for k in cal1 if not torch.equal(cal1[k], s.best["sd"][k])}
    assert changed and changed <= {"head.weight", "head.bias"}, changed
    assert all(torch.equal(cal1[k], cal2[k]) for k in cal1), "CCVR not deterministic"
    print("CCVR: pooled class moments exact, only the head changes, deterministic: OK")


if __name__ == "__main__":
    test_etf_geometry()
    test_v1_defaults_unchanged()
    test_server_momentum_limits()
    test_adam_one_step()
    test_etf_frozen()
    test_determinism_and_resume()
    test_decorr_loss()
    test_ccvr()
