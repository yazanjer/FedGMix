"""CPU checks for FedDyn (full-grid baseline). Run: PYTHONPATH=src python tests/test_v5.py"""
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import flat  # noqa: E402
from test_v2 import TMP, run, sim_for  # noqa: E402
from fedsim.fl.simulator import MethodCfg  # noqa: E402

FEDAVG = MethodCfg("fedavg", norm="local", loss="ce", client_momentum=0.0)
DYN = replace(FEDAVG, feddyn_alpha=0.01)


def test_defaults_unchanged():
    a = run(FEDAVG); b = run(replace(FEDAVG, feddyn_alpha=0.0))
    assert torch.equal(flat(a.global_sd), flat(b.global_sd))
    print("feddyn_alpha=0 reproduces FedAvg bit-exactly: OK")


def test_first_round_and_state():
    s = sim_for(DYN, rounds=1); s.run(None)
    m_sel = max(1, int(round(s.fl.participation * s.K)))
    assert len(s.dyn_local) == m_sel and s.dyn_h is not None
    assert any(float(v.abs().sum()) > 0 for v in s.dyn_h.values())
    assert s.comm_up_bytes == sim_for(FEDAVG, rounds=1).comm_up_bytes + s._round_up_bytes(m_sel), "upload must be 1x"
    print("FedDyn: per-client state and server h set after round 1; upload 1x: OK")


def test_changes_and_resume():
    a = run(FEDAVG); b = run(DYN)
    assert not torch.allclose(flat(a.global_sd), flat(b.global_sd), atol=1e-6)
    c = run(DYN)
    assert torch.equal(flat(b.global_sd), flat(c.global_sd)), "not deterministic"
    tmp = TMP / "ck5.pt"
    run(DYN, stop_at=4, ckpt=tmp)
    r = run(DYN, resume_from=tmp)
    assert torch.equal(flat(b.global_sd), flat(r.global_sd)), "resume differs"
    print("FedDyn: changes training, deterministic, resumes exactly (client + server state checkpointed): OK")


if __name__ == "__main__":
    test_defaults_unchanged()
    test_first_round_and_state()
    test_changes_and_resume()
