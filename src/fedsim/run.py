"""One federated (or centralised) experiment. Thin Hydra entry point around `execute`."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from fedsim.utils.seed import configure_env

configure_env()  # CUBLAS workspace before torch touches CUDA

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fedsim.data.normalize import global_scaler, local_scalers
from fedsim.data.partition import build_federation
from fedsim.data.prepare import load_prepared
from fedsim.evaluation.metrics import compute_metrics
from fedsim.evaluation.profile import latency_ms, peak_memory_mb
from fedsim.fl.simulator import ClientData, FLCfg, MethodCfg, Simulator
from fedsim.models.mlp import MLP, count_params, mlp_flops
from fedsim.utils.io import atomic_torch_save, atomic_write_json, git_commit, hardware_info, stable_hash
from fedsim.utils.seed import set_seed

REPO = Path(__file__).resolve().parents[2]

# v2 method keys and the values that reproduce v1. A key left at its default is dropped from the
# run identity, so every v1 configuration keeps its run_id (guarded by tests/identity_snapshot.py).
V2_METHOD_DEFAULTS = {
    "server_opt": "none", "server_lr": 1.0, "server_beta1": 0.9, "server_beta2": 0.99, "server_eps": 1e-3,
    "head": "linear", "head_scale": 10.0, "decorr": 0.0,
    "ccvr": False, "ccvr_samples": 2000, "ccvr_epochs": 10, "ccvr_lr": 0.1, "ccvr_guard": False,
    "ntd": 0.0, "ntd_tau": 1.0, "fedcm_alpha": 1.0, "anchor": 0.0, "anchor_calib": True,
    "virtual": 0.0, "vmix_k": 8, "vmix_min": 50, "vmix_cap": 5000, "feddyn_alpha": 0.0,
    "vmix_cover_fit": False, "vmix_dp_eps": 0.0, "vmix_dp_delta": 1e-5, "vmix_dp_iters": 5, "vmix_dp_k": 2,
    "vmix_dp_clip": 1.0,
}
# Sub-options that only matter when their switch is on.
V2_DEPENDENT = {
    "server_opt": ("none", ["server_lr", "server_beta1", "server_beta2", "server_eps"]),
    "head": ("linear", ["head_scale"]),
    "ccvr": (False, ["ccvr_samples", "ccvr_epochs", "ccvr_lr", "ccvr_guard"]),
    "ntd": (0.0, ["ntd_tau"]),
    "anchor": (0.0, ["anchor_calib"]),
    "virtual": (0.0, ["vmix_k", "vmix_min", "vmix_cap", "vmix_cover_fit", "vmix_dp_eps"]),
    "vmix_dp_eps": (0.0, ["vmix_dp_delta", "vmix_dp_iters", "vmix_dp_k", "vmix_dp_clip"]),
}


def _strip_v2_defaults(method: dict) -> dict:
    out = dict(method)
    for sw, (off, deps) in V2_DEPENDENT.items():
        if out.get(sw, off) == off:
            for d in deps:
                out.pop(d, None)
    for k, v in V2_METHOD_DEFAULTS.items():
        if k in out and out[k] == v:
            out.pop(k)
    return out


def run_identity(cfg: DictConfig) -> dict:
    """Everything that changes the numbers — and nothing else (not names, paths or logging)."""
    method = OmegaConf.to_container(cfg.method, resolve=True)
    method.pop("name", None)
    method = _strip_v2_defaults(method)
    meta_path = Path(cfg.paths.data_root) / cfg.dataset.name / "meta.json"
    prepared = json.loads(meta_path.read_text()).get("prepared_sha256") if meta_path.exists() else "unprepared"
    ident = {
        "dataset": cfg.dataset.name, "prepared_sha256": prepared, "timing_run": bool(cfg.get("timing_run", False)),
        "labels": cfg.task.labels, "seed": int(cfg.seed),
        "federation": OmegaConf.to_container(cfg.federation, resolve=True),
        "fl": OmegaConf.to_container(cfg.fl, resolve=True),
        "method": method,
    }
    if method.get("pooled"):
        ident["fl"] = dict(ident["fl"], rounds=int(cfg.centralised_rounds), participation=1.0)
    ident["fl"].pop("ckpt_every", None)
    return ident


def vmix_cache_path(cfg, ident, method):
    """Mixtures depend only on the data, partition, normalisation and vmix settings, not on training
    hyperparameters: cache them so runs that differ only in those settings fit them once."""
    if not method.virtual > 0:
        return None
    key = {"prepared_sha256": ident["prepared_sha256"], "labels": ident["labels"], "seed": ident["seed"],
           "federation": ident["federation"], "norm": method.norm, "vmix_k": method.vmix_k,
           "vmix_min": method.vmix_min, "vmix_cap": method.vmix_cap, "em": "torch-em-v1"}
    if method.vmix_cover_fit:
        key["cover_fit"] = True
    if method.vmix_dp_eps > 0:
        key["dp"] = [method.vmix_dp_eps, method.vmix_dp_delta, method.vmix_dp_iters, method.vmix_dp_k, method.vmix_dp_clip]
    return Path(cfg.paths.out_root) / "vmix_cache" / f"{stable_hash(key)}.pt"


def prepare_simulator(cfg):
    """Data, clients and simulator exactly as execute() builds them (used to pre-build the mixture cache)."""
    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.seed), cfg.cpu_threads)
    ident = run_identity(cfg)
    X, y_fine, meta = load_prepared(Path(cfg.paths.data_root) / cfg.dataset.name)
    if cfg.task.labels == "grouped":
        y_task = np.asarray(meta["group_of_class"], dtype=np.int64)[y_fine]
        C = len(meta["groups"])
    else:
        y_task, C = y_fine, len(meta["classes"])
    clients, splits, scalers = build_clients(cfg, X, y_fine, y_task, C, device)
    flc = OmegaConf.to_container(cfg.fl, resolve=True)
    flc["hidden"] = tuple(flc["hidden"])
    method = MethodCfg(**OmegaConf.to_container(cfg.method, resolve=True))
    return Simulator(clients, C, X.shape[1], method, FLCfg(**flc), int(cfg.seed), device, Path("/tmp"), None,
                     vmix_cache=vmix_cache_path(cfg, ident, method))


def build_clients(cfg, X, y_fine, y_task, n_classes, device):
    fed = cfg.federation
    splits = build_federation(y_fine, int(fed.n_clients), float(fed.alpha), int(cfg.seed),
                              int(fed.min_client_size), float(fed.val_frac), float(fed.test_frac))
    m = cfg.method
    if m.pooled:  # centralised reference: identical samples, one client
        splits = [{s: np.sort(np.concatenate([sp[s] for sp in splits])) for s in ("train", "val", "test")}]
    tr = [X[s["train"]] for s in splits]
    if m.norm == "global":
        mu, sd = global_scaler(tr)
        scalers = [(mu, sd)] * len(splits)
    else:
        scalers = local_scalers(tr)
    clients = []
    to = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    for s, (mu, sd) in zip(splits, scalers):
        norm = lambda idx: (X[idx] - mu) / sd
        counts = np.bincount(y_task[s["train"]], minlength=n_classes).astype(np.float32)
        clients.append(ClientData(
            to(norm(s["train"])), to(y_task[s["train"]], torch.long),
            to(norm(s["val"])), to(y_task[s["val"]], torch.long),
            to(norm(s["test"])), to(y_task[s["test"]], torch.long), to(counts)))
    return clients, splits, scalers


@torch.no_grad()
def site_shuffled_eval(cfg, X, y_task, splits, scalers, sim, best, C, train_counts, device):
    """Diagnostic (checkpoint 2b): pool every client's test samples and re-assign each one to a uniformly random
    site j, which normalises it with its own scaler (and BN statistics for FedBN) and classifies it with the final
    model. Traffic then no longer follows the label mix of the site it is scored at. Global-normalisation methods
    are unaffected by construction except through FedBN; local-normalisation methods lose any site-identity cue."""
    rng = np.random.default_rng([int(cfg.seed), 4242])
    K = len(splits)
    test_idx = np.concatenate([s["test"] for s in splits])
    site = rng.integers(0, K, len(test_idx))
    model = sim.model
    ys, ps, cs = [], [], []
    for j in range(K):
        idx = test_idx[site == j]
        if len(idx) == 0:
            continue
        mu, sd = scalers[j]
        x = torch.as_tensor((X[idx] - mu) / sd, dtype=torch.float32, device=device)
        s = dict(best["sd"])
        if sim.m.fedbn and best["bn_local"] and j in best["bn_local"]:
            s.update(best["bn_local"][j])
        model.load_state_dict(s)
        model.eval()
        out = torch.cat([model(x[i:i + 65536]).argmax(1) for i in range(0, len(x), 65536)])
        ys.append(y_task[idx]); ps.append(out.cpu().numpy()); cs.append(np.full(len(idx), j))
    return compute_metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(cs), C, train_counts)


def execute(cfg: DictConfig) -> dict:
    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.seed), cfg.cpu_threads)
    ident = run_identity(cfg)
    run_id = stable_hash(ident)
    run_dir = Path(cfg.paths.out_root) / "runs" / run_id
    res_path = run_dir / "results.json"
    if res_path.exists():
        print(f"[skip] {run_id} already complete")
        return json.loads(res_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    X, y_fine, meta = load_prepared(Path(cfg.paths.data_root) / cfg.dataset.name)
    if cfg.task.labels == "grouped":
        y_task = np.asarray(meta["group_of_class"], dtype=np.int64)[y_fine]
        class_names = meta["groups"]
    else:
        y_task, class_names = y_fine, meta["classes"]
    C = len(class_names)
    clients, splits, scalers = build_clients(cfg, X, y_fine, y_task, C, device)
    train_counts = np.bincount(np.concatenate([y_task[s["train"]] for s in splits]), minlength=C)

    flc = OmegaConf.to_container(cfg.fl, resolve=True)
    flc["hidden"] = tuple(flc["hidden"])
    if cfg.method.pooled:
        flc.update(rounds=int(cfg.centralised_rounds), participation=1.0)
    fl = FLCfg(**flc)
    method = MethodCfg(**OmegaConf.to_container(cfg.method, resolve=True))

    wb = None
    if cfg.wandb.mode != "disabled":
        import wandb
        wb = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, mode=cfg.wandb.mode,
                        group=cfg.suite, name=f"{cfg.dataset.name}-{cfg.task.labels}-{cfg.method.name}-s{cfg.seed}-{run_id}",
                        config={**ident, "run_id": run_id, "method_name": cfg.method.name, "git": git_commit(REPO)},
                        dir=os.environ.get("WANDB_DIR", str(run_dir)), reinit=True)

    def log(rec):
        print(f"[{run_id}] r{rec['round']:>3} val_mF1={rec['val_macro_f1']:.4f} acc={rec['val_acc']:.4f}", flush=True)
        if wb:
            wb.log(rec, step=rec["round"])

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    vmix_cache = vmix_cache_path(cfg, ident, method)
    sim = Simulator(clients, C, X.shape[1], method, fl, int(cfg.seed), device, run_dir, log, vmix_cache=vmix_cache)
    ckpt = run_dir / "ckpt.pt"
    if ckpt.exists():
        sim.load_state(torch.load(ckpt, map_location=device, weights_only=False))
        print(f"[resume] {run_id} from round {sim.round}")
    wall0 = time.time()
    best = sim.run(ckpt)
    extra = {}
    if method.ccvr:
        # Post-hoc server-side re-calibration of the selected model; the uncalibrated test metrics are kept.
        y0, p0, cl0 = sim.predict("test", best["sd"], best["bn_local"])
        extra["test_uncalibrated"] = compute_metrics(y0, p0, cl0, C, train_counts)
        cal_sd = sim.calibrate_head(best["sd"])
        from sklearn.metrics import f1_score
        yv, pv, _ = sim.predict("val", cal_sd, best["bn_local"])
        extra["val_macro_f1_calibrated"] = float(f1_score(yv, pv, average="macro", labels=np.unique(yv), zero_division=0))
        accept = True
        if method.ccvr_guard:
            # Federated validation guard: clients report per-class TP/FP/FN on their validation split for the
            # uncalibrated and the re-fitted head (2 x 3C counts each); the server keeps the better head.
            sim.comm_up_bytes += len(clients) * 2 * 3 * C * 4
            accept = extra["val_macro_f1_calibrated"] > best["val_macro_f1"]
            extra["ccvr_accepted"] = bool(accept)
        extra["val_macro_f1_final"] = extra["val_macro_f1_calibrated"] if accept else float(best["val_macro_f1"])
        if accept:
            best["sd"] = cal_sd
    y, p, cl = sim.predict("test", best["sd"], best["bn_local"])
    metrics = compute_metrics(y, p, cl, C, train_counts)
    if cfg.get("eval_shift", False):
        extra["test_site_shuffled"] = site_shuffled_eval(cfg, X, y_task, splits, scalers, sim, best, C, train_counts, device)

    model = MLP(X.shape[1], C, fl.hidden, fl.dropout, fl.model_norm, head=method.head,
                head_scale=method.head_scale).to(device)
    model.load_state_dict(best["sd"])
    prof = {
        "params": count_params(model),
        "flops_per_sample": mlp_flops(X.shape[1], C, fl.hidden, method.head),
        "latency_ms_b1": latency_ms(model, X.shape[1], device, 1),
        "latency_ms_b1024": latency_ms(model, X.shape[1], device, 1024),
        "train_time_s": sim.train_time,
        "vmix_components": getattr(sim, "vmix_n_components", None),
        "wall_time_s": time.time() - wall0,
        "comm_up_MB": sim.comm_up_bytes / 1e6,
        **peak_memory_mb(device),
    }
    atomic_torch_save({"sd": best["sd"], "bn_local": best["bn_local"], "round": best["round"]}, run_dir / "best_model.pt")
    result = {
        "run_id": run_id, "identity": ident, "method_name": cfg.method.name, "suite": cfg.suite,
        "dataset": cfg.dataset.name, "labels": cfg.task.labels, "seed": int(cfg.seed),
        "class_names": class_names, "train_class_counts": train_counts.tolist(),
        "client_train_sizes": [int(len(s["train"])) for s in splits],
        "best_round": best["round"], "best_val_macro_f1": best["val_macro_f1"],
        "test": metrics, "profile": prof, "curve": sim.curve, **extra,
        "provenance": {"git": git_commit(REPO), "hardware": hardware_info(), "device": device,
                       "prepared_sha256": meta.get("prepared_sha256"), "argv": sys.argv,
                       "cpu_threads": cfg.cpu_threads, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
    }
    atomic_write_json(res_path, result)
    if ckpt.exists():
        ckpt.unlink()  # results + best model are the durable record; round checkpoints are transient
    if wb:
        wb.summary.update({f"test/{k}": v for k, v in metrics.items() if isinstance(v, float)})
        wb.summary.update({f"prof/{k}": v for k, v in prof.items()})
        wb.finish()
    print(f"[done] {run_id} {cfg.method.name} test mF1={metrics['macro_f1']:.4f} acc={metrics['accuracy']:.4f} "
          f"tail={metrics['tail_recall']:.4f} ({prof['wall_time_s']:.0f}s)")
    return result


@hydra.main(config_path=str(REPO / "configs"), config_name="main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    execute(cfg)


if __name__ == "__main__":
    main()
