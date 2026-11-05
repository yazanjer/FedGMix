"""Single-thread CPU cost of one local epoch (FedAvg, SCAFFOLD, FedGMix with r = 1) and of the one-time mixture fit,
at the shapes of the main setting (K = 20, average client training size, MLP 256-128-64, batch 256).
Timing depends only on shapes, so inputs are synthetic. Usage: python time_cpu.py > timing.json"""
import json
import math
import platform
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
torch.set_num_threads(1)
torch.manual_seed(0)

DATA = {  # name: (d, fine classes, total flows)
    "CICIoT2023": (46, 34, 1005585), "Edge-IIoTset": (95, 15, 624105), "TON_IoT": (150, 10, 119874)}
B, K, TRAIN = 256, 20, 0.7


def mlp(d, C):
    layers, h = [], d
    for u in (256, 128, 64):
        layers += [nn.Linear(h, u), nn.BatchNorm1d(u), nn.ReLU(), nn.Dropout(0.1)]
        h = u
    return nn.Sequential(*layers, nn.Linear(h, C))


def gmm_em(X, k, iters=100, reg=1e-3):
    """Same computation as the simulator's EM (k-means++ seeding, full covariances, fixed iteration count)."""
    n, d = X.shape
    idx = [int(torch.randint(n, (1,)))]
    for _ in range(1, k):
        d2 = torch.cdist(X, X[idx]).min(1).values ** 2
        idx.append(int(torch.multinomial(d2 / d2.sum(), 1)))
    R = torch.nn.functional.one_hot(torch.cdist(X, X[idx]).argmin(1), k).to(X.dtype)
    eye = torch.eye(d, dtype=X.dtype)
    for _ in range(iters):
        Nk = R.sum(0) + 1e-10
        mu = (R.T @ X) / Nk[:, None]
        Xc = X[None] - mu[:, None]
        cov = torch.einsum("kn,kni,knj->kij", R.T, Xc, Xc) / Nk[:, None, None] + reg * eye
        L = torch.linalg.cholesky(cov)
        sol = torch.linalg.solve_triangular(L, Xc.transpose(1, 2), upper=False)
        logp = -0.5 * ((sol ** 2).sum(1) + 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)[:, None]) \
            + torch.log(Nk / n)[:, None]
        R = torch.softmax(logp, 0).T


def epoch(d, C, n, mode, n_comp=200):
    model = mlp(d, C)
    lr, mom = (0.02, 0.0) if mode == "scaffold" else (0.05, 0.9)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=1e-4)
    X = torch.randn(n, d); y = torch.randint(C, (n,))
    if mode == "scaffold":
        corr = [torch.randn_like(p) * 1e-3 for p in model.parameters()]
    if mode == "fedgmix":
        mus = torch.randn(n_comp, d); Ls = torch.linalg.cholesky(torch.eye(d).expand(n_comp, d, d).clone())
        comp_cls = torch.randint(C, (n_comp,))
    t0 = time.perf_counter()
    for i in range(0, n, B):
        xb, yb = X[i:i + B], y[i:i + B]
        if mode == "fedgmix":
            j = torch.randint(n_comp, (len(xb),))
            xv = mus[j] + torch.bmm(Ls[j], torch.randn(len(xb), d, 1)).squeeze(2)
            xb, yb = torch.cat([xb, xv]), torch.cat([yb, comp_cls[j]])
        opt.zero_grad()
        nn.functional.cross_entropy(model(xb), yb).backward()
        if mode == "scaffold":
            for p, c in zip(model.parameters(), corr):
                p.grad.add_(c)
        opt.step()
    return time.perf_counter() - t0


out = {"cpu": platform.processor() or platform.machine(), "threads": 1, "torch": torch.__version__, "datasets": {}}
for name, (d, C, N) in DATA.items():
    n = int(N * TRAIN / K)
    row = {"d": d, "classes": C, "client_train_flows": n}
    for mode in ("fedavg", "scaffold", "fedgmix"):
        row[f"epoch_s_{mode}"] = round(min(epoch(d, C, n, mode) for _ in range(2)), 3)
    X = torch.randn(5000, d, dtype=torch.float64)
    t0 = time.perf_counter(); gmm_em(X, 8, iters=100); row["em_fit_s_5000x8_100it"] = round(time.perf_counter() - t0, 2)
    from fedsim.fl.simulator import dp_gmm_em
    t0 = time.perf_counter(); dp_gmm_em(X, 2, torch.Generator().manual_seed(0), 4.0); row["dp_em_fit_s_5000x2_5it"] = round(time.perf_counter() - t0, 2)
    row["ratio_fedgmix_vs_fedavg"] = round(row["epoch_s_fedgmix"] / row["epoch_s_fedavg"], 2)
    out["datasets"][name] = row
    print(name, row, file=sys.stderr, flush=True)
print(json.dumps(out, indent=1))
