"""Single-process federated simulator (exact FedAvg semantics, GPU-resident client data).

Randomness is derived from (seed, round, client) at each use, so a run resumed from a
round checkpoint reproduces an uninterrupted run exactly and no RNG state has to be saved.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fedsim.models.mlp import MLP, is_bn_key
from fedsim.utils.io import atomic_torch_save


@dataclass
class ClientData:
    xtr: torch.Tensor
    ytr: torch.Tensor
    xva: torch.Tensor
    yva: torch.Tensor
    xte: torch.Tensor
    yte: torch.Tensor
    counts: torch.Tensor  # training class counts, task label space (float)

    @property
    def n(self) -> int:
        return int(self.ytr.numel())


@dataclass
class MethodCfg:
    name: str
    norm: str = "local"          # local | global (SFS)
    agg: str = "fedavg"          # fedavg | class_aware (CAA)
    beta: float = 0.1            # CAA smoothing
    loss: str = "ce"             # ce | la (PCL) | fedlc | fedrs
    tau: float = 1.0             # PCL / FedLC temperature
    fedrs_alpha: float = 0.5
    prox_mu: float = 0.0
    scaffold: bool = False
    fedbn: bool = False
    pooled: bool = False         # centralised reference
    client_momentum: float | None = None  # None -> fl.momentum; SCAFFOLD uses plain SGD (0.0)
    # ---- v2 candidates (defaults reproduce v1 exactly; run identity omits keys left at default) ----
    server_opt: str = "none"     # none | momentum (FedAvgM) | adam | yogi (FedOpt), on the aggregated pseudo-gradient
    server_lr: float = 1.0
    server_beta1: float = 0.9    # FedAvgM momentum / FedOpt first-moment decay
    server_beta2: float = 0.99
    server_eps: float = 1e-3     # FedOpt adaptivity tau
    head: str = "linear"         # linear | etf (fixed simplex ETF + learned projection) | frozen (random, fixed)
    head_scale: float = 10.0     # logit scale of the ETF head
    decorr: float = 0.0          # FedDecorr coefficient on the penultimate-feature correlation matrix
    ccvr: bool = False           # server-side head re-calibration from shared class-conditional feature moments
    ccvr_samples: int = 2000     # virtual features per class
    ccvr_epochs: int = 10
    ccvr_lr: float = 0.1
    ccvr_guard: bool = False     # keep the re-fitted head only if federated validation macro-F1 improves
    # ---- v3 candidates (pilot 2; defaults reproduce v1/v2 exactly) ----
    ntd: float = 0.0             # FedNTD (Lee et al., NeurIPS 2022): weight of not-true distillation from the global model
    ntd_tau: float = 1.0
    fedcm_alpha: float = 1.0     # FedCM (Xu et al., 2021): v = alpha*g + (1-alpha)*Delta_t in local steps; 1.0 = off
    anchor: float = 0.0          # FedFA (Zhou et al.): weight of the feature-anchor loss; 0 = off
    anchor_calib: bool = True    # FedFA classifier calibration on the anchors after each local step
    # ---- pilot 3: SFS-PCL-V (class-conditional mixture statistics shared once -> virtual samples) ----
    virtual: float = 0.0         # virtual samples per real sample in each local batch; 0 = off
    vmix_k: int = 8              # max mixture components per (client, class)
    vmix_min: int = 50           # privacy guard: every shared component must cover >= vmix_min samples
    vmix_cap: int = 5000         # max samples per (client, class) used to fit the local mixture
    vmix_cover_fit: bool = False # guard on coverage within the fitting sample (fix: w * len(fit sample), not w * n)
    vmix_dp_eps: float = 0.0     # > 0: release the mixtures with (eps, delta)-DP by noisy EM (implies the fit-sample guard)
    vmix_dp_delta: float = 1e-5
    vmix_dp_iters: int = 5
    vmix_dp_k: int = 2           # components per (client, class) under DP
    vmix_dp_clip: float = 1.0    # row L2 clip = vmix_dp_clip * sqrt(d)
    # ---- full-grid baseline: FedDyn (stateful clients) ----
    feddyn_alpha: float = 0.0    # FedDyn regularisation strength; 0 = off


@dataclass
class FLCfg:
    rounds: int = 50
    participation: float = 0.5
    local_epochs: int = 1
    batch_size: int = 256
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-4
    eval_every: int = 5
    ckpt_every: int = 10
    hidden: tuple = (256, 128, 64)
    dropout: float = 0.1
    model_norm: str = "batch"


def _gen(*key: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(np.random.SeedSequence(list(key)).generate_state(1)[0]))
    return g


def local_loss(logits, y, m: MethodCfg, counts: torch.Tensor):
    if m.loss == "ce":
        return F.cross_entropy(logits, y)
    if m.loss == "la":
        prior = (counts + 1.0) / (counts.sum() + counts.numel())
        return F.cross_entropy(logits + m.tau * torch.log(prior), y)
    if m.loss == "fedlc":
        cal = m.tau * torch.pow(counts.clamp_min(1e-8), -0.25)  # FL-bench reference: absent classes ~removed
        return F.cross_entropy(logits - cal, y)
    if m.loss == "fedrs":
        scale = torch.where(counts > 0, torch.ones_like(counts), torch.full_like(counts, m.fedrs_alpha))
        return F.cross_entropy(logits * scale, y)
    raise ValueError(m.loss)


def decorr_loss(f: torch.Tensor) -> torch.Tensor:
    """FedDecorr (Shi et al., ICLR 2023): squared Frobenius norm of the batch correlation matrix / d^2."""
    n, d = f.shape
    if n < 2:
        return f.sum() * 0.0
    z = (f - f.mean(0)) / torch.sqrt(f.var(0, unbiased=False) + 1e-5)
    corr = z.T @ z / n
    return (corr * corr).sum() / (d * d)


def ntd_loss(student: torch.Tensor, teacher: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    """FedNTD: KL(global || local) over softmax of the not-true-class logits only, scaled by tau^2."""
    n, C = student.shape
    keep = torch.ones_like(student, dtype=torch.bool)
    keep[torch.arange(n, device=student.device), y] = False
    s = student[keep].view(n, C - 1) / tau
    t = teacher[keep].view(n, C - 1) / tau
    return F.kl_div(F.log_softmax(s, 1), F.softmax(t, 1), reduction="batchmean") * tau * tau


def gmm_em(X: torch.Tensor, k: int, g: torch.Generator, iters: int = 100, tol: float = 1e-4, reg: float = 1e-3):
    """Full-covariance Gaussian mixture by EM (float64, on the data's device). Deterministic given g.
    Initialisation: k-means++ seeding, then hard assignment to the nearest seed."""
    n, d = X.shape
    dev = X.device
    first = int(torch.randint(n, (1,), generator=g))
    centers = [X[first]]
    for _ in range(1, k):
        d2 = torch.cdist(X, torch.stack(centers)).min(1).values ** 2
        p = (d2 / d2.sum()).cpu()
        centers.append(X[int(torch.multinomial(p, 1, generator=g))])
    C0 = torch.stack(centers)
    R = F.one_hot(torch.cdist(X, C0).argmin(1), k).to(X.dtype)
    eye = torch.eye(d, dtype=X.dtype, device=dev)
    prev = -math.inf
    for it in range(iters):
        Nk = R.sum(0) + 1e-10
        w = Nk / n
        mu = (R.T @ X) / Nk[:, None]
        Xc = X[None] - mu[:, None]                                   # k x n x d
        cov = torch.einsum("kn,kni,knj->kij", R.T, Xc, Xc) / Nk[:, None, None] + reg * eye
        L = torch.linalg.cholesky(cov)
        sol = torch.linalg.solve_triangular(L, Xc.transpose(1, 2), upper=False)   # k x d x n
        maha = (sol ** 2).sum(1)                                      # k x n
        logdet = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
        logp = -0.5 * (maha + logdet[:, None] + d * math.log(2 * math.pi)) + torch.log(w)[:, None]
        ll = torch.logsumexp(logp, 0)
        R = torch.softmax(logp, 0).T
        cur = float(ll.mean())
        if abs(cur - prev) < tol:
            break
        prev = cur
    Nk = R.sum(0) + 1e-10
    w = Nk / n
    mu = (R.T @ X) / Nk[:, None]
    Xc = X[None] - mu[:, None]
    cov = torch.einsum("kn,kni,knj->kij", R.T, Xc, Xc) / Nk[:, None, None] + reg * eye
    return w, mu, cov


def dp_rho(eps: float, delta: float) -> float:
    """Largest rho-zCDP budget that converts to (eps, delta)-DP via eps = rho + 2 sqrt(rho ln(1/delta))."""
    L = math.log(1.0 / delta)
    return (math.sqrt(L + eps) - math.sqrt(L)) ** 2


def dp_gmm_em(X: torch.Tensor, k: int, g: torch.Generator, eps: float, delta: float = 1e-5, iters: int = 5,
              clip: float = 1.0, reg: float = 1e-3):
    """(eps, delta)-DP Gaussian mixture by noisy EM (moment perturbation, cf. Park et al., AISTATS 2017).
    Rows are clipped to L2 norm C = clip * sqrt(d). Initialisation is data-independent (means ~ N(0, 0.25 I) in the
    shared standardised space, identity covariances, equal weights). Each iteration releases the soft counts N
    (L2 sensitivity 1), first moments S (sensitivity C) and second moments Q (Frobenius sensitivity C^2) with
    Gaussian noise of multiplier z; the release is 3 * iters Gaussian mechanisms, i.e. rho = 3 iters / (2 z^2)
    zCDP, and z is set so that the whole fit is (eps, delta)-DP for add/remove of one flow.
    Returns the noisy counts (used for the coverage guard and the merge weights), means and covariances."""
    n, d = X.shape
    dt, dev = X.dtype, X.device
    C = clip * math.sqrt(d)
    X = X * torch.clamp(C / X.norm(dim=1, keepdim=True).clamp_min(1e-12), max=1.0)
    z = math.sqrt(3 * iters / (2 * dp_rho(eps, delta)))
    eye = torch.eye(d, dtype=dt, device=dev)
    mu = (0.5 * torch.randn(k, d, generator=g, dtype=dt)).to(dev)
    cov = eye.expand(k, d, d).clone()
    w = torch.full((k,), 1.0 / k, dtype=dt, device=dev)
    N = torch.zeros(k, dtype=dt, device=dev)
    for _ in range(iters):
        Lc = torch.linalg.cholesky(cov)
        Xc = X[None] - mu[:, None]
        sol = torch.linalg.solve_triangular(Lc, Xc.transpose(1, 2), upper=False)
        maha = (sol ** 2).sum(1)
        logdet = 2 * torch.log(torch.diagonal(Lc, dim1=1, dim2=2)).sum(1)
        R = torch.softmax(-0.5 * (maha + logdet[:, None]) + torch.log(w)[:, None], 0)      # k x n
        N = R.sum(1) + z * torch.randn(k, generator=g, dtype=dt).to(dev)
        S = R @ X + z * C * torch.randn(k, d, generator=g, dtype=dt).to(dev)
        E = torch.randn(k, d, d, generator=g, dtype=dt).to(dev)
        E = (E + E.transpose(1, 2)) / math.sqrt(2)
        Q = torch.einsum("kn,ni,nj->kij", R, X, X) + z * C * C * E
        Np = N.clamp_min(1.0)
        mu = S / Np[:, None]
        cov = Q / Np[:, None, None] - mu[:, :, None] * mu[:, None, :]
        ev, V = torch.linalg.eigh((cov + cov.transpose(1, 2)) / 2)
        cov = V @ torch.diag_embed(ev.clamp_min(reg)) @ V.transpose(1, 2) + reg * eye
        cov = (cov + cov.transpose(1, 2)) / 2
        w = Np / Np.sum()
    return N, mu, cov


class Simulator:
    def __init__(self, clients: list[ClientData], n_classes: int, in_dim: int, method: MethodCfg,
                 fl: FLCfg, seed: int, device: str, run_dir: Path, logger=None, vmix_cache=None):
        self.c, self.C, self.m, self.fl, self.seed = clients, n_classes, method, fl, seed
        self.device, self.run_dir, self.log = device, Path(run_dir), logger
        self.vmix_cache = vmix_cache
        torch.manual_seed(seed)
        self.model = MLP(in_dim, n_classes, fl.hidden, fl.dropout, fl.model_norm,
                         head=method.head, head_scale=method.head_scale).to(device)
        self.keys = list(self.model.state_dict().keys())
        self.bn_keys = {k for k in self.keys if is_bn_key(k, self.model)}
        self.param_keys = [n for n, p in self.model.named_parameters() if p.requires_grad]
        self.frozen_keys = {n for n, p in self.model.named_parameters() if not p.requires_grad}
        self.bn_buffer_keys = {k for k in self.bn_keys if k not in self.param_keys}
        self.global_sd = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.bn_local: dict[int, dict] = {}
        self.c_global = {k: torch.zeros_like(self.global_sd[k]) for k in self.param_keys} if method.scaffold else None
        self.c_local: dict[int, dict] = {}
        self.dyn_local: dict[int, dict] = {}     # FedDyn client state: running gradient correction g_k
        self.dyn_h = ({k: torch.zeros_like(self.global_sd[k]) for k in self.param_keys}
                      if method.feddyn_alpha > 0 else None)
        self.server_state: dict = {}
        self.teacher = copy.deepcopy(self.model) if method.ntd > 0 else None
        self._round_stats: dict[int, dict] = {}
        self.round = 0
        self.best = {"val_macro_f1": -1.0, "round": 0, "sd": None, "bn_local": None}
        self.curve: list[dict] = []
        self.comm_up_bytes = 0.0
        self.train_time = 0.0
        self.K = len(clients)
        self.n_params = sum(v.numel() for k, v in self.global_sd.items() if k in self.param_keys)
        self.n_bn_params = sum(self.global_sd[k].numel() for k in self.param_keys if k in self.bn_keys)
        self._one_time_comm()
        self.vmix = self._build_mixtures() if method.virtual > 0 else None

    # ---------------- communication accounting ----------------
    def _one_time_comm(self):
        d = self.c[0].xtr.shape[1]
        if self.m.norm == "global" and not self.m.pooled:
            self.comm_up_bytes += self.K * (2 * d + 1) * 4
        if self.m.agg == "class_aware" and not self.m.pooled:
            self.comm_up_bytes += self.K * self.C * 4

    def _round_up_bytes(self, n_sampled: int) -> float:
        if self.m.pooled:
            return 0.0
        p = self.n_params - (self.n_bn_params if self.m.fedbn else 0)
        extra = 0
        if self.m.anchor > 0:   # FedFA: per-class feature sums + counts with each model upload
            extra = self.C * (self._feat_dim() + 1)
        return n_sampled * (p * (2 if self.m.scaffold else 1) + extra) * 4

    def _feat_dim(self) -> int:
        return int(self.fl.hidden[-1])

    def _anchors(self) -> torch.Tensor:
        A = self.server_state.get("anchors")
        if A is None:   # FedFA initialisation: pairwise-orthogonal unit anchors (identity columns)
            D = self._feat_dim()
            A = torch.zeros(self.C, D, device=self.device)
            if D >= self.C:
                A[:, :self.C] = torch.eye(self.C, device=self.device)
            else:
                q, _ = torch.linalg.qr(torch.randn(D, self.C, generator=_gen(self.seed, 515151)))
                A = q.T.contiguous().to(self.device)
            self.server_state["anchors"] = A
        return A

    # ---------------- SFS-PCL-V: class-conditional mixtures ----------------
    def _build_mixtures(self):
        """Each client fits a Gaussian mixture to every class it holds (on SFS-normalised training inputs) and
        uploads it once; components covering fewer than vmix_min samples are dropped (privacy guard). The server
        merges, per class, all clients' components weighted by the samples they cover."""
        m = self.m
        d = self.c[0].xtr.shape[1]
        if self.vmix_cache is not None and Path(self.vmix_cache).exists():
            st = torch.load(self.vmix_cache, map_location=self.device, weights_only=False)
            self.comm_up_bytes += st["n_shared"] * (1 + d + d * (d + 1) // 2) * 4
            self.vmix_classes = st["classes"].to(self.device)
            self.vmix_n_components = st["n_shared"]
            return {c: tuple(t.to(self.device) for t in v) for c, v in st["mix"].items()}
        comps = {c: [] for c in range(self.C)}            # class -> list of (n_covered, mean, cov)
        n_shared = 0
        for k, cd in enumerate(self.c):
            X = cd.xtr.detach().cpu().double().numpy()
            y = cd.ytr.detach().cpu().numpy()
            for c in np.unique(y).tolist():
                Xc = X[y == c]
                n = len(Xc)
                if n < m.vmix_min:
                    continue
                if n > m.vmix_cap:
                    rs = np.random.default_rng([self.seed, k, c, 31])
                    Xc = Xc[rs.choice(n, m.vmix_cap, replace=False)]
                if m.vmix_dp_eps > 0:
                    # DP release: guard and merge weights use the noisy counts of the fitting sample
                    gN, gmu, gcov = dp_gmm_em(torch.as_tensor(Xc, dtype=torch.float64, device=self.device),
                                              int(m.vmix_dp_k), _gen(self.seed, k, c, 41), float(m.vmix_dp_eps),
                                              float(m.vmix_dp_delta), int(m.vmix_dp_iters), float(m.vmix_dp_clip))
                    for cnt, mu, cov in zip(gN.cpu().numpy(), gmu.cpu().numpy(), gcov.cpu().numpy()):
                        if cnt >= m.vmix_min:
                            comps[c].append((float(cnt), mu, cov))
                            n_shared += 1
                    continue
                kk = int(max(1, min(m.vmix_k, len(Xc) // m.vmix_min)))
                gw, gmu, gcov = gmm_em(torch.as_tensor(Xc, dtype=torch.float64, device=self.device), kk,
                                       _gen(self.seed, k, c, 37))
                for w, mu, cov in zip(gw.cpu().numpy(), gmu.cpu().numpy(), gcov.cpu().numpy()):
                    cover = w * n
                    guard = w * len(Xc) if m.vmix_cover_fit else cover
                    if guard >= m.vmix_min:
                        comps[c].append((float(cover), mu, cov))
                        n_shared += 1
        self.comm_up_bytes += n_shared * (1 + d + d * (d + 1) // 2) * 4
        mix = {}
        for c, lst in comps.items():
            if not lst:
                continue
            w = torch.tensor([x[0] for x in lst], dtype=torch.float64)
            mus = torch.tensor(np.stack([x[1] for x in lst]), dtype=torch.float64)
            Ls = torch.linalg.cholesky(torch.tensor(np.stack([x[2] for x in lst]), dtype=torch.float64)
                                       + 1e-6 * torch.eye(d, dtype=torch.float64))
            mix[c] = ((w / w.sum()).float().to(self.device), mus.float().to(self.device), Ls.float().to(self.device))
        self.vmix_classes = torch.tensor(sorted(mix), dtype=torch.long, device=self.device)
        self.vmix_n_components = n_shared
        if self.vmix_cache is not None:
            Path(self.vmix_cache).parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save({"n_shared": n_shared, "classes": self.vmix_classes.cpu(),
                               "mix": {c: tuple(t.cpu() for t in v) for c, v in mix.items()}}, self.vmix_cache)
        return mix

    def _pack_mixtures(self):
        """Flatten all classes' components into one table so a virtual batch is drawn without Python loops."""
        cls = self.vmix_classes.tolist()
        lens = [len(self.vmix[c][0]) for c in cls]
        kmax = max(lens)
        cum = torch.full((len(cls), kmax), 2.0, device=self.device)
        for p, c in enumerate(cls):
            cum[p, :lens[p]] = torch.cumsum(self.vmix[c][0], 0)
        self._vcum = cum
        self._vlen = torch.tensor(lens, device=self.device)
        self._voff = torch.tensor([0] + list(np.cumsum(lens)[:-1]), device=self.device)
        self._vmu = torch.cat([self.vmix[c][1] for c in cls])
        self._vL = torch.cat([self.vmix[c][2] for c in cls])

    def _virtual_batch(self, n: int, g: torch.Generator):
        """n virtual inputs, classes uniform over the classes with shared components (balanced);
        within a class, components are drawn by their weights."""
        if not hasattr(self, "_vcum"):
            self._pack_mixtures()
        d = self.c[0].xtr.shape[1]
        p = torch.randint(len(self.vmix_classes), (n,), generator=g).to(self.device)
        cls = self.vmix_classes[p]
        z = torch.randn(n, d, generator=g).to(self.device)
        u = torch.rand(n, generator=g).to(self.device)
        j = torch.searchsorted(self._vcum[p], u[:, None]).squeeze(1)
        j = torch.minimum(j, self._vlen[p] - 1)
        gi = self._voff[p] + j
        xs = self._vmu[gi] + torch.bmm(self._vL[gi], z[:, :, None]).squeeze(2)
        return xs, cls

    # ---------------- client update ----------------
    def _client_update(self, k: int, t: int):
        m, fl, cd = self.m, self.fl, self.c[k]
        model = self.model
        sd = dict(self.global_sd)
        if m.fedbn and k in self.bn_local:
            sd.update(self.bn_local[k])
        model.load_state_dict(sd)
        model.train()
        torch.manual_seed(int(np.random.SeedSequence([self.seed, t, k, 7]).generate_state(1)[0]))
        mom = fl.momentum if m.client_momentum is None else m.client_momentum
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=fl.lr, momentum=mom, weight_decay=fl.weight_decay)
        gref = [self.global_sd[n] for n in self.param_keys] if m.prox_mu > 0 else None
        if m.scaffold:
            ck = self.c_local.get(k) or {n: torch.zeros_like(v) for n, v in self.c_global.items()}
            corr = {n: self.c_global[n] - ck[n] for n in self.param_keys}
        g = _gen(self.seed, t, k)
        steps = 0
        counts = cd.counts
        if self.teacher is not None:
            self.teacher.load_state_dict(sd)
            self.teacher.eval()
        cm_delta = self.server_state.get("fedcm_delta") if m.fedcm_alpha < 1.0 else None
        gv = _gen(self.seed, t, k, 77) if self.vmix is not None else None
        need_feats = m.decorr > 0 or m.anchor > 0
        if m.anchor > 0:
            A = self._anchors()
            a_sum = torch.zeros_like(A)
            a_cnt = torch.zeros(self.C, device=self.device)
            head_params = [p for p in model.head.parameters() if p.requires_grad]
            head_opt = torch.optim.SGD(head_params, lr=fl.lr) if m.anchor_calib and head_params else None
            cls = torch.arange(self.C, device=self.device)
        for _ in range(fl.local_epochs):
            perm = torch.randperm(cd.n, generator=g).to(self.device)
            for i in range(0, cd.n, fl.batch_size):
                b = perm[i:i + fl.batch_size]
                if b.numel() < 2 and self.fl.model_norm == "batch":
                    continue
                xb, yb = cd.xtr[b], cd.ytr[b]
                nv = int(round(m.virtual * len(b))) if self.vmix is not None else 0
                if nv > 0:
                    xv, yv = self._virtual_batch(nv, gv)
                    xb = torch.cat([xb, xv])
                if need_feats:
                    feats = model.features(xb)
                    logits = model.classify(feats)
                else:
                    logits = model(xb)
                if nv > 0:
                    # Real samples: PCL with the local prior; balanced virtual samples: plain CE (uniform prior).
                    nr = len(b)
                    lr_ = local_loss(logits[:nr], yb, m, counts)
                    lv = F.cross_entropy(logits[nr:], yv)
                    loss = (nr * lr_ + nv * lv) / (nr + nv)
                    logits = logits[:nr]
                    if need_feats:
                        feats = feats[:nr]
                else:
                    loss = local_loss(logits, yb, m, counts)
                if m.decorr > 0:
                    loss = loss + m.decorr * decorr_loss(feats)
                if self.teacher is not None:
                    with torch.no_grad():
                        tl = self.teacher(cd.xtr[b])
                    loss = loss + m.ntd * ntd_loss(logits, tl, yb, m.ntd_tau)
                if m.anchor > 0:
                    loss = loss + m.anchor * ((feats - A[yb]) ** 2).sum(1).mean()
                    a_sum.index_add_(0, yb, feats.detach())
                    a_cnt.index_add_(0, yb, torch.ones_like(yb, dtype=a_cnt.dtype))
                if gref is not None:
                    prox = sum(((p - q) ** 2).sum() for p, q in zip(params, gref))
                    loss = loss + 0.5 * m.prox_mu * prox
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if m.scaffold:
                    for n, p in model.named_parameters():
                        p.grad.add_(corr[n])
                if m.feddyn_alpha > 0:
                    # grad of  L_k(theta) - <g_k, theta> + alpha/2 ||theta - theta_global||^2
                    gk = self.dyn_local.get(k)
                    for n, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        p.grad.add_(p.detach() - self.global_sd[n], alpha=m.feddyn_alpha)
                        if gk is not None:
                            p.grad.sub_(gk[n])
                if cm_delta is not None:
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            p.grad.mul_(m.fedcm_alpha).add_(cm_delta[n], alpha=1.0 - m.fedcm_alpha)
                opt.step()
                if m.anchor > 0 and head_opt is not None:
                    la = F.cross_entropy(model.classify(A), cls)
                    head_opt.zero_grad(set_to_none=True)
                    la.backward()
                    head_opt.step()
                steps += 1
        local = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
        dc = None
        if m.scaffold and steps > 0:
            new_ck = {n: ck[n] - self.c_global[n] + (self.global_sd[n] - local[n]) / (steps * fl.lr) for n in self.param_keys}
            dc = {n: new_ck[n] - ck[n] for n in self.param_keys}
            self.c_local[k] = new_ck
        if m.fedbn:
            self.bn_local[k] = {k2: local[k2] for k2 in self.bn_keys}
        if m.feddyn_alpha > 0:
            gk = self.dyn_local.get(k) or {n: torch.zeros_like(self.global_sd[n]) for n in self.param_keys}
            self.dyn_local[k] = {n: gk[n] - m.feddyn_alpha * (local[n] - self.global_sd[n]) for n in self.param_keys}
        st = {"steps": steps}
        if m.anchor > 0:
            st["a_sum"], st["a_cnt"] = a_sum, a_cnt
        self._round_stats[k] = st
        return local, dc

    # ---------------- aggregation ----------------
    def _aggregate(self, sampled: list[int], locals_: list[dict], dcs: list):
        n = torch.tensor([float(self.c[k].n) for k in sampled], device=self.device)
        w = n / n.sum()
        new = {}
        for key in self.keys:
            if (self.m.fedbn and key in self.bn_keys) or key in self.frozen_keys:
                new[key] = self.global_sd[key]
                continue
            ref = self.global_sd[key]
            if not torch.is_floating_point(ref):
                new[key] = torch.stack([l[key] for l in locals_]).max(0).values
                continue
            stack = torch.stack([l[key] for l in locals_])
            new[key] = torch.einsum("k,k...->...", w, stack)
        if self.m.agg == "class_aware":
            cnt = torch.stack([self.c[k].counts for k in sampled])          # S x C
            om = cnt + self.m.beta * n[:, None] / self.C
            den = om.sum(0, keepdim=True)
            om = torch.where(den > 0, om / den.clamp_min(1e-12), w[:, None].expand_as(om))
            Wst = torch.stack([l["head.weight"] for l in locals_])          # S x C x d
            bst = torch.stack([l["head.bias"] for l in locals_])            # S x C
            new["head.weight"] = torch.einsum("sc,scd->cd", om, Wst)
            new["head.bias"] = (om * bst).sum(0)
        if self.m.scaffold:
            for nme in self.param_keys:
                valid = [d[nme] for d in dcs if d is not None]
                if valid:
                    self.c_global[nme] = self.c_global[nme] + torch.stack(valid).sum(0) / self.K
        if self.m.fedcm_alpha < 1.0:
            # Delta_{t+1}: sample-weighted mean per-step client displacement, divided by the client lr and
            # rescaled by (1 - client momentum) so that it estimates a gradient (FedCM, eq. for Delta).
            mom = self.fl.momentum if self.m.client_momentum is None else self.m.client_momentum
            delta = {}
            for key in self.param_keys:
                acc = torch.zeros_like(self.global_sd[key])
                for wk, k, l in zip(w, sampled, locals_):
                    s_k = max(1, self._round_stats[k]["steps"])
                    acc += wk * (self.global_sd[key] - l[key]) / (s_k * self.fl.lr)
                delta[key] = acc * (1.0 - mom)
            self.server_state["fedcm_delta"] = delta
        if self.m.anchor > 0:
            s_ = sum(self._round_stats[k]["a_sum"] for k in sampled)
            c_ = sum(self._round_stats[k]["a_cnt"] for k in sampled)
            A = self._anchors().clone()
            seen = c_ > 0
            A[seen] = s_[seen] / c_[seen, None]
            self.server_state["anchors"] = A
        if self.m.feddyn_alpha > 0:
            a = self.m.feddyn_alpha
            for key in self.param_keys:
                drift = sum(l[key] - self.global_sd[key] for l in locals_)
                self.dyn_h[key] = self.dyn_h[key] - a * drift / self.K
                new[key] = new[key] - self.dyn_h[key] / a
        self._round_stats = {}
        if self.m.server_opt != "none":
            new = self._server_step(new)
        self.global_sd = new

    def _server_step(self, avg: dict) -> dict:
        """FedAvgM / FedAdam / FedYogi on the pseudo-gradient g = x_t - avg (Hsu et al. 2019; Reddi et al. 2021).
        BatchNorm running statistics are not gradients; they keep the plain average."""
        m, st, out = self.m, self.server_state, dict(avg)
        for k in self.param_keys:
            if self.m.fedbn and k in self.bn_keys:
                continue
            g = self.global_sd[k] - avg[k]
            if m.server_opt == "momentum":
                v = st.get(("v", k))
                v = g.clone() if v is None else m.server_beta1 * v + g
                st[("v", k)] = v
                out[k] = self.global_sd[k] - m.server_lr * v
            elif m.server_opt in ("adam", "yogi"):
                mt = st.get(("m", k), torch.zeros_like(g))
                vt = st.get(("v", k), torch.full_like(g, m.server_eps ** 2))
                mt = m.server_beta1 * mt + (1 - m.server_beta1) * g
                g2 = g * g
                if m.server_opt == "adam":
                    vt = m.server_beta2 * vt + (1 - m.server_beta2) * g2
                else:
                    vt = vt - (1 - m.server_beta2) * g2 * torch.sign(vt - g2)
                st[("m", k)], st[("v", k)] = mt, vt
                out[k] = self.global_sd[k] - m.server_lr * mt / (vt.sqrt() + m.server_eps)
            else:
                raise ValueError(m.server_opt)
        return out

    # ---------------- CCVR head re-calibration (server side, after training) ----------------
    @torch.no_grad()
    def class_feature_moments(self, sd: dict):
        """Per-client (n_c, sum f, sum f f^T) on training features; the server only needs their sums."""
        model = self.model
        model.load_state_dict(sd)
        model.eval()
        D = None
        tot_n = tot_s = tot_ss = None
        for cd in self.c:
            f = torch.cat([model.features(cd.xtr[i:i + 65536]) for i in range(0, cd.n, 65536)]).double()
            if D is None:
                D = f.shape[1]
                tot_n = torch.zeros(self.C, dtype=torch.float64, device=f.device)
                tot_s = torch.zeros(self.C, D, dtype=torch.float64, device=f.device)
                tot_ss = torch.zeros(self.C, D, D, dtype=torch.float64, device=f.device)
            for c in torch.unique(cd.ytr).tolist():
                fc = f[cd.ytr == c]
                tot_n[c] += fc.shape[0]
                tot_s[c] += fc.sum(0)
                tot_ss[c] += fc.T @ fc
        return tot_n, tot_s, tot_ss

    def calibrate_head(self, sd: dict) -> dict:
        """CCVR (Luo et al., NeurIPS 2021): fit Gaussians to the pooled class-conditional features and
        re-train the head on balanced virtual features. Uses one extra statistics upload per client."""
        m, model = self.m, self.model
        n, s, ss = self.class_feature_moments(sd)
        D = s.shape[1]
        self.comm_up_bytes += self.K * self.C * (1 + D + D * (D + 1) // 2) * 4
        g = _gen(self.seed, 424242)
        feats, labels = [], []
        for c in range(self.C):
            if n[c] < 2:
                continue
            mu = s[c] / n[c]
            cov = ss[c] / n[c] - torch.outer(mu, mu)
            cov = (cov + cov.T) / 2 + 1e-4 * torch.eye(D, dtype=cov.dtype, device=cov.device)
            evals, evecs = torch.linalg.eigh(cov.cpu())
            L = evecs * evals.clamp_min(0).sqrt()
            z = torch.randn(m.ccvr_samples, D, generator=g, dtype=torch.float64)
            feats.append((mu.cpu() + z @ L.T).float())
            labels.append(torch.full((m.ccvr_samples,), c, dtype=torch.long))
        Fv = torch.cat(feats).to(self.device)
        Yv = torch.cat(labels).to(self.device)
        model.load_state_dict(sd)
        model.train()
        head_params = [p for p in model.head.parameters()]
        for p in head_params:
            p.requires_grad_(True)
        opt = torch.optim.SGD(head_params, lr=m.ccvr_lr, momentum=0.9)
        for ep in range(m.ccvr_epochs):
            perm = torch.randperm(len(Yv), generator=_gen(self.seed, 424243, ep)).to(self.device)
            for i in range(0, len(Yv), 256):
                b = perm[i:i + 256]
                loss = F.cross_entropy(model.classify(Fv[b]), Yv[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        if m.head != "linear":
            for p in head_params:
                p.requires_grad_(False)
        out = dict(sd)
        for k, v in model.state_dict().items():
            if k.startswith("head."):
                out[k] = v.detach().clone()
        return out

    # ---------------- evaluation ----------------
    @torch.no_grad()
    def predict(self, split: str, sd: dict, bn_local: dict | None):
        model = self.model
        ys, ps, cs = [], [], []
        for k, cd in enumerate(self.c):
            x, y = (cd.xva, cd.yva) if split == "val" else (cd.xte, cd.yte)
            if y.numel() == 0:
                continue
            s = dict(sd)
            if self.m.fedbn and bn_local and k in bn_local:
                s.update(bn_local[k])
            model.load_state_dict(s)
            model.eval()
            out = torch.cat([model(x[i:i + 65536]).argmax(1) for i in range(0, len(y), 65536)])
            ys.append(y.cpu()); ps.append(out.cpu()); cs.append(torch.full((len(y),), k))
        return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(cs).numpy()

    def _eval_val(self, t: int):
        from sklearn.metrics import f1_score, accuracy_score
        y, p, _ = self.predict("val", self.global_sd, self.bn_local)
        f1 = f1_score(y, p, average="macro", labels=np.unique(y), zero_division=0)
        acc = accuracy_score(y, p)
        rec = {"round": t, "val_macro_f1": float(f1), "val_acc": float(acc),
               "comm_up_MB": self.comm_up_bytes / 1e6, "train_time_s": self.train_time}
        self.curve.append(rec)
        if f1 > self.best["val_macro_f1"]:
            self.best = {"val_macro_f1": float(f1), "round": t,
                         "sd": {k: v.clone() for k, v in self.global_sd.items()},
                         "bn_local": copy.deepcopy(self.bn_local) if self.m.fedbn else None}
        if self.log:
            self.log(rec)
        return rec

    # ---------------- checkpointing ----------------
    def state(self) -> dict:
        return {"round": self.round, "global_sd": self.global_sd, "bn_local": self.bn_local,
                "c_global": self.c_global, "c_local": self.c_local, "best": self.best,
                "curve": self.curve, "comm_up_bytes": self.comm_up_bytes, "train_time": self.train_time,
                "server_state": self.server_state, "dyn_local": self.dyn_local, "dyn_h": self.dyn_h}

    def load_state(self, s: dict):
        self.round, self.global_sd, self.bn_local = s["round"], s["global_sd"], s["bn_local"]
        self.c_global, self.c_local, self.best = s["c_global"], s["c_local"], s["best"]
        self.curve, self.comm_up_bytes, self.train_time = s["curve"], s["comm_up_bytes"], s["train_time"]
        self.server_state = s.get("server_state", {})
        self.dyn_local, self.dyn_h = s.get("dyn_local", {}), s.get("dyn_h", self.dyn_h)

    def run(self, ckpt_path: Path | None = None):
        fl = self.fl
        m_sel = max(1, int(round(fl.participation * self.K)))
        while self.round < fl.rounds:
            t = self.round + 1
            rng = np.random.default_rng([self.seed, t, 99])
            sampled = sorted(rng.choice(self.K, size=m_sel, replace=False).tolist())
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            locals_, dcs = [], []
            for k in sampled:
                l, dc = self._client_update(k, t)
                locals_.append(l); dcs.append(dc)
            self._aggregate(sampled, locals_, dcs)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            self.train_time += time.perf_counter() - t0
            self.comm_up_bytes += self._round_up_bytes(len(sampled))
            self.round = t
            if t % fl.eval_every == 0 or t == fl.rounds:
                self._eval_val(t)
            if ckpt_path is not None and (t % fl.ckpt_every == 0 or t == fl.rounds):
                atomic_torch_save(self.state(), ckpt_path)
        return self.best
