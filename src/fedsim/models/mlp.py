"""MLP body + classifier head for tabular flow features.

Head variants (v2; the default "linear" is the v1 model, with an unchanged state_dict):
- linear: learned nn.Linear head.
- etf:    a learned projection (d -> d), L2 normalisation and a fixed simplex-ETF head scaled by
          `head_scale` (FedETF-style). The ETF weights are frozen and never uploaded.
- frozen: a randomly initialised linear head that is frozen for the whole run (FedBABU-style body training).
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _norm(kind: str, d: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(d)
    if kind == "layer":
        return nn.LayerNorm(d)
    if kind == "none":
        return nn.Identity()
    raise ValueError(kind)


def simplex_etf(n_classes: int, d: int, seed: int = 0) -> torch.Tensor:
    """C x d simplex equiangular tight frame: unit-norm rows with pairwise cosine -1/(C-1)."""
    if d < n_classes - 1:
        raise ValueError(f"ETF needs feature dim >= C-1 (d={d}, C={n_classes})")
    g = torch.Generator().manual_seed(seed)
    k = min(d, n_classes)
    U, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))   # d x k orthonormal columns
    M = torch.eye(n_classes, dtype=torch.float64) - 1.0 / n_classes                 # centred identity, rank C-1
    if k < n_classes:                                                               # d == C-1: use the rank-(C-1) factor
        evals, evecs = torch.linalg.eigh(M)
        M = (evecs[:, 1:] * evals[1:].sqrt()).T                                     # (C-1) x C, M^T M = centred identity
    W = (U @ M).T                                                                   # C x d
    W = W / W.norm(dim=1, keepdim=True)
    return W.float()


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden=(256, 128, 64), dropout: float = 0.1, norm: str = "batch",
                 head: str = "linear", head_scale: float = 10.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), _norm(norm, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head_kind, self.head_scale = head, float(head_scale)
        if head == "etf":
            self.proj = nn.Linear(d, d)
            self.head = nn.Linear(d, n_classes, bias=False)
            with torch.no_grad():
                self.head.weight.copy_(simplex_etf(n_classes, d))
            self.head.weight.requires_grad_(False)
        else:
            self.head = nn.Linear(d, n_classes)
            if head == "frozen":
                for p in self.head.parameters():
                    p.requires_grad_(False)
            elif head != "linear":
                raise ValueError(head)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Input to the classifier head (the penultimate representation)."""
        f = self.body(x)
        if self.head_kind == "etf":
            f = F.normalize(self.proj(f), dim=1)
        return f

    def classify(self, f: torch.Tensor) -> torch.Tensor:
        z = self.head(f)
        return z * self.head_scale if self.head_kind == "etf" else z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.features(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def mlp_flops(in_dim: int, n_classes: int, hidden, head: str = "linear") -> int:
    """Multiply-accumulate x2 for linear layers (norm/activation ignored, <1%)."""
    dims = [in_dim, *hidden, n_classes]
    fl = sum(2 * a * b for a, b in zip(dims[:-1], dims[1:]))
    if head == "etf":
        fl += 2 * hidden[-1] * hidden[-1]
    return int(fl)


def is_bn_key(name: str, model: nn.Module) -> bool:
    mod = name.rsplit(".", 1)[0]
    m = dict(model.named_modules()).get(mod)
    return isinstance(m, nn.BatchNorm1d)
