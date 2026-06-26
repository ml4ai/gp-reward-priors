#!/usr/bin/env python
# coding: utf-8
"""diagnose_sampling_tail.py — lower-tail MCMC convergence for a saved BNN run.

Reuses the SAVED chain weights of a completed `run_bnn_training.py` run (no
re-sampling) and reports convergence of the quantities that actually matter for
a risk-averse reward prior:

  * BULK  — reproduces the `pred_*` ESS/R-hat logged to W&B (a sanity check that
    this script matches the training pipeline).
  * VaR   — the lower 5% quantile (95% lower confidence bound).
  * CVaR  — the MEAN of the lowest 5% (conditional value-at-risk); this is the
    downstream quantity.  CVaR is harder to estimate than VaR because it averages
    the extreme tail, so it gets its own ESS / MCSE / R-hat.

Why weight-space and bulk diagnostics are NOT enough: the BNN posterior is
non-identifiable, so `param_*` R-hat/ESS are meaningless, and bulk `pred_*`
convergence reflects the median, not the tail (see memory
`bnn-sampling-tail-diagnostics`).

Methods (arviz_stats 1.1.0):
  * VaR  : ESS/MCSE via method="quantile", prob=alpha; tail R-hat via "folded".
  * CVaR : Rockafellar–Uryasev identity  CVaR_a = VaR_a + (1/a) E[min(X-VaR_a,0)]
    is EXACT, so CVaR's MC error is the mean-ESS/MCSE of the integrand
    u = (1/a) min(X - VaR_a, 0).  A between-chain CVaR spread is reported as an
    assumption-light cross-check.
  * MCSE is reported both absolute (reward units) and scale-free (÷ per-point
    posterior-predictive sd).  Reward magnitude spans orders of magnitude here,
    so only the scale-free `_max` is trustworthy.

Examples
--------
    python scripts_bnn/diagnose_sampling_tail.py \
        --run-dir exp/reward_learning/antmaze_medium_play_bnn/bnn-D4RL_antmaze-medium-play-v2-fb642974

    # override the auto-read config (e.g. a relocated dataset) and use a GPU
    python scripts_bnn/diagnose_sampling_tail.py --run-dir <dir> \
        --dataset data/antmaze/antmaze-medium-play-v2_pref_nt.hdf5 \
        --alpha 0.05 --device cuda
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import arviz_stats as azs
import torch
import yaml

from optbnn.bnn.nets.mlp import MLP
from optbnn.utils import util


def _load_run_config(run_dir):
    """Read the pyrallis-dumped config.yaml that every run writes to its OUT_DIR."""
    with open(os.path.join(run_dir, "config.yaml")) as f:
        return yaml.safe_load(f)


def _to_numpy_weights(weights):
    return tuple(
        np.asarray(a.detach().cpu().numpy()) if torch.is_tensor(a) else np.asarray(a)
        for a in weights
    )


def build_pred_chains(run_dir, dataset, width, depth, num_chains, b_rhat, device):
    """Load saved chains and evaluate each weight set at the diagnostic inputs.

    Mirrors the eval block of run_bnn_training.py exactly: the first `b_rhat`
    preference pairs, member 0, all non-padded timesteps, features only (the
    trailing attn_mask column is dropped).  Returns (pred_chains, x_rhat) with
    pred_chains shaped [chain, draw, point].
    """
    X, _ = util.load_pref_data(dataset, training_ratio=1.0)
    obs_dim = X.shape[-1] - 1
    n = min(b_rhat, X.shape[0])
    block = X[:n, 0, :, :]                                   # [B, T, obs_dim+1]
    valid = block[..., obs_dim].reshape(-1) > 0.5           # attn_mask column
    x_rhat = block[..., :obs_dim].reshape(-1, obs_dim).astype(np.float32)[valid]
    print(f"[data] {dataset}")
    print(f"[data] x_rhat {x_rhat.shape}  "
          f"({int(valid.sum())}/{valid.size} valid, input_dim={obs_dim})")

    net = MLP(input_dim=obs_dim, output_dim=1,
              hidden_dims=[width] * depth, activation_fn="relu").to(device)
    net.eval()
    x_t = torch.from_numpy(x_rhat).to(device)

    pred_chains, n_loaded = [], []
    for i in range(num_chains):
        path = os.path.join(run_dir, "sampling_f", f"chain_{i}",
                            "sampled_weights", "sampled_weights_0000000")
        ckpt = torch.load(path, weights_only=False, map_location=device)
        weights = [_to_numpy_weights(w) for w in ckpt["sampled_weights"]]
        n_loaded.append(len(weights))
        preds = []
        with torch.no_grad():
            for w in weights:
                for p, a in zip(net.parameters(), w):
                    p.copy_(torch.from_numpy(a).to(device))
                preds.append(net(x_t).detach().cpu().numpy().ravel())
        pred_chains.append(np.stack(preds))

    m = min(n_loaded)
    if len(set(n_loaded)) > 1:
        print(f"[warn] uneven sample counts {n_loaded}; truncating to {m}")
    pred_chains = np.stack([p[:m] for p in pred_chains])     # [chain, draw, point]
    print(f"[chains] loaded {n_loaded} -> using {num_chains}x{m} = "
          f"{num_chains * m} draws")
    return pred_chains, x_rhat


def _summ(name, arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    print(f"  {name:26s} min {a.min():9.4f}  median {np.median(a):9.4f}  "
          f"max {a.max():9.4f}")


def tail_diagnostics(pred_chains, alpha=0.05):
    """Print bulk, VaR(alpha), and CVaR(alpha) convergence diagnostics."""
    C, D, P = pred_chains.shape
    total = C * D
    flat = pred_chains.reshape(-1, P)
    pred_sd = flat.std(axis=0)                               # per-point spread
    eps = 1e-8

    print("\n=== BULK (should match logged pred_* ESS/R-hat) ===")
    _summ("ess_bulk", azs.ess(pred_chains))
    _summ("rhat_bulk (rank)", azs.rhat(pred_chains))

    print(f"\n=== VaR (lower {alpha:.0%} quantile = 95% lower bound) ===")
    ess_var = np.asarray(azs.ess(pred_chains, method="quantile", prob=alpha))
    mcse_var = np.asarray(azs.mcse(pred_chains, method="quantile", prob=alpha))
    _summ("VaR ESS", ess_var)
    print(f"  {'VaR ESS min / total':26s} {ess_var.min() / total:.4f}")
    _summ("VaR R-hat (folded)", azs.rhat(pred_chains, method="folded"))
    _summ("VaR MCSE / pred_sd", mcse_var / (pred_sd + eps))

    print(f"\n=== CVaR (mean of lowest {alpha:.0%} — the downstream quantity) ===")
    var = np.quantile(flat, alpha, axis=0)                   # VaR per point
    # Rockafellar-Uryasev integrand: CVaR = VaR + mean(u), u = (1/a)min(X-VaR,0)
    u = np.minimum(pred_chains - var[None, None, :], 0.0) / alpha
    ess_cvar = np.asarray(azs.ess(u, method="mean"))         # ESS for E[u]
    mcse_cvar = np.asarray(azs.mcse(u, method="mean"))       # sd(u)/sqrt(ESS)
    rhat_cvar = np.asarray(azs.rhat(u, method="folded"))
    # assumption-light cross-check: between-chain CVaR spread / sqrt(C)
    per_chain = np.stack([
        var + np.minimum(pred_chains[c] - var[None, :], 0.0).mean(axis=0) / alpha
        for c in range(C)
    ])
    mcse_bc = per_chain.std(axis=0, ddof=1) / np.sqrt(C)
    n_unresolved = int(np.sum(mcse_cvar / (pred_sd + eps) > 1.0))

    _summ("CVaR effective draws", ess_cvar)
    print(f"  {'raw draws below VaR':26s} {int(round(alpha * total))} of {total}")
    _summ("CVaR R-hat (folded)", rhat_cvar)
    _summ("CVaR MCSE / pred_sd", mcse_cvar / (pred_sd + eps))
    _summ("  cross-check (between-chain)", mcse_bc / (pred_sd + eps))
    print(f"  {'points MCSE>sd (unresolved)':26s} {n_unresolved} of {P} "
          f"({100 * n_unresolved / P:.2f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="A run's OUT_DIR (contains config.yaml and sampling_f/).")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Lower-tail fraction for VaR/CVaR (default 0.05).")
    ap.add_argument("--b-rhat", type=int, default=64,
                    help="Number of preference pairs to evaluate (default 64, "
                         "matching run_bnn_training.py).")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (default cpu).")
    # The following default to the run's config.yaml; override only if needed.
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--width", type=int, default=None,
                    help="Actual (already-expanded) layer width; default from config.")
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--num-chains", type=int, default=None)
    args = ap.parse_args()

    cfg = _load_run_config(args.run_dir)
    dataset = args.dataset or cfg["dataset"]
    width = args.width or cfg["width"]
    depth = args.depth or cfg["depth"]
    num_chains = args.num_chains or cfg["num_chains"]
    print(f"[run] {args.run_dir}")
    print(f"[run] seed={cfg.get('seed')} width={width} depth={depth} "
          f"num_chains={num_chains} num_samples={cfg.get('num_samples')}")

    pred_chains, _ = build_pred_chains(
        args.run_dir, dataset, width, depth, num_chains, args.b_rhat, args.device)
    tail_diagnostics(pred_chains, alpha=args.alpha)


if __name__ == "__main__":
    main()
