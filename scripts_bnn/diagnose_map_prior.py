#!/usr/bin/env python
# coding: utf-8
"""diagnose_map_prior.py — validation & diagnostics for MapInformedGPPrior.

Run the handoff's load-bearing checks (§9) before trusting any reward numbers:

  1. PSD check on K_geo and a sampled Gram matrix; condition number.
  2. Prior-sample heatmaps over all free cells — the key qualitative check.
     Look for smooth variation along corridors, sharp decorrelation across
     walls, and NO systematic bias toward any cell (no goal leakage).
  3. Correlation-vs-distance: kernel should track graph (geodesic) distance and
     violate Euclidean monotonicity at wall-separated pairs.
  4. (Optional) Cell-mapping overlay for a sample of measurement points, to
     confirm c(s) maps torso (x, y) to the right maze cell.

Outputs PNGs to --out-dir.  Layout is taken from a live D4RL env (--env, the
authoritative path) or the hardcoded fallback (--size).

Examples
--------
    python scripts_bnn/diagnose_map_prior.py --env antmaze-medium-play-v2
    python scripts_bnn/diagnose_map_prior.py --size large --eta 1.0
    python scripts_bnn/diagnose_map_prior.py --env antmaze-medium-play-v2 \
        --meas data/antmaze/antmaze-medium-play-v2_tuning_set.hdf5
"""

import argparse
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from optbnn.gp.maze_layouts import get_antmaze_layout
from optbnn.gp.models.map_informed_prior import MapInformedGPPrior


def _cell_grid(prior, values):
    """Scatter per-node `values` back onto a (R, C) grid (NaN on walls)."""
    grid = np.full(prior.free_mask.shape, np.nan)
    for k, (r, c) in enumerate(prior.free_rc):
        grid[r, c] = values[k]
    return grid


def _graph_distances(prior):
    """All-pairs shortest-path (hop) distance over the free-space grid graph."""
    n = prior.n_cells
    A = prior._adjacency
    INF = np.inf
    dist = np.full((n, n), INF)
    for s in range(n):
        # BFS
        dist[s, s] = 0.0
        frontier = [s]
        d = 0
        seen = {s}
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in np.nonzero(A[u])[0]:
                    if v not in seen:
                        seen.add(int(v))
                        dist[s, v] = d
                        nxt.append(int(v))
            frontier = nxt
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None,
                    help="D4RL env id to extract the layout from (authoritative).")
    ap.add_argument("--size", default="medium", choices=["medium", "large"],
                    help="Hardcoded fallback layout when --env is not given.")
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--sig-c2", type=float, default=1.0)
    ap.add_argument("--sig-g2", type=float, default=1.0)
    ap.add_argument("--sig-n2", type=float, default=1e-3)
    ap.add_argument("--n-samples", type=int, default=4,
                    help="Number of prior-sample heatmaps to render.")
    ap.add_argument("--meas", default=None,
                    help="Optional measurement HDF5 for the cell-mapping overlay.")
    ap.add_argument("--out-dir", default="./exp/map_prior_diag")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    free_mask, scaling, offset = get_antmaze_layout(args.size, env_name=args.env)
    prior = MapInformedGPPrior(
        free_mask=free_mask, scaling=scaling, offset=offset,
        eta=args.eta, sig_c2=args.sig_c2, sig_g2=args.sig_g2, sig_n2=args.sig_n2,
        device=torch.device("cpu"),
    )
    tag = (args.env or f"{args.size}-hardcoded")
    print(f"[layout] {tag}: grid {free_mask.shape}, {prior.n_cells} free cells, "
          f"scaling={scaling}, offset={offset}")

    # --- 1. PSD check + condition number ---------------------------------
    Kgeo = prior.Kgeo.cpu().numpy()
    eig_geo = np.linalg.eigvalsh(Kgeo)
    Xc = prior.free_cell_inputs()
    Xc_t = torch.from_numpy(Xc).double()
    K = prior.gram(Xc_t).cpu().numpy()
    eig_K = np.linalg.eigvalsh(K)
    print(f"[psd] Kgeo  min eig = {eig_geo.min():.3e}  (PSD ok: {eig_geo.min() >= -1e-8})")
    print(f"[psd] Gram  min eig = {eig_K.min():.3e}  max eig = {eig_K.max():.3e}  "
          f"cond = {eig_K.max() / eig_K.min():.3e}  (PD ok: {eig_K.min() > 0})")

    # --- 2. Prior-sample heatmaps ----------------------------------------
    samples = prior.sample_prior(Xc_t, num_samples=args.n_samples).cpu().numpy()
    ncol = args.n_samples
    fig, axes = plt.subplots(1, ncol, figsize=(3.2 * ncol, 3.2))
    if ncol == 1:
        axes = [axes]
    vmax = np.nanmax(np.abs(samples))
    for j in range(ncol):
        grid = _cell_grid(prior, samples[:, j])
        im = axes[j].imshow(grid, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                            origin="upper")
        axes[j].set_title(f"prior sample {j}")
        axes[j].set_xticks([]); axes[j].set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle(f"Prior-sample heatmaps — {tag} (eta={args.eta})")
    p = os.path.join(args.out_dir, f"heatmaps_{tag}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[heatmap] wrote {p}  — check: smooth along corridors, sharp across "
          "walls, no consistent corner bias (goal leakage).")

    # --- 3. Correlation vs distance --------------------------------------
    gdist = _graph_distances(prior)
    # Euclidean distance between free-cell centres (grid coords).
    rc = prior.free_rc.astype(float)
    edist = np.sqrt(((rc[:, None, :] - rc[None, :, :]) ** 2).sum(-1))
    # Correlation from K_geo (already a covariance; diag ~ const so ~ correlation).
    iu = np.triu_indices(prior.n_cells, k=1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].scatter(gdist[iu], Kgeo[iu], s=6, alpha=0.4)
    ax[0].set_xlabel("graph (geodesic) hop distance"); ax[0].set_ylabel("K_geo")
    ax[0].set_title("tracks geodesic distance")
    ax[1].scatter(edist[iu], Kgeo[iu], s=6, alpha=0.4, color="C1")
    ax[1].set_xlabel("Euclidean grid distance"); ax[1].set_ylabel("K_geo")
    ax[1].set_title("violates Euclidean monotonicity\n(wall-separated pairs)")
    fig.suptitle(f"Correlation vs distance — {tag}")
    p = os.path.join(args.out_dir, f"corr_vs_dist_{tag}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[corr] wrote {p}")

    # --- 4. Cell-mapping overlay (optional) ------------------------------
    if args.meas is not None:
        from optbnn.utils.util import load_measurement_data
        x_meas, aux_meas = load_measurement_data(args.meas)
        n = min(5000, x_meas.shape[0])
        sel = np.random.choice(x_meas.shape[0], n, replace=False)
        X = torch.from_numpy(x_meas[sel]).float()
        idx = prior.cell_of(X)
        counts = np.bincount(idx, minlength=prior.n_cells)
        grid = _cell_grid(prior, counts.astype(float))
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(grid, cmap="viridis", origin="upper")
        ax.set_title(f"measurement-point cell occupancy\n{tag} (n={n})")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
        p = os.path.join(args.out_dir, f"cellmap_{tag}.png")
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[cellmap] wrote {p}  — occupancy should fill reachable corridors, "
              "not walls.  If cells look shifted/rotated, the scaling/offset is off.")

    print("[done] Inspect the PNGs and iterate --eta until the heatmaps look right "
          "(correlation length ~2-4 cells).  Set eta from geometry, never reward.")


if __name__ == "__main__":
    main()
