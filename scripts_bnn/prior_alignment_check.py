"""Does stratified measurement sampling (4.3.109) require recalibrating map_amp2
or any other pinned parameter?  Measured, not argued.  No GPU, no box access --
it reads the maze layout and the real 999,000-point measurement pools.

Produces the tables in 4.3.109's "Does stratification need map_amp2 -- or
anything else -- recalibrated?".  Answers:

  Q1  does the DERIVED map_amp2 (= T^2 / marginal-variance multiplier) move?
      No: +0.18% (medium) / -0.05% (large).  Leave it pinned.
  Q2  how much does effective PRIOR STRENGTH change, and can map_amp2
      compensate?  It changes a lot and in two opposing ways, and map_amp2
      CANNOT compensate: it rescales K^-1 uniformly, while stratification
      changes the SHAPE of the spectrum.
  Q3  what does the n_meas=26 control actually hold fixed?  Point count and
      most of the de-duplication -- but it reaches only ~13.7 of 26 cells, so
      what it really isolates is COVERAGE.

Caveat: run locally this uses the HARDCODED layout (get_antmaze_layout falls
back when `gym` is absent).  Free-cell counts match the live env (26 / 33) but
the multiplier comes out ~3% from 4.3.55's.  The random-vs-stratified shift is
computed from the same diag(K_geo) on both sides and is unaffected.

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/prior_alignment_check.py
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from optbnn.gp.maze_layouts import get_antmaze_layout
from optbnn.gp.models.map_informed_prior import MapInformedGPPrior

T = 100.0
rng = np.random.default_rng(0)

POOL = {
    "medium": "data/antmaze/antmaze-medium-play-v2/antmaze-medium-play-v2_tuning_set.hdf5",
    "large": "data/antmaze/antmaze-large-play-v2/antmaze-large-play-v2_tuning_set.hdf5",
}
AMP2 = {"medium": 6626.0, "large": 6611.0}


def load_pool(path):
    import h5py
    with h5py.File(os.path.join(ROOT, path), "r") as f:
        keys = list(f.keys())
        for k in ("observations", "obs", "x_meas"):
            if k in f:
                return np.asarray(f[k]), keys
        raise KeyError(keys)


for size in ("medium", "large"):
    print("=" * 92)
    print(f"{size.upper()}")
    print("=" * 92)
    free_mask, scaling, offset = get_antmaze_layout(size)
    ncell = int(free_mask.sum())
    print(f"layout: {free_mask.shape} grid, {ncell} free cells, "
          f"scaling={scaling}, offset={offset}")

    prior = MapInformedGPPrior(
        free_mask=free_mask, scaling=scaling, offset=offset,
        eta=1.0, sig_c2=1.0, sig_g2=1.0, sig_n2=0.001,
        amp2=AMP2[size], xy_source="obs", device="cpu",
    )

    X, keys = load_pool(POOL[size])
    print(f"pool: {X.shape} (hdf5 keys {keys})")
    Xt = torch.as_tensor(X, dtype=torch.float64)
    idx_pool = np.asarray(prior.cell_of(Xt, None)).ravel()
    occ, counts = np.unique(idx_pool, return_counts=True)
    frac = counts / counts.sum()
    order = np.argsort(-frac)
    print(f"occupied cells: {len(occ)} of {ncell};  "
          f"top cell {100*frac[order[0]]:.1f}%, top 5 {100*frac[order[:5]].sum():.1f}%")

    # ---- Q1: the marginal-variance multiplier, hence the derived map_amp2 ----
    # K[i,i]/amp2 = sig_c2 + sig_g2*Kgeo[c_i,c_i] + sig_n2
    Kgeo = prior._Kgeo.cpu().numpy() if hasattr(prior, "_Kgeo") else None
    if Kgeo is None:
        for attr in ("K_geo", "_K_geo", "Kgeo"):
            if hasattr(prior, attr):
                Kgeo = np.asarray(getattr(prior, attr))
                break
    dg = np.diag(Kgeo)
    mult_cell = 1.0 + 1.0 * dg + 0.001                      # per cell
    mult_pool = float((mult_cell[idx_pool]).mean())          # random draw weighting
    mult_strat = float(mult_cell[occ].mean())                # one point per occupied cell
    print(f"\nQ1  diag(K_geo) per cell: min {dg.min():.4f} max {dg.max():.4f} "
          f"spread {dg.max()/dg.min():.3f}x")
    print(f"    multiplier  pool-weighted {mult_pool:.4f}   "
          f"uniform-over-occupied-cells {mult_strat:.4f}   "
          f"ratio {mult_strat/mult_pool:.4f}")
    print(f"    derived map_amp2 = T^2/multiplier: "
          f"random {T**2/mult_pool:8.1f}   stratified {T**2/mult_strat:8.1f}   "
          f"shift {100*(mult_pool/mult_strat - 1):+.2f}%")
    print(f"    (pinned value in the config: {AMP2[size]:.0f})")

    # ---- Q2: effective prior strength ----
    # Split f into a cell-mean part and a within-cell part; measure the prior
    # force K^-1 f on each.  w = within-cell sd of f as a fraction of the
    # across-cell sd.
    def gram(idx):
        return prior._gram_from_idx(torch.as_tensor(np.asarray(idx))).cpu().numpy()

    draw256 = rng.choice(len(idx_pool), 256, replace=False)
    idx256 = idx_pool[draw256]
    draw26 = rng.choice(len(idx_pool), ncell, replace=False)
    idx26 = idx_pool[draw26]
    idx_strat = occ.copy()

    modes = {
        f"random n_meas=256": idx256,
        f"random n_meas={ncell} (CONTROL)": idx26,
        f"stratified ({len(occ)} cells)": idx_strat,
    }
    print(f"\nQ2  prior force  ||K^-1 f|| / ||f||, by component of f")
    print(f"    {'mode':32s} {'n':>4s} {'cells':>6s} {'dup%':>5s} {'cond(K)':>10s} "
          f"{'at nugget':>10s} {'cellmean':>9s} {'withincell':>11s}")
    for label, idx in modes.items():
        K = gram(idx)
        n = len(idx)
        ev = np.linalg.eigvalsh(K)
        nug = int((ev < 1.5 * 0.001 * AMP2[size]).sum())
        u, c = np.unique(idx, return_counts=True)
        dup = 100 * (n - len(u)) / n
        Kinv = np.linalg.inv(K)

        # cell-mean component: f constant within each cell, random across cells
        f_cell = rng.standard_normal(len(u))
        m = {cc: f_cell[j] for j, cc in enumerate(u)}
        f_mean = np.array([m[i] for i in idx])
        # within-cell component: zero cell-mean, unit sd within cells
        f_w = rng.standard_normal(n)
        for cc in u:
            sel = idx == cc
            if sel.sum() > 1:
                f_w[sel] -= f_w[sel].mean()
            else:
                f_w[sel] = 0.0
        def force(f):
            nf = np.linalg.norm(f)
            return np.linalg.norm(Kinv @ f) / nf if nf > 1e-12 else float("nan")
        print(f"    {label:32s} {n:4d} {len(u):6d} {dup:5.1f} {np.linalg.cond(K):10.3g} "
              f"{nug:10d} {force(f_mean):9.4g} {force(f_w):11.4g}")

    print(f"\n    map_amp2 multiplies K, so K^-1 -> K^-1/amp2: it rescales BOTH")
    print(f"    columns by the SAME factor.  It cannot change their RATIO.")

    # ---- Q3: what the control holds fixed ----
    ncells_26 = [len(np.unique(idx_pool[rng.choice(len(idx_pool), ncell,
                                                   replace=False)]))
                 for _ in range(2000)]
    ncells_256 = [len(np.unique(idx_pool[rng.choice(len(idx_pool), 256,
                                                    replace=False)]))
                  for _ in range(2000)]
    print(f"\nQ3  distinct cells hit by a random draw (2000 reps)")
    print(f"    n_meas=256 : {np.mean(ncells_256):.1f} of {len(occ)} occupied")
    print(f"    n_meas={ncell:<3d}: {np.mean(ncells_26):.1f} of {len(occ)} occupied "
          f"<- the CONTROL arm")
    print(f"    stratified : {len(occ)} of {len(occ)} by construction")
