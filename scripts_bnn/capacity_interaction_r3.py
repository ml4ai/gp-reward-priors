#!/usr/bin/env python
"""Is there a width x depth interaction in drift / degeneracy?  (handoff 4.3.106)

Uses the 28 round-3 BNN sweep trials -- the only runs where width and depth
vary together (the capacity ladders hold depth fixed).  Fits, pooled within
variant (variant fixed effects),

    y ~ width(log2) + depth + width x depth

and tests the interaction by Freedman-Lane permutation within variant.

READ WITH THE DESIGN IN MIND: <= 5 distinct (depth, width) cells per variant,
one trial per cell, chosen by a Bayes optimiser, with sghmc_lr / mdecay /
fraction_cool varying across trials too.  A null here is absence of evidence,
not evidence of absence.

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/capacity_interaction_r3.py
"""
import math

import numpy as np
import wandb

api = wandb.Api(timeout=90)
rows = []
for r in api.runs("champlin-university-of-arizona/BNN-training", per_page=300):
    c = r.config
    if not r.sweep or c.get("n_meas") != 256 or r.state != "finished": continue
    s = dict(r.summary)
    ratio = s.get("val_fn_drift_centred_scale_ratio_median")
    rows.append(dict(v=c["antmaze_variant"].replace("antmaze-", "").replace("-v2", ""), w=int(c["width"]), d=int(c["depth"]),
        lr=c.get("sghmc_lr"), lrmax=c.get("sghmc_lr_max"), md=c.get("mdecay"), fc=c.get("fraction_cool"),
        logr=abs(math.log(ratio)) if ratio else np.nan, gap=s.get("val_cvar_degeneracy_gap"),
        margin=s.get("val_cvar_degeneracy_margin"), cvar=s.get("val_cvar_ce"), sd=s.get("val_pred_centred_sd_median")))
print(f"{len(rows)} round-3 sweep trials")
coll = [x for x in rows if x["cvar"] and abs(x["cvar"] - math.log(2)) < 1e-3]
print("collapsed to CE = log 2 (excluded):", [(x["v"], x["w"], x["d"]) for x in coll])
rows = [x for x in rows if x not in coll]
print("\nsampler settings also vary across trials (per-variant range):")
for v in sorted({x["v"] for x in rows}):
    g = [x for x in rows if x["v"] == v]
    print(f"  {v:<15} n={len(g)}  sghmc_lr {min(x['lr'] for x in g):.1e}-{max(x['lr'] for x in g):.1e}  "
          f"mdecay {min(x['md'] for x in g):.3f}-{max(x['md'] for x in g):.3f}  fraction_cool {min(x['fc'] for x in g):.2f}-{max(x['fc'] for x in g):.2f}")
    cells = sorted({(x["d"], x["w"]) for x in g})
    print(f"    (depth,width) cells: {cells}")

def design(g, keys):
    X = []
    for x in g:
        row = []
        for k in keys:
            if k == "w": row.append(x["w"])
            elif k == "d": row.append(x["d"])
            elif k == "wd": row.append(x["w"] * x["d"])
        X.append(row)
    return np.array(X, float)

def fit(rows, yk, transform=lambda y: y, perms=20000, seed=0):
    rng = np.random.default_rng(seed)
    vs = sorted({x["v"] for x in rows})
    g = [x for x in rows if x[yk] is not None and np.isfinite(x[yk])]
    # within-variant centring of y, w, d (variant fixed effects)
    y = np.array([transform(x[yk]) for x in g], float)
    W = np.array([x["w"] for x in g], float); D = np.array([x["d"] for x in g], float)
    vid = np.array([vs.index(x["v"]) for x in g])
    def centre(a):
        out = a.copy()
        for i in range(len(vs)): out[vid == i] -= a[vid == i].mean()
        return out
    Wc, Dc = centre(W), centre(D)
    INT = centre(Wc * Dc)
    X = np.column_stack([Wc, Dc, INT])
    yc = centre(y)
    beta = np.linalg.lstsq(X, yc, rcond=None)[0]
    resid = yc - X @ beta
    r2 = 1 - resid.var() / yc.var()
    X0 = np.column_stack([Wc, Dc])
    b0 = np.linalg.lstsq(X0, yc, rcond=None)[0]
    r2_0 = 1 - (yc - X0 @ b0).var() / yc.var()
    # Freedman-Lane permutation for the interaction: permute residuals of the additive model within variant
    res0 = yc - X0 @ b0
    hits = 0
    for _ in range(perms):
        rp = res0.copy()
        for i in range(len(vs)):
            idx = np.where(vid == i)[0]; rp[idx] = res0[rng.permutation(idx)]
        yp = X0 @ b0 + rp
        bp = np.linalg.lstsq(X, yp, rcond=None)[0]
        hits += abs(bp[2]) >= abs(beta[2]) - 1e-12
    return dict(n=len(g), b_w=beta[0], b_d=beta[1], b_wd=beta[2], r2_add=r2_0, r2_int=r2, p_int=hits / perms)

print("\nPOOLED WITHIN-VARIANT MODEL  y ~ width(log2) + depth + width x depth  (variant fixed effects; Freedman-Lane permutation p for the interaction)")
for yk, lab, tf in (("logr", "|log r| (drift)", lambda y: y), ("gap", "log degeneracy gap", lambda y: math.log(y)),
                    ("margin", "degeneracy margin", lambda y: y), ("cvar", "cvar_ce", lambda y: y)):
    f = fit(rows, yk, tf)
    print(f"  {lab:<20} n={f['n']}  b_width {f['b_w']:+.4f}  b_depth {f['b_d']:+.4f}  b_wxd {f['b_wd']:+.4f}   "
          f"R2 additive {f['r2_add']:.2f} -> with interaction {f['r2_int']:.2f}   p(interaction) {f['p_int']:.3f}")

print("\nCELL VIEW (each cell one trial; other sampler settings differ between cells):")
for v in sorted({x["v"] for x in rows}):
    g = sorted([x for x in rows if x["v"] == v], key=lambda x: (x["d"], x["w"]))
    print(f"  {v}: " + "; ".join(f"d{x['d']}w{x['w']} r={x['logr']:.3f} gap={x['gap']:.3f}" for x in g))
