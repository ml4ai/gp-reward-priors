#!/usr/bin/env python
"""MR / PT stage-1 sweeps: does capacity matter, and do the models memorise?  (handoff 4.3.106)

Zero compute: reads the completed MR and PT stage-1 sweep trials.  Per variant,
rank-correlates approximate capacity with eval_loss_best (the selection
objective: minimum validation loss over 5000 epochs), test_loss (the seed-0 test
split, evaluated once on the best checkpoint), best_epoch and the final
eval-train gap; lists the top-5 trials; then tabulates final training loss by
capacity quartile.

Capacity is approximate: MR = MLP parameter count with d_in 37; PT = num_layers
x 12 x embd^2.  Rank use only.  Sweep trials are the Bayes optimiser's choices,
not a designed sample.

CAUTION (handoff 1): the seed-0 test split "is not used for selection".  The
test_loss columns here are diagnostic; using them to change a range requires
amending section 1 and disclosing it.

Usage:
    /opt/anaconda3/envs/irl/bin/python check_capacity_mrpt.py
"""
import collections
import math

import numpy as np
import wandb

api = wandb.Api(timeout=90)
E = "champlin-university-of-arizona"
N = {"medium-play": 358, "medium-diverse": 498, "large-play": 254, "large-diverse": 514}
def rk(a): return np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    return (float(np.corrcoef(rk(x[ok]), rk(y[ok]))[0, 1]) if ok.sum() > 3 else float("nan")), int(ok.sum())
def mlp_params(w, d, din=37): W = 2 ** w; return W * (din + 1) + (d - 1) * W * (W + 1) + W + 1
for proj in ("MR-training", "PT-training"):
    runs = [r for r in api.runs(f"{E}/{proj}", per_page=300) if r.sweep and r.state == "finished"]
    print(f"\n{'=' * 100}\n{proj}: {len(runs)} finished sweep trials")
    grp = collections.defaultdict(list)
    for r in runs:
        c, s = r.config, dict(r.summary)
        v = str(c.get("antmaze_variant", "")).replace("antmaze-", "").replace("-v2", "")
        if proj.startswith("MR"):
            cap = mlp_params(c["width"], c["depth"]); desc = f"w{c['width']} d{c['depth']}"
            a, b = c["width"], c["depth"]
        else:
            E_, L = 2 ** c["embd_dim"], c["num_layers"]
            cap = L * 12 * E_ * E_; desc = f"e{c['embd_dim']} h{c['head_dim']} L{L}"
            a, b = c["embd_dim"], L
        grp[v].append(dict(id=r.id, cap=cap, desc=desc, a=a, b=b, sweep=r.sweep.id,
                           best=s.get("eval_loss_best"), be=s.get("best_epoch"), tr=s.get("training_loss"),
                           ev=s.get("eval_loss"), te=s.get("test_loss"), lr=c.get("lr")))
    for v in ("medium-play", "medium-diverse", "large-play", "large-diverse"):
        g = [x for x in grp[v] if x["best"] is not None]
        if not g: continue
        g.sort(key=lambda x: x["best"])
        cap = [x["cap"] for x in g]; best = [x["best"] for x in g]
        be = [x["be"] if x["be"] is not None else np.nan for x in g]
        gap = [(x["ev"] - x["tr"]) if (x["ev"] is not None and x["tr"] is not None) else np.nan for x in g]
        te = [x["te"] if x["te"] is not None else np.nan for x in g]
        print(f"\n  {v}  (N_train {N[v]}, {len(g)} trials, sweep {g[0]['sweep']})")
        print(f"    rho(capacity, eval_loss_best) {sp(cap, best)[0]:+.3f}   rho(capacity, test_loss) {sp(cap, te)[0]:+.3f}"
              f"   rho(capacity, best_epoch) {sp(cap, be)[0]:+.3f}   rho(capacity, final eval-train gap) {sp(cap, gap)[0]:+.3f}")
        print(f"    rho(axisA, best) {sp([x['a'] for x in g], best)[0]:+.3f}  rho(axisB, best) {sp([x['b'] for x in g], best)[0]:+.3f}"
              f"   (MR: A=width B=depth; PT: A=embd_dim B=num_layers)")
        print(f"    best_epoch: median {np.nanmedian(be):.0f}, max {np.nanmax(be):.0f} of 5000;  trials with best_epoch < 500: {np.mean(np.array(be) < 500):.0%}")
        print(f"    top-5 by eval_loss_best:")
        for x in g[:5]:
            print(f"      {x['desc']:<12} cap~{x['cap']:>10,} ({x['cap']/N[v]:>6,.0f}/pair)  best {x['best']:.4f}  test {x['te'] if x['te'] is None else round(x['te'],4)}  "
                  f"best_epoch {x['be']}  final train {x['tr'] if x['tr'] is None else round(x['tr'],4)}  final eval {x['ev'] if x['ev'] is None else round(x['ev'],4)}  lr {x['lr']:.2e}")
        ca = np.array(cap)
        print(f"    capacity sampled: min {ca.min():,} / median {np.median(ca):,.0f} / max {ca.max():,};  winner's capacity percentile {np.mean(ca <= g[0]['cap']):.0%}")

print()
print('FINAL TRAINING LOSS BY CAPACITY QUARTILE')
api = wandb.Api(timeout=90)
for proj, capkey in (("MR-training", lambda c: (2**c["width"])*38 + (c["depth"]-1)*(2**c["width"])*((2**c["width"])+1) + 2**c["width"] + 1),
                     ("PT-training", lambda c: c["num_layers"]*12*(2**c["embd_dim"])**2)):
    rows = []
    for r in api.runs(f"champlin-university-of-arizona/{proj}", per_page=300):
        if not r.sweep or r.state != "finished": continue
        s = dict(r.summary); c = r.config
        if s.get("training_loss") is None: continue
        rows.append((capkey(c), s["training_loss"], s.get("training_acc"), c["antmaze_variant"]))
    rows.sort()
    caps = np.array([x[0] for x in rows]); tl = np.array([x[1] for x in rows]); ta = np.array([x[2] if x[2] is not None else np.nan for x in rows])
    q = np.quantile(caps, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"\n{proj}: {len(rows)} trials; final training_loss by capacity quartile")
    for lo, hi in zip(q[:-1], q[1:]):
        m = (caps >= lo) & (caps <= hi)
        print(f"  cap {lo:>10,.0f}-{hi:>10,.0f}: n={m.sum():>3}  final train loss <1e-3: {np.mean(tl[m] < 1e-3):>4.0%}  median {np.median(tl[m]):.4f}  median train acc {np.nanmedian(ta[m]):.3f}")
    small = [x for x in rows if x[0] == caps.min()]
    print(f"  smallest capacity ({caps.min():,}): n={len(small)}, final train loss <1e-3 in {np.mean([x[1] < 1e-3 for x in small]):.0%}")
