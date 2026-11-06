#!/usr/bin/env python
"""calibration_mechanism.py -- why does the BNN's H3 calibration gate fail?  (handoff §14.92)

SCOPE: the author's scoped exception to the BNN/CVaR hold (§14.92, 2026-11-06): read-only
access to the BNN reward models' calibration diagnostics, for the mechanism only.  Reads
`calibration_results/measurements/` (the anchor's per-pair arrays, §13.10); touches no IQL
result.  Output belongs in `calibration_results/held_bnn_cvar/`; nothing here is quoted in the
benchmarking chapter.  DESCRIPTIVE and post hoc: the registered gate (§13.10.1) is not re-read
and no verdict changes.

STAGE A (desk, no GPU).  Per variant, the anchor's ten models, from the per-pair arrays
(y1, p_pred, spread, d_mean):

  1. WHERE THE SOFTNESS COMES FROM.  §13.10.1 found the posterior predictive UNDER-confident.
     The predictive averages sigma over draws, so it is softer than the plug-in sigma(d_mean)
     by an amount set by the spread (probit approximation: sigma(d / sqrt(1 + pi s^2 / 8))).
     ECE of the predictive, the plug-in, and the probit approximation (a check that the
     arrays agree).  If the plug-in is calibrated and the predictive is not, the posterior
     is too WIDE; if the plug-in is under-confident too, the mean logit itself is too SMALL.
  2. ONE TEMPERATURE.  sigma(z / T) with one T per variant, fitted by maximum likelihood
     leave-one-seed-out (fit on nine seeds, scored on the tenth), for z = d_mean and for
     z = logit(p_pred).  T < 1: the logits are too small by a single global factor.  Reported:
     T, and held-out CE / ECE before and after.
  3. WHAT THE SPREAD IS MADE OF.  Per model, spread^2 = a + b * d_mean^2 (least squares,
     a, b >= 0): a part proportional to the margin (a global scale uncertainty: draws that
     differ mainly by an overall factor give spread = c |d|) and a floor a.  Reported: R^2,
     the share of mean spread^2 in the scale term, and the error-AUROC of the excess spread
     (spread^2 - b d^2) -- the part a scale mode does not explain.
  4. THE LABELS.  The fraction of validation labels that are not 0 or 1.

    /opt/anaconda3/envs/irl/bin/python scripts_bnn/calibration_mechanism.py \\
        > calibration_results/held_bnn_cvar/mechanism_stage_a.txt
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/calibration_mechanism.py --selftest
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import calibration_gate as G  # noqa: E402

MEAS = os.path.join(ROOT, "calibration_results", "measurements")
EPS = 1e-12


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def ce(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def probit_pred(d, s):
    return sig(np.asarray(d, float) / np.sqrt(1.0 + np.pi * np.asarray(s, float) ** 2 / 8.0))


def fit_temperature(z, y, grid=np.exp(np.linspace(np.log(0.05), np.log(20.0), 2001))):
    """The T maximising the likelihood of y under sigma(z / T) (a fine log grid: robust, exact
    enough for a diagnostic)."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    losses = [ce(sig(z / t), y) for t in grid]
    return float(grid[int(np.argmin(losses))])


def loso_temperature(recs, key):
    """Leave-one-seed-out: for each seed, T fitted on the other seeds' pairs, applied to it.
    -> (median T, held-out CE before, after, held-out ECE before, after)."""
    zs = {s: (np.asarray(r["d_mean"]) if key == "plug" else logit(r["p_pred"])) for s, r in recs.items()}
    ys = {s: np.asarray(r["y1"], float) for s, r in recs.items()}
    ts, pb, pa, yy = [], [], [], []
    for s in recs:
        others = [o for o in recs if o != s]
        t = fit_temperature(np.concatenate([zs[o] for o in others]), np.concatenate([ys[o] for o in others]))
        ts.append(t)
        pb.append(sig(zs[s]))
        pa.append(sig(zs[s] / t))
        yy.append(ys[s])
    pb, pa, yy = np.concatenate(pb), np.concatenate(pa), np.concatenate(yy)
    return float(np.median(ts)), ce(pb, yy), ce(pa, yy), G.ece(pb, yy)[0], G.ece(pa, yy)[0]


def spread_split(d, s):
    """spread^2 = a + b d^2, a, b >= 0.  -> (a, b, R^2, share of mean spread^2 in b d^2)."""
    d2, s2 = np.asarray(d, float) ** 2, np.asarray(s, float) ** 2
    X = np.stack([np.ones_like(d2), d2], 1)
    a, b = np.linalg.lstsq(X, s2, rcond=None)[0]
    if b < 0:
        a, b = s2.mean(), 0.0
    elif a < 0:
        a, b = 0.0, float((d2 @ s2) / (d2 @ d2))
    fit = a + b * d2
    r2 = 1.0 - float(((s2 - fit) ** 2).sum() / ((s2 - s2.mean()) ** 2).sum()) if s2.var() > 0 else 0.0
    share = float((b * d2).mean() / s2.mean()) if s2.mean() > 0 else 0.0
    return float(a), float(b), r2, share


def analyse(recs):
    """recs: {seed: record}.  -> dict of the four tests, pooled over the variant's seeds."""
    y = np.concatenate([np.asarray(r["y1"], float) for r in recs.values()])
    p = np.concatenate([np.asarray(r["p_pred"], float) for r in recs.values()])
    d = np.concatenate([np.asarray(r["d_mean"], float) for r in recs.values()])
    s = np.concatenate([np.asarray(r["spread"], float) for r in recs.values()])
    out = {"n": int(y.size), "labels_not_01": float(np.mean((y > 1e-9) & (y < 1 - 1e-9)))}
    out["ece_pred"], out["ece_plug"], out["ece_probit"] = (G.ece(p, y)[0], G.ece(sig(d), y)[0],
                                                           G.ece(probit_pred(d, s), y)[0])
    out["probit_vs_pred"] = float(np.abs(probit_pred(d, s) - p).max())
    out["soft_ratio"] = float(np.median(np.abs(logit(p)) / np.maximum(np.abs(d), EPS)))   # < 1: averaging softens
    out["T_plug"] = loso_temperature(recs, "plug")
    out["T_pred"] = loso_temperature(recs, "pred")
    rows = []
    for r in recs.values():
        dd, ss = np.asarray(r["d_mean"], float), np.asarray(r["spread"], float)
        wrong = (np.asarray(r["p_pred"]) > 0.5) != (np.asarray(r["y1"]) > 0.5)
        a, b, r2, share = spread_split(dd, ss)
        excess = ss ** 2 - b * dd ** 2
        rows.append((r2, share, G.auroc(excess, wrong) if 0 < wrong.sum() < wrong.size else np.nan,
                     G.auroc(ss, wrong) if 0 < wrong.sum() < wrong.size else np.nan))
    rows = np.asarray(rows, float)
    out["split_r2"], out["split_share"] = float(np.median(rows[:, 0])), float(np.median(rows[:, 1]))
    out["auroc_excess"], out["auroc_spread"] = float(np.nanmean(rows[:, 2])), float(np.nanmean(rows[:, 3]))
    return out


def report(name, o, out=print):
    out(f"\n=== {name}: {o['n']} validation pairs over the ten seeds; labels not 0/1: {100 * o['labels_not_01']:.1f}%")
    out(f"  1. ECE  predictive {o['ece_pred']:.3f}   plug-in sigma(d_mean) {o['ece_plug']:.3f}   "
        f"probit(d, spread) {o['ece_probit']:.3f}   (probit vs predictive, max |dp| {o['probit_vs_pred']:.3f}); "
        f"median |logit p_pred| / |d_mean| {o['soft_ratio']:.2f}")
    for k, lab in (("T_plug", "plug-in d_mean"), ("T_pred", "logit(p_pred)")):
        t, cb, ca, eb, ea = o[k]
        out(f"  2. one temperature on {lab:15s}: T = {t:.3f} (median over the ten leave-one-seed-out fits); "
            f"held-out CE {cb:.3f} -> {ca:.3f}, ECE {eb:.3f} -> {ea:.3f}")
    out(f"  3. spread^2 = a + b d^2: median R^2 {o['split_r2']:.2f}, median share of spread^2 in b d^2 "
        f"{o['split_share']:.2f}; error-AUROC of the spread {o['auroc_spread']:.3f}, of the excess spread "
        f"(spread^2 - b d^2) {o['auroc_excess']:.3f}")


def run(meas=MEAS, out=print):
    out("BNN H3 CALIBRATION, MECHANISM STAGE A (handoff §14.92; scoped hold exception; post hoc, descriptive)")
    for v in G.E.VARIANTS:
        recs = {}
        for s in G.SEEDS:
            path = os.path.join(meas, f"antmaze_{v}_bnn_eval_{s}.json")
            if os.path.exists(path):
                import json
                recs[s] = json.load(open(path))
        if len(recs) != len(G.SEEDS):
            out(f"\n=== {v}: {len(recs)} of {len(G.SEEDS)} measurements -- not read")
            continue
        report(v, analyse(recs), out)
    return 0


def selftest():
    rng = np.random.default_rng(0)
    # a model whose logits are too small by a factor 3 is under-confident; T recovers ~1/3
    recs = {}
    for s in range(1, 11):
        true = rng.normal(0, 3, 400)
        y = (rng.random(400) < sig(true)).astype(float)
        d = true / 3.0
        sp = 0.4 * np.abs(d) + 0.05                       # spread mostly proportional to the margin
        recs[s] = {"y1": y.tolist(), "d_mean": d.tolist(), "spread": sp.tolist(),
                   "p_pred": probit_pred(d, sp).tolist()}
    o = analyse(recs)
    assert 0.25 < o["T_plug"][0] < 0.45, o["T_plug"]
    assert o["T_plug"][2] < o["T_plug"][1] and o["T_plug"][4] < o["T_plug"][3]   # held-out CE and ECE improve
    assert o["ece_plug"] > 0.05 and o["ece_pred"] >= o["ece_plug"] - 1e-9      # the averaging softens further
    assert o["probit_vs_pred"] < 1e-9 and o["soft_ratio"] < 1
    assert o["split_share"] > 0.8 and o["split_r2"] > 0.9                        # the spread is a scale mode
    assert o["labels_not_01"] == 0.0
    # a calibrated model: T ~ 1
    for s in recs:
        true = rng.normal(0, 3, 400)
        recs[s]["y1"] = (rng.random(400) < sig(true)).astype(float).tolist()
        recs[s]["d_mean"] = true.tolist()
        recs[s]["p_pred"] = sig(true).tolist()
        recs[s]["spread"] = (0.01 * np.ones(400)).tolist()
    o = analyse(recs)
    assert 0.85 < o["T_plug"][0] < 1.2, o["T_plug"]
    a, b, r2, share = spread_split(np.array([1.0, 2.0, 3.0]), np.sqrt(np.array([1.0, 1.0, 1.0])))
    assert abs(b) < 1e-9 and abs(a - 1.0) < 1e-9 and abs(share) < 1e-9, (a, b, share)   # constant spread: no scale part
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--meas", default=MEAS)
    a = ap.parse_args()
    return selftest() if a.selftest else run(a.meas)


if __name__ == "__main__":
    sys.exit(main())
