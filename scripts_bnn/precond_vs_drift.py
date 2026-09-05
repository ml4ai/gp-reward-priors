#!/usr/bin/env python
"""Does the frozen preconditioner predict the centred `scale_z` failure?

Handoff HANDOFF_HP_SELECTION.md section 10.2 records that 16 of 17 round-3
rejections were the centred `scale_z` gate, "correlating with nothing swept
(max |rho| 0.399)".  `precond_minv_median` is NOT swept -- it is an emergent
property of where the 20k burn-in lands -- and it has been logged on the most
recent runs.  If `scale_z` tracks it where it tracks nothing swept, that is a
mechanism the search cannot see.

WHAT THIS CAN AND CANNOT TEST
-----------------------------
The wandb metric `precond_minv_median` is logged by the PARENT process only
(run_bnn_training_antmaze_eval.py, after the warm-up).  Chain workers run with
wandb disabled AND are spawned via mp.spawn, so their own `[precond] FROZEN`
lines never reach wandb's output.log either -- verified on four 32-chain runs,
each of which contains exactly one such line.

So this script tests the BETWEEN-RUN question: does the operating point the
preconditioner freezes at predict drift?  It does NOT test review section 9.4's
BETWEEN-CHAIN question -- whether chains within one run freeze different
preconditioners and therefore sample at different effective steps.  That needs
`chainspread_precond_*`, which is new instrumentation (section 4.3.75) present
only on runs from 2026-09-04 onward.

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/precond_vs_drift.py
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/precond_vs_drift.py --limit 400
"""

import argparse
import sys

import numpy as np

ENTITY = "champlin-university-of-arizona"
PROJECT = "BNN-training"

PRECOND = ["precond_minv_median", "precond_minv_max", "precond_v_hat_median",
           "precond_v_hat_at_floor", "precond_tau_median",
           "precond_tau_over_burnin"]
SWEPT = ["width", "depth", "sghmc_lr", "sghmc_lr_max", "mdecay",
         "cycle_length", "n_meas", "map_amp2", "num_chains", "num_samples"]
TARGET = "val_fn_drift_centred_scale_z_median"
EXTRA = ["val_fn_drift_centred_loc_z_median",
         "val_fn_drift_centred_scale_ratio_median", "val_cvar_ce"]


def spearman(x, y):
    """Rank correlation, computed without scipy.

    Rank, not Pearson: `scale_z` spans 0.0 to 848.9 across these runs, and a
    single outlier would otherwise set the answer.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), int(x.size)

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, float)
        r[order] = np.arange(v.size, dtype=float)
        # average ties, so a clamp-saturated column (many exact 100.0) does
        # not get an arbitrary ordering imposed on it
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(cnt.size)
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    return (float(np.dot(rx, ry) / d) if d > 0 else float("nan")), int(x.size)


def fetch(limit):
    import wandb
    api = wandb.Api(timeout=60)
    rows = []
    for i, r in enumerate(api.runs(f"{ENTITY}/{PROJECT}",
                                   order="-created_at", per_page=100)):
        if i >= limit:
            break
        s = dict(r.summary)
        if s.get("precond_minv_median") is None:
            continue
        # Sweep trials carry config_path; the ad-hoc diagnostic runs of
        # sections 4.3.x were launched without one, so fall back to the data
        # path, which is derived from the variant either way (section 1).
        cp = (str(r.config.get("config_path", "")) + " "
              + str(r.config.get("train_dataset", "")) + " "
              + str(r.config.get("dataset_id", "")))
        cp = cp.replace("-", "_")
        variant = "?"
        for v in ("medium_play", "medium_diverse", "large_play",
                  "large_diverse"):
            if v in cp:
                variant = v
                break
        # A sweep trial is one of a designed sample; an ad-hoc diagnostic run
        # is not.  Pooling them is what would make a correlation here
        # uninterpretable, so the distinction is carried through.
        row_kind = "sweep" if r.config.get("config_path") else "adhoc"
        row = {"id": r.id, "created": str(r.created_at)[:16],
               "variant": variant, "kind": row_kind}
        for k in PRECOND + [TARGET] + EXTRA:
            row[k] = s.get(k, float("nan"))
        for k in SWEPT:
            row[k] = r.config.get(k, float("nan"))
        rows.append(row)
    return rows


def col(rows, k):
    out = []
    for r in rows:
        v = r.get(k, float("nan"))
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.asarray(out, float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("WHAT")[0].strip())
    ap.add_argument("--limit", type=int, default=400,
                    help="how many recent runs to scan")
    ap.add_argument("--min-n", type=int, default=8,
                    help="minimum runs before a per-variant block is shown")
    args = ap.parse_args()

    rows = fetch(args.limit)
    if not rows:
        print("No runs with precond_minv_median found.")
        return 1
    print(f"{len(rows)} runs with a logged preconditioner "
          f"({rows[-1]['created']} .. {rows[0]['created']})")
    vs, cs = np.unique([r["variant"] for r in rows], return_counts=True)
    print("by variant: " + ", ".join(f"{v} {c}" for v, c in zip(vs, cs)))
    ks, kc = np.unique([r["kind"] for r in rows], return_counts=True)
    print("by kind:    " + ", ".join(f"{k} {c}" for k, c in zip(ks, kc))
          + "   (adhoc = section 4.3.x diagnostic runs, NOT a designed sample)")
    print()

    # ---- 1. Clamp saturation (section 4.3.68) ---------------------------
    mv = col(rows, "precond_minv_median")
    mx = col(rows, "precond_minv_max")
    fl = col(rows, "precond_v_hat_at_floor")
    print("1. CLAMP SATURATION -- minv is capped at 1/sqrt(v_hat_min); the "
          "default\n   v_hat_min=1e-4 caps it at exactly 100 (section 4.3.71).")
    print(f"   precond_minv_max  == its cap in {np.mean(np.isclose(mx, 100.0)):.0%} "
          f"of runs (values seen: {sorted(set(np.round(mx[np.isfinite(mx)], 2)))[:6]})")
    print(f"   precond_minv_MEDIAN at the cap (>=99.9) in "
          f"{np.mean(mv >= 99.9):.0%} of runs -- i.e. over half of ALL elements "
          f"pinned")
    print(f"   v_hat_at_floor: median {np.nanmedian(fl):.1%}, "
          f"range {np.nanmin(fl):.1%}-{np.nanmax(fl):.1%}")
    print()

    # ---- 2. Correlations with the gate that does the rejecting ----------
    y = col(rows, TARGET)
    print(f"2. RANK CORRELATION with {TARGET}")
    print("   Section 10.2: this gate 'correlates with nothing swept "
          "(max |rho| 0.399)'.")
    print()
    print(f"   {'quantity':>28} {'rho':>7} {'n':>5}   {'swept?':>7}")
    print("   " + "-" * 52)
    best_swept = 0.0
    for k in PRECOND + SWEPT:
        rho, n = spearman(col(rows, k), y)
        if not np.isfinite(rho):
            continue
        is_swept = k in SWEPT
        if is_swept:
            best_swept = max(best_swept, abs(rho))
        print(f"   {k:>28} {rho:+7.3f} {n:>5}   {'swept' if is_swept else '-':>7}")
    print()
    best_pre = max(
        (abs(spearman(col(rows, k), y)[0]) for k in PRECOND
         if np.isfinite(spearman(col(rows, k), y)[0])), default=0.0)
    print(f"   strongest |rho|:  preconditioner {best_pre:.3f}   "
          f"swept {best_swept:.3f}")
    if best_pre <= best_swept:
        print("   -> The preconditioner does NOT out-predict the swept "
              "dimensions.  It is\n      not the missing variable, on this "
              "evidence.")
    else:
        print("   -> The preconditioner out-predicts everything swept.  That "
              "is a variable\n      the search cannot see driving the gate "
              "that does the rejecting.")
    print()

    # ---- 3. Per variant, because the variants differ by 60x -------------
    print("3. PER VARIANT (the variants span 60x in steps/indep sample, so a "
          "pooled\n   correlation can be driven entirely by between-variant "
          "differences)")
    print()
    for v in vs:
        sub = [r for r in rows if r["variant"] == v]
        if len(sub) < args.min_n:
            print(f"   {v}: n={len(sub)}, below --min-n, skipped")
            continue
        ys = col(sub, TARGET)
        print(f"   {v} (n={len(sub)}, "
              f"scale_z median {np.nanmedian(ys):.2f}, "
              f"{np.mean(ys > 2.0):.0%} over 2.0)")
        for k in ("precond_minv_median", "precond_v_hat_at_floor",
                  "sghmc_lr", "sghmc_lr_max", "mdecay", "depth", "width"):
            rho, n = spearman(col(sub, k), ys)
            if np.isfinite(rho):
                print(f"      {k:>26} rho {rho:+.3f} (n={n})")
        print()

    # ---- 4. Is the objective pulling toward configs the gate rejects? ----
    # Both val_cvar_ce and scale_z are MINIMISED, so a dimension with the same
    # sign against both is bad for both and the optimiser walks away from it
    # on its own; OPPOSITE signs would be a genuine trade-off the search
    # cannot win, which is the failure mode 4.3.14 found on the old objective.
    sweep = [r for r in rows if r["kind"] == "sweep"]
    print("4. OBJECTIVE vs GATE (designed sample only -- the ad-hoc runs of "
          "4.3.x are\n   not a sample from the search space).  Both are "
          "minimised, so SAME sign =\n   bad for both, OPPOSITE sign = a "
          "trade-off the search cannot win.")
    print()
    print(f"   {'variant':>15} {'n':>3} {'dim':>13} {'rho vs cvar_ce':>15} "
          f"{'rho vs scale_z':>15}")
    print("   " + "-" * 66)
    for v in vs:
        sub = [r for r in sweep if r["variant"] == v]
        if len(sub) < 5:
            continue
        ce, sz = col(sub, "val_cvar_ce"), col(sub, TARGET)
        for k in ("depth", "width", "sghmc_lr", "sghmc_lr_max", "mdecay"):
            x = col(sub, k)
            r1, _ = spearman(x, ce)
            r2, _ = spearman(x, sz)
            if np.isfinite(r1) and np.isfinite(r2):
                flag = "  <-- TRADE-OFF" if (r1 < -0.3 and r2 > 0.3) else ""
                print(f"   {v:>15} {len(sub):>3} {k:>13} {r1:>+15.3f} "
                      f"{r2:>+15.3f}{flag}")
    print()

    # ---- 5. Does eligibility improve as the optimiser converges? ---------
    # Section 10.2 relaunch criterion 1 asks for either a change that raises
    # the eligible fraction to >= 60%, or an accepted account of why ~36% is
    # intrinsic.  "It is an early-exploration artefact that will resolve
    # itself" is the cheapest such account, and it is testable for free.
    sweep.sort(key=lambda r: r["created"])
    if len(sweep) >= 8:
        y = col(sweep, TARGET)
        d = col(sweep, "depth")
        idx = np.arange(1, len(sweep) + 1, dtype=float)
        print("5. DOES ELIGIBILITY IMPROVE AS THE SEARCH CONVERGES? "
              "(section 10.2 criterion 1)")
        print(f"   rho(trial order, scale_z) = {spearman(idx, y)[0]:+.3f}   "
              f"rho(trial order, depth) = {spearman(idx, d)[0]:+.3f}")
        h = len(sweep) // 2
        for lbl, sl in (("first half", slice(0, h)),
                        ("second half", slice(h, None))):
            yy = y[sl][np.isfinite(y[sl])]
            dd = d[sl][np.isfinite(d[sl])]
            print(f"   {lbl:>12}: {np.mean(yy <= 2.0):.0%} pass "
                  f"(n={yy.size}), median depth {np.median(dd):.1f}")
        print("   If the optimiser moves off a dimension that predicts "
              "failure and the pass\n   rate does NOT follow, the exploration "
              "account is refuted and the residual\n   failures are driven by "
              "something not in the search space.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
