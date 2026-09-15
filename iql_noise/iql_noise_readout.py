#!/usr/bin/env python
"""Experiment 1 readout: IQL run-to-run noise on the stage-4 statistic (handoff 4.3.107).

Applies the readout pre-registered in 4.3.107 before any run:

  * statistic: max over the 200 evaluation points of the 100-episode mean
    (stage 4's selection statistic and the reported statistic); secondary:
    the final point and the mean of the last 10 points
  * sigma: pooled within-arm SD (10 df at n = 6 per arm), 90% CI
  * delta: mean(idx 3) - mean(idx 2), 95% CI
  * P(correct pick) = Phi(|delta| / (sigma * sqrt 2)): the chance one run per
    index picks the higher-mean arm (plug-in)
  * DECISION: sigma >= 0.028 (sigma * sqrt 2 >= the 0.04 median top-2 gap)
    means single-run stage 4 cannot resolve typical gaps

Arms are the 10 replicate runs (wandb group iql-noise-large-play-mr-best,
seeds 100..500) plus the original seed-0 stage-4 runs (sc0xntv4 idx 2,
yx2uryjf idx 3).  Every run's config is audited against its arm's original:
only seed, name, group and reward_model_root may differ, and all must resolve
to the same reward_model_path.  Safe to run while runs are in flight: it
reports state and evaluation-point counts and only applies the decision when
all 12 runs are finished with 200 points.

Usage:
    /opt/anaconda3/envs/irl/bin/python iql_noise/iql_noise_readout.py
"""

import math
import sys
import time

import numpy as np
from scipy import stats

ENTITY, PROJECT = "champlin-university-of-arizona", "IQL-pref"
GROUP = "iql-noise-large-play-mr-best"
ORIGINALS = {2: "sc0xntv4", 3: "yx2uryjf"}
SEEDS = (100, 200, 300, 400, 500)
N_EVALS = 200
THRESHOLD = 0.028
MEDIAN_GAP = 0.04
EXPECTED_DIFF = {"seed", "name", "group", "reward_model_root"}
STAGE4_COMMIT = "8c3b0dbd545acb36469efac2d2776cbdcd486df3"
CURRENT_COMMIT = "1d469ea462a04ec557be9ab2a9e8cb6498951ef8"


def history(run, tries=4):
    for i in range(tries):
        try:
            h = run.history(keys=["mean_score"], samples=1000, pandas=False)
            return [x["mean_score"] for x in h if x.get("mean_score") is not None]
        except Exception:
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"could not fetch history for {run.id}")


def pooled(arms):
    ss = sum(((np.asarray(v) - np.mean(v)) ** 2).sum() for v in arms)
    df = sum(len(v) - 1 for v in arms)
    return math.sqrt(ss / df), df


def main():
    import wandb
    api = wandb.Api(timeout=300)
    runs = {i: [api.run(f"{ENTITY}/{PROJECT}/{rid}")] for i, rid in ORIGINALS.items()}
    for r in api.runs(f"{ENTITY}/{PROJECT}", filters={"config.group": GROUP}, per_page=50):
        runs.setdefault(r.config.get("normalize_reward"), []).append(r)

    print("=" * 96)
    print("RUNS AND CONFIG AUDIT (vs the arm's original stage-4 run)")
    print("=" * 96)
    ok, complete = True, True
    rows = {}
    for idx in sorted(runs):
        if idx not in ORIGINALS:
            print(f"  UNEXPECTED normalize_reward {idx}: {[r.id for r in runs[idx]]}")
            ok = False
            continue
        orig = runs[idx][0]
        oc = orig.config
        seeds = sorted(r.config.get("seed") for r in runs[idx][1:])
        if seeds != list(SEEDS):
            print(f"  idx {idx}: replicate seeds {seeds}, expected {list(SEEDS)}")
            ok = False
        rows[idx] = []
        for r in sorted(runs[idx], key=lambda r: r.config.get("seed")):
            c = r.config
            diffs = sorted(k for k in set(c) | set(oc)
                           if repr(c.get(k)) != repr(oc.get(k)) and k not in EXPECTED_DIFF)
            commit = ((r.metadata or {}).get("git") or {}).get("commit")
            want = STAGE4_COMMIT if r.id == orig.id else CURRENT_COMMIT
            h = history(r)
            fin = r.state == "finished" and len(h) == N_EVALS
            complete &= fin
            flag = []
            if diffs:
                flag.append(f"CONFIG DIFFERS: {diffs}")
            if commit != want:
                flag.append(f"commit {str(commit)[:8]} (expected {want[:8]})")
            if flag:
                ok = False
            rows[idx].append(dict(id=r.id, seed=c.get("seed"), state=r.state, n=len(h),
                                  mx=max(h) if h else float("nan"),
                                  last=h[-1] if h else float("nan"),
                                  last10=float(np.mean(h[-10:])) if h else float("nan"),
                                  argmax=int(np.argmax(h)) if h else -1))
            x = rows[idx][-1]
            print(f"  idx {idx} seed {x['seed']:>3} {x['id']:<10} {x['state']:<9} evals {x['n']:>3}/{N_EVALS}"
                  f"  max {x['mx']:.3f} (at eval {x['argmax']:>3})  final {x['last']:.3f}  last10 {x['last10']:.3f}"
                  f"  path {c.get('reward_model_path')}" + ("   " + "; ".join(flag) if flag else ""))
    print(f"\n  audit clean: {ok}   all 12 finished with {N_EVALS} evals: {complete}")

    if set(rows) != set(ORIGINALS) or not all(len(rows[i]) == 1 + len(SEEDS) for i in ORIGINALS):
        print("\nArms incomplete -- stopping before the analysis.")
        return 1

    print()
    print("=" * 96)
    print("ANALYSIS" + ("" if complete else "  -- PROVISIONAL: not all runs finished; do NOT apply the decision"))
    print("=" * 96)
    for label, key in (("max over evals (PRIMARY)", "mx"), ("final point", "last"), ("mean of last 10", "last10")):
        a2 = [x[key] for x in rows[2]]
        a3 = [x[key] for x in rows[3]]
        s, df = pooled([a2, a3])
        lo = math.sqrt(df * s * s / stats.chi2.ppf(0.95, df))
        hi = math.sqrt(df * s * s / stats.chi2.ppf(0.05, df))
        d = float(np.mean(a3) - np.mean(a2))
        half = stats.t.ppf(0.975, df) * s * math.sqrt(1 / len(a2) + 1 / len(a3))
        p_ok = stats.norm.cdf(abs(d) / (s * math.sqrt(2))) if s > 0 else float("nan")
        print(f"\n  {label}")
        print(f"    idx 2: mean {np.mean(a2):.3f}  sd {np.std(a2, ddof=1):.3f}  {np.round(a2, 3).tolist()}")
        print(f"    idx 3: mean {np.mean(a3):.3f}  sd {np.std(a3, ddof=1):.3f}  {np.round(a3, 3).tolist()}")
        print(f"    pooled sigma {s:.4f}  (90% CI {lo:.4f}-{hi:.4f}, {df} df)")
        print(f"    delta idx3 - idx2 {d:+.4f}  (95% CI {d - half:+.4f} to {d + half:+.4f})")
        print(f"    P(one run per index picks the higher-mean arm) {p_ok:.2f}")
        if key == "mx":
            fires = s >= THRESHOLD
            print(f"    sigma*sqrt2 = {s * math.sqrt(2):.4f} vs median top-2 gap {MEDIAN_GAP}")
            print(f"    DECISION (sigma >= {THRESHOLD}): "
                  + ("FIRES -- single-run stage 4 cannot resolve typical top-2 gaps"
                     if fires else "does not fire -- single-run stage 4 stands, disclose sigma")
                  + ("" if complete else "   [PROVISIONAL]"))
            print(f"    CI context: lower 90% bound {lo:.4f} {'>=' if lo >= THRESHOLD else '<'} threshold; "
                  f"upper {hi:.4f} {'>=' if hi >= THRESHOLD else '<'} threshold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
