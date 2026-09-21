#!/usr/bin/env python
"""Populate the reward-label cache for EVERY conservatism the evaluation needs.

Handoff §3.2.9 / to-do 17.  The IQL evaluation is run at more than one level —
**CVaR at alpha 0.95** and **the posterior mean at alpha 0.0** — and `alpha` is
cache-key material, so each is a SEPARATE entry.  Deleting the chains after
caching only one destroys the ability to produce the other, permanently and
silently.  This tool caches all of them in one pass, then reports per-source
completeness so the deletion decision is made on evidence.

It drives the same `qlearning_dataset_bnn` the evaluation uses, so the cached
labels are produced by the verified code path, not by a reimplementation of it.

COST.  One forward pass per (run dir, alpha): the pass over S draws x N
transitions is repeated for each alpha even though only the tail reduction
differs.  Caching two alphas therefore costs 2x the labelling compute, once.
That is deliberate — collapsing it would mean restructuring the labelling
helper again and re-running `verify_label_cache.py`, and the compute is a
one-off against a verified critical path.

Usage:
    python gp_reward-priors/precompute_labels.py \\
        --env antmaze-large-diverse-v2 --alphas 0.95,0.0 \\
        --centre-draws --device cuda \\
        --run-dir exp/bnn_large_diverse_s0 --run-dir exp/bnn_large_diverse_s1 ...

    # or let it expand a glob:
    python gp_reward-priors/precompute_labels.py \\
        --env antmaze-large-diverse-v2 --alphas 0.95,0.0 --centre-draws \\
        --glob 'exp/bnn_large_diverse_s*'
"""

import argparse
import glob as _glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../algorithms/offline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, help="d4rl env id")
    ap.add_argument("--run-dir", action="append", default=[],
                    help="BNN run dir; repeatable")
    ap.add_argument("--glob", default=None, help="glob expanded into run dirs")
    ap.add_argument("--alphas", default="0.95,0.0",
                    help="comma-separated conservatism levels (default 0.95,0.0: "
                         "CVaR and posterior mean)")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--centre-draws", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-centring-invariance", action="store_true",
                    help="at alpha=0, verify centred and raw labels differ by a "
                         "CONSTANT — the claim that item F leaves the "
                         "mean-reward baseline untouched (handoff 4.3.114)")
    a = ap.parse_args()

    dirs = list(a.run_dir)
    if a.glob:
        dirs += sorted(d for d in _glob.glob(a.glob) if os.path.isdir(d))
    dirs = sorted(set(dirs))
    if not dirs:
        sys.exit("no run dirs given (--run-dir / --glob)")
    alphas = [float(t) for t in a.alphas.split(",") if t.strip()]
    for al in alphas:
        if not (0.0 <= al < 1.0):
            sys.exit(f"alpha must be in [0, 1), got {al}")

    print(f"[precompute] {len(dirs)} run dir(s) x {len(alphas)} alpha(s) "
          f"{alphas}  env={a.env}  centre_draws={a.centre_draws}")

    import gym
    import d4rl  # noqa: F401 — registers the envs
    import reward_label_cache as rlc
    import iql_eval as M

    env = gym.make(a.env)
    dataset = env.get_dataset()          # loaded ONCE and reused throughout
    N = dataset["rewards"].shape[0]
    print(f"[precompute] {a.env}: N={N}, labels per entry {N - 1}")

    done, failed = [], []
    t_start = time.time()
    for d in dirs:
        if not os.path.isdir(os.path.join(d, "sampling_f")):
            print(f"\n!! {d}: no sampling_f/ — chains already gone?  SKIPPING "
                  f"(nothing here can be cached)")
            failed.append((d, None, "no sampling_f"))
            continue
        for al in alphas:
            t0 = time.time()
            try:
                ds = M.qlearning_dataset_bnn(
                    env, d, alpha=al, n_samples=a.n_samples,
                    device=a.device, dataset=dataset,
                    centre_draws=a.centre_draws)
                del ds
                done.append((d, al, time.time() - t0))
                print(f"[precompute] {d}  alpha={al}  ok  "
                      f"({time.time() - t0:.1f}s)")
            except Exception as e:  # noqa: BLE001 — report, keep going
                failed.append((d, al, f"{type(e).__name__}: {e}"))
                print(f"!! {d}  alpha={al}  FAILED: {type(e).__name__}: {e}")

    if a.check_centring_invariance:
        print("\n[precompute] centring invariance at alpha=0 "
              "(handoff 4.3.114 / 4.3.124)")
        d = next((x for x in dirs
                  if os.path.isdir(os.path.join(x, "sampling_f"))), None)
        if d is None:
            print("   skipped: no run dir with chains present")
        else:
            obs = dataset["observations"].astype(np.float32)
            act = dataset["actions"].astype(np.float32)
            cen, _ = M._bnn_cvar_labels(d, 0.0, a.n_samples, a.device, True,
                                        obs, act, N)
            raw, _ = M._bnn_cvar_labels(d, 0.0, a.n_samples, a.device, False,
                                        obs, act, N)
            diff = cen.astype(np.float64) - raw.astype(np.float64)
            spread = float(diff.max() - diff.min())
            scale = float(np.abs(raw).mean())
            print(f"   offset {diff.mean():+.6g}, spread {spread:.3e}, "
                  f"reward scale {scale:.6g}")
            ok = spread <= 1e-4 * max(scale, 1e-9)
            print(f"   [{'PASS' if ok else '** FAIL **'}] at alpha=0 centring "
                  f"shifts every label by the SAME constant, which "
                  f"gauge_reward() removes exactly -- so the posterior-mean "
                  f"baseline is untouched by item F.")
            if not ok:
                failed.append((d, 0.0, "centring invariance"))

    print(f"\n[precompute] {len(done)} cached, {len(failed)} failed, "
          f"{time.time() - t_start:.0f}s total")
    for d, al, why in failed:
        print(f"   FAILED {d} alpha={al}: {why}")

    print("\n[precompute] completeness (this is what gates deletion):")
    have = {}
    for key, meta in rlc.entries():
        if "ERROR" in meta:
            continue
        src = meta.get("source_dir")
        p = meta.get("payload", {}).get("params", {})
        if src:
            have.setdefault(os.path.abspath(src), set()).add(float(p.get("alpha")))
    req = set(alphas)
    complete = 0
    for d in dirs:
        got = have.get(os.path.abspath(d), set())
        missing = sorted(req - got)
        complete += not missing
        print(f"   {'OK  ' if not missing else 'WAIT'} {d}: have "
              f"{sorted(got)}{'' if not missing else f', MISSING {missing}'}")
    print(f"\n   {complete}/{len(dirs)} run dir(s) have every required alpha.")
    if complete == len(dirs) and not failed:
        print("   Next: verify_label_cache.py on at least one dir, then\n"
              f"   reward_label_cache.py --emit-rm --require-alphas {a.alphas}")
        return 0
    print("   DO NOT DELETE ANY CHAINS until every dir reads OK.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
