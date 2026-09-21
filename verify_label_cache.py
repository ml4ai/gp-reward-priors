#!/usr/bin/env python
"""Prove the reward-label cache returns EXACTLY what recomputing would.

Handoff §3.2.9 / to-do 17.  The cache exists so ~5 TB of posterior chains can be
deleted after labelling.  Once they are gone the labels cannot be re-derived, so
the equality it rests on has to be demonstrated BEFORE anything is deleted, on a
real run directory, not argued from the code.

What it checks, in order:

  1. `_bnn_cvar_labels` computed directly (cache bypassed)      -> reference
  2. `qlearning_dataset_bnn` with an empty cache (populates)    -> must equal (1)
  3. `qlearning_dataset_bnn` again (must HIT)                   -> must equal (1)
  4. bit-identical, not merely close: `np.array_equal` on the raw float32
  5. the keep-masked rewards the caller actually receives also match
  6. a changed parameter must MISS rather than silently reuse
  7. timing, so the stage-4 saving is measured rather than assumed

Runs on the GPU box (needs d4rl/gym/torch).  It writes to a TEMPORARY cache
directory by default, so it cannot disturb a real cache.

Usage:
    python gp_reward-priors/verify_label_cache.py \\
        --run-dir exp/<bnn_run>_0 --env antmaze-large-diverse-v2 \\
        --alpha 0.95 --n-samples 500 --centre-draws --device cuda
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../algorithms/offline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="BNN run dir (with sampling_f/)")
    ap.add_argument("--env", required=True, help="d4rl env id, e.g. antmaze-large-diverse-v2")
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--centre-draws", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None,
                    help="default: a temp dir, so a real cache is never touched")
    a = ap.parse_args()

    tmp = a.cache_dir or tempfile.mkdtemp(prefix="rlc_verify_")
    os.environ["REWARD_LABEL_CACHE_DIR"] = tmp
    print(f"[verify] cache dir: {tmp}"
          f"{'' if a.cache_dir else '  (temporary — will be removed)'}")

    import gym
    import d4rl  # noqa: F401 — registers the envs
    import reward_label_cache as rlc
    import iql_eval as M

    env = gym.make(a.env)
    dataset = env.get_dataset()
    N = dataset["rewards"].shape[0]
    obs_all = dataset["observations"].astype(np.float32)
    act_all = dataset["actions"].astype(np.float32)
    print(f"[verify] {a.env}: N={N}, labels expected {N - 1}")

    ok = True
    try:
        # 1. reference: the helper directly, cache bypassed entirely
        t0 = time.time()
        ref, meta = M._bnn_cvar_labels(a.run_dir, a.alpha, a.n_samples, a.device,
                                       a.centre_draws, obs_all, act_all, N)
        t_direct = time.time() - t0
        print(f"\n[verify] 1. direct compute: {t_direct:.1f}s, "
              f"{ref.shape[0]} labels, dtype {ref.dtype}, "
              f"mean {ref.mean():.6f}")

        # 2. through the wrapper with an empty cache -> populates
        t0 = time.time()
        d_miss = M.qlearning_dataset_bnn(
            env, a.run_dir, alpha=a.alpha, n_samples=a.n_samples,
            device=a.device, dataset=dataset, centre_draws=a.centre_draws)
        t_miss = time.time() - t0

        # 3. again -> must HIT
        t0 = time.time()
        d_hit = M.qlearning_dataset_bnn(
            env, a.run_dir, alpha=a.alpha, n_samples=a.n_samples,
            device=a.device, dataset=dataset, centre_draws=a.centre_draws)
        t_hit = time.time() - t0

        # 4/5. bit-identical, on the raw labels and on what the caller receives
        key, payload = rlc.make_key(
            "bnn", a.run_dir, a.env, N - 1,
            {"alpha": float(a.alpha), "n_samples": int(a.n_samples),
             "centre_draws": bool(a.centre_draws)})
        cached, status = rlc.load(key, payload, N - 1, None)
        checks = [
            ("cached labels are bit-identical to direct compute",
             np.array_equal(cached, ref)),
            ("cached dtype is float32", cached.dtype == np.float32),
            ("miss-path rewards match hit-path rewards, bit for bit",
             np.array_equal(d_miss["rewards"], d_hit["rewards"])),
            ("caller-visible rewards are finite",
             bool(np.all(np.isfinite(d_hit["rewards"])))),
            ("observations identical across miss/hit",
             np.array_equal(d_miss["observations"], d_hit["observations"])),
            ("terminals identical across miss/hit",
             np.array_equal(d_miss["terminals"], d_hit["terminals"])),
            ("second call was a cache HIT (>=5x faster than the miss)",
             t_hit * 5 <= t_miss),
        ]

        # 6. a changed parameter must MISS, never silently reuse
        k2, p2 = rlc.make_key(
            "bnn", a.run_dir, a.env, N - 1,
            {"alpha": float(a.alpha), "n_samples": int(a.n_samples),
             "centre_draws": not a.centre_draws})
        checks.append(("flipping centre_draws MISSES rather than reusing",
                       rlc.load(k2, p2, N - 1, None)[0] is None))

        print()
        for label, passed in checks:
            print(f"  [{'PASS' if passed else '** FAIL **'}] {label}")
            ok &= bool(passed)

        print(f"\n[verify] timing: direct {t_direct:.1f}s | miss {t_miss:.1f}s "
              f"| hit {t_hit:.1f}s")
        if t_hit > 0:
            print(f"[verify] stage 4 runs 8 normalization indices per cell: "
                  f"{t_miss:.1f}s + 7 x {t_hit:.1f}s = "
                  f"{t_miss + 7 * t_hit:.1f}s against "
                  f"{8 * t_miss:.1f}s uncached "
                  f"({8 * t_miss / max(t_miss + 7 * t_hit, 1e-9):.1f}x)")
        entry = os.path.join(tmp, f"{key}.npz")
        if os.path.exists(entry):
            print(f"[verify] entry size {os.path.getsize(entry) / 1e6:.1f} MB "
                  f"vs chains under {a.run_dir}/sampling_f")
        print(f"[verify] provenance recorded: {meta.get('n_chains')} chains, "
              f"{meta.get('n_draws')} draws, digest {meta.get('weights_digest')}")

        print("\n" + ("VERIFIED — the cache reproduces the labels exactly. "
                      "Chains may be deleted."
                      if ok else
                      "** FAILED — DO NOT DELETE ANY CHAINS. **"))
        return 0 if ok else 1
    finally:
        if not a.cache_dir:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
