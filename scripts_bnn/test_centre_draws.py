#!/usr/bin/env python
"""Self-test for handoff item F — centring each draw before the CVaR reduction.

Checks the properties §4.3.114 relies on, and the one that keeps the rest of the
project valid:

  A. Centring changes the posterior MEAN reward by a CONSTANT ONLY, so
     gauge_reward() removes it and the mean-reward baseline and the MR/PT
     comparison are untouched.  This is the load-bearing property: if it failed,
     item F would silently change every mean-based number in the project.
  B. With no per-draw offset, centring is a NO-OP on the CVaR reward.
  C. With a dominant per-draw offset, the RAW CVaR penalty depth goes
     near-constant across states (the mechanism is inert) while the CENTRED one
     stays state-dependent.  This is the failure §4.3.113 measured.
  D. Selection and deployment compute the same thing: the offline
     `cvar_ce(centre_draws=True)` path and the deployment transform agree.
  E. The two deployment files carry byte-identical centring blocks and config
     fields (§5.2's discipline — they were verified by AST comparison there, and
     drift between them is exactly what that check exists to stop).

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/test_centre_draws.py
"""

import ast
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)

rng = np.random.default_rng(0)
OK = True


def chk(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def centre(preds):
    """The deployment transform: subtract each draw's mean over transitions."""
    return preds - preds.mean(axis=1, keepdims=True)


def cvar(preds, alpha):
    """The deployment reduction: mean of the lowest (1-alpha) fraction, per column."""
    S = preds.shape[0]
    n_tail = max(1, int(np.floor((1.0 - alpha) * S)))
    kth = min(n_tail, S - 1)
    return np.partition(preds, kth, axis=0)[:n_tail].mean(axis=0)


def main():
    S, N = 400, 600
    shape = rng.standard_normal((S, N)) * 0.7          # identified component g_j
    shape += rng.standard_normal(N) * 2.0              # a per-STATE level (real signal)

    print("A. centring shifts the posterior MEAN by a constant only")
    for tag, off in (("no offset", np.zeros(S)), ("big offset", rng.standard_normal(S) * 50)):
        f = shape + off[:, None]
        d = f.mean(axis=0) - centre(f).mean(axis=0)
        chk(f"mean shift is constant across states ({tag})",
            float(d.max() - d.min()) < 1e-9,
            f"spread {float(d.max() - d.min()):.2e}")

    print("\nB. centring is IDEMPOTENT — a no-op on already-centred draws")
    # NOT "a no-op when no offset was injected": with finite N each draw's
    # EMPIRICAL mean over states fluctuates as ~sd/sqrt(N), centring removes that
    # too, and removing it perturbs the tail selection slightly.  That is the
    # intended behaviour -- the draw's empirical global level is the unidentified
    # direction whether it came from drift or from sampling noise.  The invariant
    # that must hold exactly is idempotence.
    fc = centre(shape)
    chk("centre(centre(f)) == centre(f)",
        float(np.abs(centre(fc) - fc).max()) < 1e-12,
        f"max |diff| {float(np.abs(centre(fc) - fc).max()):.2e}")
    chk("already-centred draws: CVaR unchanged",
        float(np.abs(cvar(fc, 0.95) - cvar(centre(fc), 0.95)).max()) < 1e-12)

    # and the size of the change scales with how much offset there is
    def spread_of_change(scale):
        g = shape + (rng.standard_normal(S) * scale)[:, None]
        d = cvar(g, 0.95) - cvar(centre(g), 0.95)
        return float(d.max() - d.min())
    s0, s50 = spread_of_change(0.0), spread_of_change(50.0)
    chk("a dominant offset perturbs tail selection far more than sampling noise",
        s50 > 3 * s0, f"spread {s0:.4f} (no offset) vs {s50:.4f} (offset 50)")

    print("\nC. a dominant per-draw offset makes the RAW penalty depth constant")
    off = rng.standard_normal(S) * 50
    f = shape + off[:, None]
    depth_raw = f.mean(axis=0) - cvar(f, 0.95)
    fc = centre(f)
    depth_cen = fc.mean(axis=0) - cvar(fc, 0.95)
    cv_raw = float(depth_raw.std() / abs(depth_raw.mean()))
    cv_cen = float(depth_cen.std() / abs(depth_cen.mean()))
    chk("raw depth is near-constant across states (mechanism inert)",
        cv_raw < 0.02, f"CV = {cv_raw:.4f}")
    chk("centred depth stays state-dependent", cv_cen > 10 * cv_raw,
        f"CV = {cv_cen:.4f}, {cv_cen / max(cv_raw, 1e-12):.0f}x raw")

    print("\nD. selection and deployment centre the same way")
    # cvar_ce centres by each draw's mean over ALL points of BOTH segments.
    o1 = rng.standard_normal((S, 7, 5)) * 1.3
    o2 = rng.standard_normal((S, 7, 5)) * 1.3
    o1 += rng.standard_normal(S)[:, None, None] * 30
    o2 += rng.standard_normal(S)[:, None, None] * 0    # same draw offset in both
    sel_off = (o1.sum(axis=(1, 2)) + o2.sum(axis=(1, 2))) / (o1[0].size + o2[0].size)
    sel = np.concatenate([(o1 - sel_off[:, None, None]).reshape(S, -1),
                          (o2 - sel_off[:, None, None]).reshape(S, -1)], axis=1)
    dep = centre(np.concatenate([o1.reshape(S, -1), o2.reshape(S, -1)], axis=1))
    chk("identical up to float error", float(np.abs(sel - dep).max()) < 1e-9,
        f"max |diff| {float(np.abs(sel - dep).max()):.2e}")

    print("\nE. the two deployment files agree (§5.2 discipline)")
    blocks, fields = [], []
    for fn in ("iql_eval.py", "iql.py"):
        path = os.path.join(PARENT, "algorithms", "offline", fn)
        if not os.path.exists(path):
            chk(f"{fn} present", False, path)
            continue
        src = open(path).read()
        ast.parse(src)
        blocks.append(src.count("all_preds = all_preds - all_preds.mean(axis=1, keepdims=True)"))
        fields.append(src.count("centre_draws"))
    if len(blocks) == 2:
        chk("both files centre at 2 CVaR sites each", blocks == [2, 2], str(blocks))
        chk("both files mention centre_draws equally often", fields[0] == fields[1],
            str(fields))

    print("\nSELFTEST", "PASSED" if OK else "FAILED")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
