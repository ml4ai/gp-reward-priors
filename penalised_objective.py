"""penalised_objective.py — the metric the OPTIMISER sees (handoff §3.2.17).

The hard gates (`selection_gates.py`, §3.2.12) decide who may WIN, and they are
unchanged by this module.  This is the separate question of what the wandb Bayes
optimiser should MINIMISE while it searches.

The problem it fixes, measured at the round-4 pause (§4.3.108): the optimiser saw
raw `val_cvar_ce`, which is ungated, and §3.2.7 measured ρ(`val_cvar_ce`,
degeneracy margin) = +0.670 — the configurations that score best on the objective
are the ones least distinguishable from the mean.  So the search is pulled toward
the ineligible region.  At the pause **all four** sweeps' ungated best was
ineligible, and taking the eligible best instead cost +17.6% / +12.4% / +3.7% on
the objective, with large_diverse having no eligible trial at all.

                                  The form

    J  =  cvar_ce  +  (1 - P)  *  max(0, log 2 - cvar_ce)

    P  =  prod_g  Phi( (slack of gate g) / sd_g )

Read it as: **a trial forfeits the part of its advantage over chance that it
probably cannot keep.**  `P` is the probability that every gate genuinely passes,
each gate's slack measured in units of that gate's own run-to-run sd; `log 2` is
the cross-entropy of an uninformative predictor, which this project already uses
everywhere as the reference for a useless reward model (§3.6, §4.3.22, §4.3.40).

Properties that made this the chosen form:

  * **No free parameter.**  There is no penalty weight to tune — the exchange
    rate is fixed by `log 2` and the measured sds.  A hinge penalty
    `cvar_ce + lam * max_g z_g` was the alternative and was rejected: at the
    derived `lam = 2 * SE(cvar_ce)` it fixed only 1 of 3 sweeps, and the `lam`
    that worked (~10 * SE) is a tuned number, which §10.2 forbids.
  * **Bounded, and never flattering.**  `J` lies in [cvar_ce, log 2] and can
    never make a bad model look better than it is: the `max(0, ...)` means a
    trial already worse than chance is left exactly where it is.
  * **Noise-calibrated.**  A trial 0.1 sd inside a threshold is treated as the
    coin flip it is, rather than as a pass.  §4.3.108 measured 15 of 27 round-4
    verdicts sitting within one sd of some threshold.
  * **It prefers ROBUST eligibility.**  Among eligible trials `J` is not monotone
    in `cvar_ce`, so the optimiser is steered toward configurations that pass
    comfortably rather than marginally.  That is a deliberate second benefit and
    a disclosed cost — see §3.2.17.

Validated on the 27 discarded round-4 trials: all four eligible trials rank in
the top four by `J`, and the argmin is eligible in all three sweeps that contain
an eligible trial.

The sds are RUN-TO-RUN sds at this budget, not within-run SEs.  See §3.2.17 for
provenance and for the one that is weak (`loc_sd`).
"""

import math

import selection_gates as G

LOG2 = math.log(2.0)

# Run-to-run sd of each gate statistic, measured at the round-4 budget.
#   scale, degen, ess : §4.3.101's four pinned medium_play replicates
#   loc               : §3.2.12's four pinned replicates, from a RANGE not an sd
#                       -- the weakest input here; see §3.2.17.
SD = {
    "scale": 0.0226,
    "loc": 0.0117,
    "degen": 0.00358,
    "ess": 0.58,
}

K_CE = "val_cvar_ce"


def _phi(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def gate_slacks(summ):
    """Signed slack of each gate, in units of that gate's own run-to-run sd.

    Positive = inside the threshold.  Returns None for any gate whose key is
    missing or NaN, which the caller must treat as a failure (§4.3.108: an
    unmeasured gate cannot be passed).
    """
    ratio = summ.get(G.K_RATIO)
    loc = summ.get(G.K_LOC_SD)
    marg = summ.get("val_cvar_degeneracy_margin")
    ess = summ.get(G.K_ESS)

    out = {}
    out["scale"] = (
        (math.log(G.TAU_SCALE) - abs(math.log(ratio))) / SD["scale"]
        if _finite(ratio) and ratio > 0 else None
    )
    out["loc"] = (G.TAU_LOC - loc) / SD["loc"] if _finite(loc) else None
    # The degeneracy margin is already gap - threshold, so its slack IS the margin.
    out["degen"] = marg / SD["degen"] if _finite(marg) else None
    out["ess"] = (ess - G.ESS_MIN) / SD["ess"] if _finite(ess) else None
    return out


def p_eligible(summ):
    """P(every gate genuinely passes), treating the gates as independent.

    The independence assumption is an approximation and its direction is known:
    §4.3.101 measured ρ(|log r|, degeneracy margin) = +0.397, i.e. gate 1 and
    gate 2 are NEGATIVELY associated as passes, so the product OVER-estimates P
    and the penalty is if anything too small.  Correcting it would need a joint
    model there is no basis for; it is stated rather than fudged (§3.2.17).
    """
    p = 1.0
    for z in gate_slacks(summ).values():
        if z is None:
            return 0.0
        p *= _phi(z)
    return p


def penalised_objective(summ):
    """J — the quantity the sweep minimises.  NaN if the objective is missing."""
    ce = summ.get(K_CE)
    if not _finite(ce):
        return float("nan")
    return ce + (1.0 - p_eligible(summ)) * max(0.0, LOG2 - ce)


def _selftest():
    ok = True

    def chk(name, got, want, tol=1e-9):
        nonlocal ok
        good = abs(got - want) <= tol if isinstance(want, float) else got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got!r}")

    # a comfortably eligible trial is barely penalised
    good = {G.K_RATIO: 1.0, G.K_LOC_SD: 0.0, G.K_ESS: 400.0,
            "val_cvar_degeneracy_margin": 1.0, K_CE: 0.30}
    chk("P(elig) ~ 1 for a clean trial", round(p_eligible(good), 6), 1.0)
    chk("J == cvar_ce for a clean trial", round(penalised_objective(good), 6), 0.30)

    # a hard failure on one gate is pushed to log 2
    bad = dict(good, **{G.K_RATIO: math.exp(1.0)})       # |log r| = 1.0, ~39 sd out
    chk("P(elig) ~ 0 on a hard gate-1 failure", round(p_eligible(bad), 9), 0.0)
    chk("J ~ log 2 on a hard failure", round(penalised_objective(bad), 6),
        round(LOG2, 6))

    # exactly ON a threshold is a coin flip
    edge = dict(good, **{"val_cvar_degeneracy_margin": 0.0})
    chk("P(elig) = 0.5 exactly on the gate-2 threshold",
        round(p_eligible(edge), 6), 0.5)

    # never flatters a model that is already worse than chance
    awful = dict(bad, **{K_CE: 0.9})
    chk("J == cvar_ce when cvar_ce > log 2", round(penalised_objective(awful), 6), 0.9)

    # bounded
    mid = dict(good, **{"val_cvar_degeneracy_margin": 0.0, K_CE: 0.40})
    j = penalised_objective(mid)
    chk("J is inside [cvar_ce, log 2]", 0.40 <= j <= LOG2 + 1e-12, True)

    # a missing gate key is a failure, not a free pass
    missing = {k: v for k, v in good.items() if k != G.K_ESS}
    chk("missing gate key -> P = 0", p_eligible(missing), 0.0)

    # monotone: more slack is never worse
    a = dict(good, **{"val_cvar_degeneracy_margin": 0.002})
    b = dict(good, **{"val_cvar_degeneracy_margin": 0.010})
    chk("J decreases as gate-2 slack grows",
        penalised_objective(b) < penalised_objective(a), True)

    # the thresholds come from selection_gates, so they cannot drift apart
    chk("tau shared with selection_gates", G.TAU_SCALE, 1.122)
    chk("ess floor shared with selection_gates", G.ESS_MIN, 40.0)

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
