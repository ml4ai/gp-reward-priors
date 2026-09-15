"""selection_gates.py — the BNN winner-eligibility gates, in ONE place.

Imported by check_sweep_convergence.py and check_winner_eligibility.py.  Before
2026-09-15 each tool carried its own copy; the convergence tool still applied the
§3.2.1 z-form and the eligibility tool the round-2 RAW z-form plus a clamp limit,
both superseded by §3.2.12.  One module stops that drift.

Operative gates (HANDOFF_HP_SELECTION.md §3.2.12, amending §3.2.1).  A BNN trial is
ELIGIBLE only if all hold:

  1. stationarity, scale     |log val_fn_drift_centred_scale_ratio_median| <= log(1.122)
     stationarity, location  val_fn_drift_centred_loc_sd_median            <= 0.155
  2. non-degeneracy          val_cvar_degeneracy_pass                       == 1
  3. resolution              val_pred_centred_ess_median                    >= 40

τ = 1.122 and τ_loc = 0.155 are DERIVED from the CVaR's own measurement precision
at conservatism 0.75 and ess 40 (§4.3.88), not calibrated on results.  They are
constants here, deliberately not command-line options: a pre-registered threshold
that can be overridden at selection time invites tuning.

Missing keys.  MR/PT sweeps log none of these keys and are ungated.  In a GATED
sweep (any trial logs any gate key), a trial missing a gate key is INELIGIBLE —
an unmeasured gate cannot be passed.  (The old tools skipped absent keys, which
would silently admit such a trial.)
"""

import math

TAU_SCALE = 1.122
TAU_LOC = 0.155
ESS_MIN = 40.0

K_RATIO = "val_fn_drift_centred_scale_ratio_median"
K_LOC_SD = "val_fn_drift_centred_loc_sd_median"
K_DEGEN = "val_cvar_degeneracy_pass"
K_ESS = "val_pred_centred_ess_median"


def _log_ratio_ok(v):
    return v > 0 and abs(math.log(v)) <= math.log(TAU_SCALE)


GATES = (
    (K_RATIO, "scale", _log_ratio_ok),
    (K_LOC_SD, "loc", lambda v: v <= TAU_LOC),
    (K_DEGEN, "degen", lambda v: bool(v)),
    (K_ESS, "ess", lambda v: v >= ESS_MIN),
)
GATE_KEYS = tuple(k for k, _, _ in GATES)


def has_gate_keys(summ):
    """True if this summary logs any gate key (i.e. it comes from a gated BNN sweep)."""
    return any(summ.get(k) is not None for k in GATE_KEYS)


def gate_failures(summ, gated=None):
    """Labels of the gates this trial fails.

    gated: whether the trial belongs to a gated sweep.  None -> infer from this
    summary alone.  Callers that see the whole sweep should pass the sweep-level
    answer, so a trial that logged no gate key at all is not waved through.
    """
    if gated is None:
        gated = has_gate_keys(summ)
    if not gated:
        return []
    bad = []
    for key, label, ok in GATES:
        v = summ.get(key)
        if v is None:
            bad.append(f"{label}=missing")
        elif isinstance(v, float) and math.isnan(v):
            bad.append(f"{label}=NaN")
        elif not ok(v):
            bad.append(label)
    return bad


def gate_values(summ):
    """Display values: |log ratio|, loc_sd, degeneracy margin, centred ess."""
    r = summ.get(K_RATIO)
    lr = abs(math.log(r)) if isinstance(r, (int, float)) and r > 0 else float("nan")

    def num(k):
        v = summ.get(k)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")

    return {"logr": lr, "loc_sd": num(K_LOC_SD),
            "margin": num("val_cvar_degeneracy_margin"), "ess": num(K_ESS)}


def _selftest():
    ok = {K_RATIO: 1.05, K_LOC_SD: 0.10, K_DEGEN: 1, K_ESS: 60.0}
    assert gate_failures(ok) == []
    assert gate_failures({**ok, K_RATIO: 1.13}) == ["scale"]
    assert gate_failures({**ok, K_RATIO: 1 / 1.13}) == ["scale"]          # symmetric in log
    assert gate_failures({**ok, K_RATIO: 1.122}) == []                     # boundary passes
    assert gate_failures({**ok, K_LOC_SD: 0.156}) == ["loc"]
    assert gate_failures({**ok, K_DEGEN: 0}) == ["degen"]
    assert gate_failures({**ok, K_ESS: 39.9}) == ["ess"]
    assert gate_failures({**ok, K_ESS: float("nan")}) == ["ess=NaN"]
    partial = {k: v for k, v in ok.items() if k != K_ESS}
    assert gate_failures(partial) == ["ess=missing"]
    assert gate_failures({"val_mean_cross_entropy": 0.3}) == []            # MR/PT: ungated
    assert gate_failures({"val_mean_cross_entropy": 0.3}, gated=True) == [
        "scale=missing", "loc=missing", "degen=missing", "ess=missing"]
    print("selection_gates self-test: PASS")


if __name__ == "__main__":
    _selftest()
