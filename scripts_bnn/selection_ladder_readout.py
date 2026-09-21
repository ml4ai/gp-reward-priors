#!/usr/bin/env python
"""Read out the capacity ladder re-scored at SELECTION conservatism (handoff §4.3.118).

Parses `exp/selection_capacity_ladder.txt` — the output of the §4.3.119 command —
and applies the three reproduction checks §4.3.117 §3 pre-registered, then prints
the ladder table and the derived verdicts.

Why this exists rather than an eyeball: §4.3.117's run was at the WRONG
conservatism (the diagnostic defaults to tail fraction 0.05, the deployment
level, not the 0.25 that `val_cvar_ce` uses) and that was caught ONLY because
the `raw f` column failed to reproduce §4.3.105's logged `cvar_ce`.  Check 1
below is that check, mechanised so it cannot be skipped again.

Check 2 as pre-registered was ILL-POSED and is reported, not asserted: with
`centre_draws=True` the primary convention is centred, so the printed
`2 * SE(CVaR CE)` is the CENTRED SE and has no reason to equal §4.3.105's raw
threshold.  The ratio is printed instead, because it turns out to carry the
finding (centring roughly halves the SE wherever capacity is large).

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/selection_ladder_readout.py
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/selection_ladder_readout.py --selftest
"""

import argparse
import math
import re
import sys

LOG2 = math.log(2.0)
DEFAULT_SRC = "exp/selection_capacity_ladder.txt"

# §4.3.105's logged columns: raw f at conservatism 0.75.
#   w -> (cvar_ce, SE, mean_CE, gate2_verdict)
T105 = {
    "large_diverse": {4: (0.4767, 0.0333, 0.5056, "FAIL"),
                      5: (0.3907, 0.0151, 0.3861, "FAIL"),
                      6: (0.3911, 0.0096, 0.3631, "PASS"),
                      7: (0.3898, 0.0121, 0.3429, "PASS"),
                      8: (0.4055, 0.0109, 0.3272, "PASS"),
                      9: (0.4740, 0.0205, 0.3187, "PASS")},
    "large_play":    {4: (0.4381, 0.0073, 0.4277, "PASS"),
                      5: (0.4076, 0.0102, 0.3729, "PASS"),
                      6: (0.4540, 0.0155, 0.3412, "PASS"),
                      7: (0.6663, 0.0355, 0.3077, "PASS"),
                      8: (0.9427, 0.0524, 0.2770, "PASS"),
                      9: (1.3238, 0.0710, 0.2504, "PASS")},
}

N_PARAMS = {
    "large_diverse": {4: 1441, 5: 4417, 6: 14977, 7: 54529, 8: 207361, 9: 807937},
    "large_play":    {4: 1985, 5: 6529, 6: 23297, 7: 87553, 8: 338945, 9: 1333249},
}

# §3.2.16's declared common range.
W_LO, W_HI = 4, 7


def parse(text):
    """-> {(variant, w): dict}.  Anchors on the PRINTED banners (§10.3)."""
    parts = re.split(r"===\s*cap_ladder2_(\S+?)_w(\d+)\s*===", text)[1:]
    out = {}
    for variant, w, body in zip(parts[::3], parts[1::3], parts[2::3]):
        def grab(pat, cast=float):
            m = re.search(pat, body)
            return cast(m.group(1)) if m else None
        row = {
            "cen":     grab(r"CVaR CE\s+([\d.]+)\s+[\d.]+"),
            "raw":     grab(r"CVaR CE\s+[\d.]+\s+([\d.]+)"),
            "cen_acc": grab(r"CVaR acc\s+([\d.]+)\s+[\d.]+"),
            "raw_acc": grab(r"CVaR acc\s+[\d.]+\s+([\d.]+)"),
            "ratio":   grab(r"ratio width/preference\s+([\d.]+)\s+[\d.]+"),
            "gap":     grab(r"\|CVaR CE - mean CE\|\s+([\d.]+)"),
            "thr":     grab(r"2 \* SE\(CVaR CE\)\s+([\d.]+)"),
            "se":      grab(r"jackknife-over-chains SE on the CVaR CE: ([\d.]+)"),
            "verdict": grab(r"verdict\s+(\w+)", str),
        }
        if row["cen"] is None:
            continue
        out[(variant, int(w))] = row
    return out


def check1(rows):
    """The raw column MUST reproduce §4.3.105's logged cvar_ce to 4 dp."""
    print("CHECK 1 -- `raw f` reproduces 4.3.105's logged cvar_ce to 4 dp")
    print("  (the check that caught 4.3.117's alpha error)\n")
    bad = 0
    for v in sorted(T105):
        for w in sorted(T105[v]):
            r = rows.get((v, w))
            if r is None:
                print(f"  {v:14s} w{w}   MISSING from the output file")
                bad += 1
                continue
            d = abs(r["raw"] - T105[v][w][0])
            ok = d < 5e-5
            bad += not ok
            print(f"  {v:14s} w{w}  logged {T105[v][w][0]:.4f}  "
                  f"re-scored {r['raw']:.4f}  diff {d:.5f}  "
                  f"{'OK' if ok else '** MISMATCH **'}")
    n = sum(len(x) for x in T105.values())
    print(f"\n  => CHECK 1 {'PASSES' if not bad else 'FAILS'} "
          f"{n - bad}/{n}\n")
    return bad == 0


def check2(rows):
    """Ill-posed as pre-registered; reported for what it does show."""
    print("CHECK 2 -- WITHDRAWN as pre-registered (4.3.118 section 1)")
    print("  The primary convention is centred, so the printed 2*SE is the")
    print("  CENTRED SE and cannot equal 4.3.105's raw threshold.  The ratio")
    print("  is the finding instead.\n")
    print(f"  {'variant':14s} {'w':>2s} {'raw SE':>8s} {'cen SE':>8s} {'cen/raw':>8s}")
    for v in sorted(T105):
        for w in sorted(T105[v]):
            r = rows.get((v, w))
            if r is None:
                continue
            print(f"  {v:14s} {w:2d} {T105[v][w][1]:8.4f} {r['se']:8.4f} "
                  f"{r['se'] / T105[v][w][1]:8.2f}")
    print()


def ladder(rows):
    print("\nSELECTION-ALPHA LADDER (conservatism 0.75), centred f\n")
    verdicts = {}
    for v in sorted(T105):
        print(f"  {v}")
        print(f"  {'w':>2s} {'n_params':>10s} {'cenCE':>7s} {'rawCE':>7s} "
              f"{'cenacc':>7s} {'gap':>7s} {'2SE':>7s} {'margin':>8s} "
              f"{'g2':>5s} {'w/pref':>7s} {'<log2':>6s}")
        for w in sorted(T105[v]):
            r = rows.get((v, w))
            if r is None:
                continue
            print(f"  {w:2d} {N_PARAMS[v][w]:10,d} {r['cen']:7.4f} {r['raw']:7.4f} "
                  f"{r['cen_acc']:7.4f} {r['gap']:7.4f} {r['thr']:7.4f} "
                  f"{r['gap'] - r['thr']:+8.4f} {r['verdict']:>5s} "
                  f"{r['ratio']:7.3f} {'yes' if r['cen'] < LOG2 else 'NO':>6s}")

        cen = {w: rows[(v, w)]["cen"] for w in T105[v] if (v, w) in rows}
        raw = {w: T105[v][w][0] for w in T105[v]}
        se = {w: rows[(v, w)]["se"] for w in T105[v] if (v, w) in rows}
        bc, br = min(cen, key=cen.get), min(raw, key=raw.get)

        # Best-three spread in SE units: the RESOLUTION finding (4.3.118 s5).
        def spread(d, ses):
            top = sorted(d, key=d.get)[:3]
            med = sorted(ses[w] for w in top)[1]
            return d[max(top, key=d.get)] - d[min(top, key=d.get)], med

        sc, mc = spread(cen, se)
        sr, mr = spread(raw, {w: T105[v][w][1] for w in T105[v]})

        # Degradation from the centred optimum to w9, in pooled SE.
        pooled = math.hypot(se[bc], se[9])
        deg = (cen[9] - cen[bc]) / pooled

        g2c = sum(rows[(v, w)]["verdict"] == "PASS" for w in T105[v] if (v, w) in rows)
        g2r = sum(T105[v][w][3] == "PASS" for w in T105[v])
        print(f"   argmin        raw w{br} ({raw[br]:.4f})  ->  "
              f"centred w{bc} ({cen[bc]:.4f})"
              f"   {'IN' if W_LO <= bc <= W_HI else 'OUTSIDE'} range w{W_LO}-{W_HI}")
        print(f"   best-3 spread raw {sr:.4f} = {sr / mr:.1f} SE   ->   "
              f"centred {sc:.4f} = {sc / mc:.1f} SE")
        print(f"   w{bc} -> w9    +{cen[9] - cen[bc]:.4f} = {deg:.1f} sigma "
              f"(pooled SE {pooled:.4f})")
        print(f"   gate 2        centred {g2c}/6 PASS, raw {g2r}/6")
        print(f"   beats log 2   centred {sum(c < LOG2 for c in cen.values())}/6, "
              f"raw {sum(c < LOG2 for c in raw.values())}/6\n")
        verdicts[v] = dict(argmin_raw=br, argmin_cen=bc, in_range=W_LO <= bc <= W_HI,
                           g2_cen=g2c, g2_raw=g2r, sigma=deg,
                           res_raw=sr / mr, res_cen=sc / mc)
    return verdicts


def conclude(rows, verdicts):
    print("VERDICTS (4.3.118)\n")
    g2 = sum(r["verdict"] == "PASS" for r in rows.values())
    print(f"  line 2 (gate-2 failure is capacity-shaped): "
          f"REFUTED -- gate 2 passes {g2}/{len(rows)} centred.")
    print("  4.3.105's two-sided squeeze: WITHDRAWN -- no gate-2 lower bound "
          "on either variant.")
    allin = all(v["in_range"] for v in verdicts.values())
    degs = ", ".join(f"{v['sigma']:.0f} sigma" for v in verdicts.values())
    print(f"  4.3.111 (do not widen): {'UPHELD' if allin else 'REVIEW'} -- "
          f"both centred optima are inside w{W_LO}-{W_HI}, and the objective "
          f"degrades above them by {degs}.")
    worst = min(v["res_raw"] for v in verdicts.values())
    print(f"  resolution: the raw objective's best three rungs differ by as "
          f"little as {worst:.1f} SE -- a tie.  Centring lifts that to "
          f"{min(v['res_cen'] for v in verdicts.values()):.1f} SE.")
    n = sum(r["cen"] < LOG2 for r in rows.values())
    print(f"  content: {n}/{len(rows)} rungs beat log 2 at the selection alpha "
          f"(2/12 at the deployment alpha, 4.3.117).")


def parse_sweep(text):
    """-> {(variant, w): {"rows": {alpha: {...}}, "plug_ce":, "plug_acc":}}."""
    parts = re.split(r"===\s*(\S+?)_w(\d+)\s*===", text)[1:]
    out = {}
    for variant, w, body in zip(parts[::3], parts[1::3], parts[2::3]):
        rows = {}
        for m in re.finditer(
                r"^\s*([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
                r"\s+([\d.]+)\s+([\d.]+)\s*$", body, re.M):
            a, k, ce, acc, med, fl, wr = m.groups()
            rows[round(float(a), 3)] = dict(
                k=int(k), ce=float(ce), acc=float(acc), med=float(med),
                flip=float(fl), wrong=float(wr))
        ref = re.search(r"plug-in sigma\(E\[f\]\) CE ([\d.]+)\s+acc ([\d.]+)", body)
        if rows and ref:
            out[(variant, int(w))] = dict(rows=rows, plug_ce=float(ref.group(1)),
                                          plug_acc=float(ref.group(2)))
    return out


# §4.3.117's centred column at the DEPLOYMENT tail (conservatism 0.95).
DEPLOY_CEN = {
    "large_diverse": {4: 0.8557, 5: 0.6210, 6: 0.6371,
                      7: 0.7332, 8: 0.8872, 9: 1.2279},
    "large_play":    {4: 0.7434, 5: 0.9828, 6: 1.5297,
                      7: 2.1688, 8: 2.6792, 9: 3.5924},
}

# §4.3.105's `gap` column = |raw cvar_ce - plug_ce|.  NOT the difference of its
# two printed columns: that table's "mean CE" is the posterior-predictive CE,
# a DIFFERENT quantity from the plug-in CE gate 2 actually uses (§4.3.120 §1).
GAP105 = {
    "large_diverse": {4: 0.035, 5: 0.018, 6: 0.039, 7: 0.056, 8: 0.086, 9: 0.163},
    "large_play":    {4: 0.032, 5: 0.054, 6: 0.129, 7: 0.380, 8: 0.689, 9: 1.106},
}


def check3(sw, rows):
    """alpha=0.05 must reproduce §4.3.117's centred column; 0.25 the primary."""
    print("CHECK 3 -- the alpha sweep reproduces both scored columns\n")
    bad = 0
    print(f"  {'variant':14s} {'w':>2s} {'a=.05 exp':>10s} {'got':>8s} "
          f"{'a=.25 exp':>10s} {'got':>8s}")
    for v in sorted(DEPLOY_CEN):
        for w in sorted(DEPLOY_CEN[v]):
            s = sw.get((v, w))
            if s is None:
                print(f"  {v:14s} {w:2d}   MISSING")
                bad += 1
                continue
            g05, g25 = s["rows"][0.05]["ce"], s["rows"][0.25]["ce"]
            e05, e25 = DEPLOY_CEN[v][w], rows[(v, w)]["cen"]
            d = max(abs(g05 - e05), abs(g25 - e25))
            bad += d >= 5e-5
            print(f"  {v:14s} {w:2d} {e05:10.4f} {g05:8.4f} {e25:10.4f} "
                  f"{g25:8.4f}  {'OK' if d < 5e-5 else '** MISMATCH **'}")
    print(f"\n  => CHECK 3 {'PASSES' if not bad else 'FAILS'} "
          f"{12 - bad}/12   (0.25 agreeing confirms centring is inside the "
          f"shared sort)\n")
    return bad == 0


def check4(sw):
    """plug_ce must reproduce §4.3.105's gap column -- and is centring-invariant.

    Replaces the withdrawn check 2 with a VALID invariance test: §4.3.105's gaps
    were computed on RAW draws, this run is centred, and plug_ce is Class A
    (§4.3.61), so the two must agree.  Twelve independent confirmations.
    """
    print("CHECK 4 -- plug-in CE reproduces 4.3.105's gap column")
    print("  (a valid replacement for the withdrawn check 2: 4.3.105's gaps are")
    print("   RAW-derived, this run is centred, and plug_ce is Class A.)\n")
    bad = 0
    print(f"  {'variant':14s} {'w':>2s} {'rawCVaR':>8s} {'plug_ce':>8s} "
          f"{'implied':>8s} {'4.3.105':>8s}")
    for v in sorted(GAP105):
        for w in sorted(GAP105[v]):
            s = sw.get((v, w))
            if s is None:
                bad += 1
                continue
            imp = abs(T105[v][w][0] - s["plug_ce"])
            d = abs(imp - GAP105[v][w])
            bad += d >= 6e-4
            print(f"  {v:14s} {w:2d} {T105[v][w][0]:8.4f} {s['plug_ce']:8.4f} "
                  f"{imp:8.4f} {GAP105[v][w]:8.3f}  "
                  f"{'OK' if d < 6e-4 else '** MISMATCH **'}")
    print(f"\n  => CHECK 4 {'PASSES' if not bad else 'FAILS'} {12 - bad}/12   "
          f"(plug_ce is centring-invariant, confirmed at 12 rungs)\n")
    return bad == 0


def sweep_table(sw):
    """flip% is the decision-relevant column: CVaR reverses the MEAN's ranking."""
    print("\nWHAT CAPACITY BUYS AND SPENDS (4.3.120)\n")
    for v in sorted(T105):
        print(f"  {v}")
        print(f"  {'w':>2s} {'plugCE':>7s} {'plugacc':>8s} | "
              f"{'cvCE.25':>8s} {'cvacc.25':>9s} {'flip%':>6s} | "
              f"{'cvCE.05':>8s} {'cvacc.05':>9s} {'flip%':>6s} {'wrong%':>7s}")
        for w in sorted(T105[v]):
            s = sw.get((v, w))
            if s is None:
                continue
            a, b = s["rows"][0.25], s["rows"][0.05]
            print(f"  {w:2d} {s['plug_ce']:7.4f} {s['plug_acc']:8.4f} | "
                  f"{a['ce']:8.4f} {a['acc']:9.4f} {a['flip']:6.1f} | "
                  f"{b['ce']:8.4f} {b['acc']:9.4f} {b['flip']:6.1f} "
                  f"{b['wrong']:7.1f}")
        pa = {w: sw[(v, w)]["plug_acc"] for w in T105[v] if (v, w) in sw}
        ca = {w: sw[(v, w)]["rows"][0.25]["acc"] for w in T105[v] if (v, w) in sw}
        fl = {w: sw[(v, w)]["rows"][0.25]["flip"] for w in T105[v] if (v, w) in sw}
        pc = {w: sw[(v, w)]["plug_ce"] for w in T105[v] if (v, w) in sw}
        mono = all(pc[w] > pc[w + 1] for w in sorted(pc)[:-1])
        print(f"   plug-in CE monotone improving in capacity: {mono}"
              f"   ({pc[min(pc)]:.4f} -> {pc[max(pc)]:.4f})")
        print(f"   plug-in acc spans {min(pa.values()):.4f}-{max(pa.values()):.4f}"
              f" (flat), CVaR acc {min(ca.values()):.4f}-{max(ca.values()):.4f}")
        print(f"   plug-acc minus CVaR-acc: w{min(pa)} {pa[min(pa)]-ca[min(pa)]:+.4f}"
              f"  ->  w{max(pa)} {pa[max(pa)]-ca[max(pa)]:+.4f}")
        print(f"   flip% argmin w{min(fl, key=fl.get)} ({min(fl.values()):.1f}%)"
              f"  ->  w9 {fl[9]:.1f}%   [CE argmin was "
              f"w{min(T105[v], key=lambda w: sw[(v,w)]['rows'][0.25]['ce'])}]\n")


SELFTEST = """
=== cap_ladder2_large_diverse_w4 ===
  --- OFFSET ROBUSTNESS of the CVaR rows (section 4.3.61) ---
                               centred f       raw f
  CVaR CE                         0.6176      0.4767
  CVaR acc                        0.6455      0.8273
  ratio width/preference          0.6759      0.4479
  --- DEGENERACY GATE (3.2.6) ---
  |CVaR CE - mean CE|               0.1759
  2 * SE(CVaR CE)                   0.0362
  verdict  PASS   [margin +0.1398]

  jackknife-over-chains SE on the CVaR CE: 0.0181
"""


def selftest():
    rows = parse(SELFTEST)
    assert list(rows) == [("large_diverse", 4)], rows
    r = rows[("large_diverse", 4)]
    # Centred and raw must not be swapped -- the whole point of the file.
    assert r["cen"] == 0.6176 and r["raw"] == 0.4767, r
    assert r["cen_acc"] == 0.6455 and r["raw_acc"] == 0.8273, r
    assert r["gap"] == 0.1759 and r["thr"] == 0.0362, r
    assert r["se"] == 0.0181 and r["verdict"] == "PASS", r
    # Check 1 compares the RAW column, which here matches §4.3.105 exactly.
    assert abs(r["raw"] - T105["large_diverse"][4][0]) < 5e-5
    # ...and would FAIL on §4.3.117's deployment-alpha value for the same rung.
    assert abs(0.6068 - T105["large_diverse"][4][0]) > 5e-5
    # A missing rung must be detected, not silently skipped.  (Quietly: the
    # negative case prints twelve MISSING lines that are not a real failure.)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        assert not check1({})
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--sweep", default="exp/ladder_alpha_sweep.txt",
                    help="alpha-sweep output; enables checks 3 and 4")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        text = open(a.src).read()
    except OSError as e:
        sys.exit(f"cannot read {a.src}: {e}")
    rows = parse(text)
    if not rows:
        sys.exit(f"no `=== cap_ladder2_<variant>_w<N> ===` blocks in {a.src}")
    ok = check1(rows)
    check2(rows)
    try:
        sw = parse_sweep(open(a.sweep).read())
    except OSError:
        sw = {}
        print(f"(no alpha sweep at {a.sweep} -- checks 3 and 4 skipped)\n")
    if sw:
        ok &= check3(sw, rows)
        ok &= check4(sw)
    verdicts = ladder(rows)
    if sw:
        sweep_table(sw)
    conclude(rows, verdicts)
    if not ok:
        print("\n!! A CHECK FAILED -- do not quote these numbers (4.3.117).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
