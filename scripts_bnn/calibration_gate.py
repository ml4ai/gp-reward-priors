#!/usr/bin/env python
"""calibration_gate.py -- the H3 calibration gate (handoff §13.10, Chapter 3 H3).

H3: "posterior spread and CVaR magnitude predict reward error", validated on
held-out PREFERENCES before any policy result is trusted; a domain that fails is
reported, and its policy numbers are conditional.  Falsifier: "uncertainty
uninformative about error".

THE RULE (fixed 2026-10-14 by the user, before any calibration number existed)
  data       each BNN model's own seed's VALIDATION split, always the base
             (uncorrupted) file.  Seeds 1-10.  The BNN selects no checkpoint and
             validation selected nothing at seeds 1-10, so it is held out.
  error      a pair is WRONG when the posterior-predictive preference
             p = mean over draws of sigma(Phi_1 - Phi_2) puts > 0.5 on the
             segment the label does not prefer.
  quantities per pair, both GATED:
             spread   sd over draws of the pair's logit difference Phi_1 - Phi_2
             penalty  the CVaR penalty the deployed reward applies to the pair:
                      (mean - CVaR) pooled over segment 1 plus the same over
                      segment 2, CVaR at conservatism 0.95 (tail fraction 0.05),
                      per (s,a), on centred draws -- exactly as deployed.
  statistic  per seed, the AUROC with which the quantity separates WRONG pairs
             from right ones (Mann-Whitney, ties counted one half).  0.5 is
             uninformative.
  pass       per (variant, level): the exact 95% bootstrap interval over the ten
             seeds (§12.5's enumeration) of the mean AUROC lies ENTIRELY above
             0.5, for spread AND for penalty.
  unit       variant x level: the anchor and each sparsity level separately.  A
             unit that FAILS makes that cell's policy numbers conditional.
  undefined  a seed with no wrong pair (or no right pair) has no AUROC; its unit
             reads UNDEFINED and is treated as not passed, and disclosed.
  reported, never gated: accuracy, predictive cross-entropy, ECE (10
             equal-width bins) and the reliability table, for the predictive
             preference and for the deployed CVaR preference.

Two steps:
  measure   ON THE BOX (needs the chains): one JSON per model of per-pair arrays.
      python scripts_bnn/calibration_gate.py measure --cond anchor --device cuda
      python scripts_bnn/calibration_gate.py measure --cond r16 --device cuda
  check     here (needs wandb): each measurement reproduces the metrics its
            training run logged on the same split, before anything is gated.
      python scripts_bnn/calibration_gate.py check --conds anchor
  gate      anywhere: reads the JSONs, applies the rule.
      python scripts_bnn/calibration_gate.py gate --conds "anchor r16 r32 r64 r128"
  --selftest
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import experiment_conditions as E                          # noqa: E402

CONSERVATISM = 0.95                 # the deployed level (bnn_alpha)
TAIL = 1.0 - CONSERVATISM           # tail fraction, as diagnose_sampling_tail's cvar_ce
N_BINS = 10
SEEDS = list(E.EVAL_SEEDS)
DEFAULT_OUT = os.path.join(ROOT, "exp", "calibration")
ANCHOR = "anchor"


def model_dir_name(variant, cond, seed):
    return f"{E.model_base(variant, 'bnn', None if cond == ANCHOR else cond)}_{seed}"


# --------------------------------------------------------------------------
# measure (box)
# --------------------------------------------------------------------------
def pair_arrays(P1, P2, am1, am2, y1, bt_pool="mean", tail=TAIL):
    """Per-pair quantities from per-step draw predictions.

    P1, P2  [S, B, T] predictions of every draw at every step of each segment
    am1/2   [B, T] attention masks;  y1 [B] label mass on segment 1.
    Centring (handoff item F, §4.3.114): each draw's offset is its mean over all
    points of both segments, removed before the per-(s,a) tail selection -- the
    same convention as diagnose_sampling_tail.cvar_ce.
    """
    from optbnn.utils.util import bt_pool_logit_np
    P1 = np.asarray(P1, dtype=np.float64)
    P2 = np.asarray(P2, dtype=np.float64)
    S = P1.shape[0]
    off = (P1.sum(axis=(1, 2)) + P2.sum(axis=(1, 2))) / (P1[0].size + P2[0].size)
    P1 = P1 - off[:, None, None]
    P2 = P2 - off[:, None, None]
    f1 = np.stack([bt_pool_logit_np(P1[s] * am1, am1, bt_pool) for s in range(S)])
    f2 = np.stack([bt_pool_logit_np(P2[s] * am2, am2, bt_pool) for s in range(S)])
    dlt = f1 - f2                                            # [S, B]
    p_pred = (1.0 / (1.0 + np.exp(-dlt))).mean(axis=0)
    spread = dlt.std(axis=0, ddof=1)
    k = max(1, int(math.floor(tail * S + 1e-9)))
    c1 = np.partition(P1, k - 1, axis=0)[:k].mean(axis=0)   # CVaR per (s,a): lowest k draws
    c2 = np.partition(P2, k - 1, axis=0)[:k].mean(axis=0)
    m1, m2 = P1.mean(axis=0), P2.mean(axis=0)
    g1m, g2m = bt_pool_logit_np(m1 * am1, am1, bt_pool), bt_pool_logit_np(m2 * am2, am2, bt_pool)
    g1c, g2c = bt_pool_logit_np(c1 * am1, am1, bt_pool), bt_pool_logit_np(c2 * am2, am2, bt_pool)
    return {"y1": [float(x) for x in y1], "p_pred": p_pred.tolist(), "spread": spread.tolist(),
            "penalty": ((g1m - g1c) + (g2m - g2c)).tolist(),
            "d_mean": (g1m - g2m).tolist(), "d_cvar": (g1c - g2c).tolist(),
            "n_draws": int(S), "k_tail": int(k)}


def measure_one(run_dir, device="cpu", chunk_pairs=64):
    import torch
    import diagnose_sampling_tail as D
    from optbnn.bnn.nets.mlp import MLP
    from optbnn.utils import util
    cfg = D._load_run_config(run_dir)
    val = cfg.get("val_dataset")
    if not val or "/noise/" in val or "/reduction/" in val:
        raise RuntimeError(f"{run_dir}: val_dataset {val!r} is not a base validation file")
    X, y = util.load_pref_data(val, training_ratio=1.0)
    B, _, T, d_dim = X.shape
    obs = d_dim - 1
    am1 = X[:, 0, :, obs].astype(np.float64)
    am2 = X[:, 1, :, obs].astype(np.float64)
    x1 = torch.from_numpy(X[:, 0, :, :obs].reshape(-1, obs).astype(np.float32)).to(device)
    x2 = torch.from_numpy(X[:, 1, :, :obs].reshape(-1, obs).astype(np.float32)).to(device)
    net = MLP(input_dim=obs, output_dim=1, hidden_dims=[cfg["width"]] * cfg["depth"],
              activation_fn="relu").to(device)
    net.eval()
    chain_ids = D._resolve_chain_ids(run_dir, None, None, cfg["num_chains"])
    blocks1, blocks2 = [], []
    for cid in chain_ids:
        for w in D._load_chain_weights(run_dir, cid, device):
            with torch.no_grad():
                for p, a in zip(net.parameters(), w):
                    p.copy_(torch.from_numpy(a).to(device))
                o1 = np.empty((B, T), np.float32)
                o2 = np.empty((B, T), np.float32)
                for s in range(0, B, chunk_pairs):
                    e = min(s + chunk_pairs, B)
                    o1[s:e] = net(x1[s * T:e * T]).cpu().numpy().reshape(e - s, T)
                    o2[s:e] = net(x2[s * T:e * T]).cpu().numpy().reshape(e - s, T)
            blocks1.append(o1)
            blocks2.append(o2)
    out = pair_arrays(np.stack(blocks1), np.stack(blocks2), am1, am2,
                      np.asarray(y, dtype=np.float64)[:, 0], cfg.get("bt_pool", "mean"))
    out.update({"run_dir": os.path.abspath(run_dir), "val_dataset": val,
                "seed": cfg.get("seed"), "num_chains": len(chain_ids),
                "width": cfg["width"], "depth": cfg["depth"], "n_pairs": int(B)})
    return out


def cmd_measure(a):
    os.makedirs(a.out_dir, exist_ok=True)
    conds = a.cond.split()
    for cond in conds:
        if cond != ANCHOR:
            E.parse(cond)
        for v in a.variants.split():
            for s in [int(x) for x in a.seeds.split()]:
                name = model_dir_name(v, cond, s)
                dst = os.path.join(a.out_dir, f"{name}.json")
                if os.path.exists(dst) and not a.force:
                    print(f"  {name}: exists, skipped")
                    continue
                src = os.path.join(a.models_root, name)
                if not os.path.isdir(os.path.join(src, "sampling_f")):
                    print(f"  {name}: NO MODEL at {src}")
                    continue
                r = measure_one(src, a.device)
                if r["seed"] != s:
                    raise RuntimeError(f"{name}: config seed {r['seed']} != {s}")
                with open(dst + ".tmp", "w") as f:
                    json.dump(r, f)
                os.replace(dst + ".tmp", dst)
                acc = np.mean((np.array(r["p_pred"]) > 0.5) == (np.array(r["y1"]) > 0.5))
                print(f"  {name}: {r['n_pairs']} pairs x {r['n_draws']} draws "
                      f"({r['num_chains']} chains)  accuracy {acc:.3f}  -> {dst}")
    return 0


# --------------------------------------------------------------------------
# gate (anywhere)
# --------------------------------------------------------------------------
def auroc(score, wrong):
    """P(score of a wrong pair > score of a right pair), ties 1/2.  None if a
    class is empty."""
    score = np.asarray(score, dtype=np.float64)
    wrong = np.asarray(wrong, dtype=bool)
    nw, nr = int(wrong.sum()), int((~wrong).sum())
    if nw == 0 or nr == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(score)
    return float((r[wrong].sum() - nw * (nw + 1) / 2) / (nw * nr))


def ece(p, y, n_bins=N_BINS):
    """Expected calibration error of P(segment 1 preferred), equal-width bins;
    and the reliability table [(lo, hi, n, mean p, observed)]."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    tot, table = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.sum() / p.size * abs(p[m].mean() - y[m].mean())
            table.append((edges[b], edges[b + 1], int(m.sum()), float(p[m].mean()),
                          float(y[m].mean())))
    return float(tot), table


def seed_stats(r):
    y = np.asarray(r["y1"], float)
    p = np.asarray(r["p_pred"], float)
    pc = 1.0 / (1.0 + np.exp(-np.asarray(r["d_cvar"], float)))
    wrong = (p > 0.5) != (y > 0.5)
    eps = 1e-12
    ce = float(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean())
    ce_c = float(-(y * np.log(pc + eps) + (1 - y) * np.log(1 - pc + eps)).mean())
    return {"auroc_spread": auroc(r["spread"], wrong), "auroc_penalty": auroc(r["penalty"], wrong),
            "acc": float((~wrong).mean()), "n_wrong": int(wrong.sum()), "n": int(y.size),
            "ce": ce, "ce_cvar": ce_c, "ece": ece(p, y)[0], "ece_cvar": ece(pc, y)[0],
            "acc_cvar": float(((pc > 0.5) == (y > 0.5)).mean())}


def gate_unit(records):
    """records: {seed: measurement}.  -> dict(verdict, ...)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_readout", os.path.join(os.path.dirname(ROOT), "results", "eval_readout.py"))
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    missing = [s for s in SEEDS if s not in records]
    if missing:
        return {"verdict": f"INCOMPLETE (no measurement for seeds {missing})"}
    st = {s: seed_stats(records[s]) for s in SEEDS}
    out = {"seeds": st}
    undefined = [s for s in SEEDS if st[s]["auroc_spread"] is None or st[s]["auroc_penalty"] is None]
    for q in ("spread", "penalty"):
        vals = [st[s][f"auroc_{q}"] for s in SEEDS]
        if undefined:
            out[q] = None
            continue
        b = R.exact_bootstrap(vals)
        out[q] = {"mean": b["mean"], "lower": b["lower"], "upper": b["upper"],
                  "checks": R.self_checks(b, np.asarray(vals))}
    if undefined:
        out["verdict"] = f"UNDEFINED (no wrong or no right pair on seeds {undefined}) -- not passed"
    elif out["spread"]["checks"] or out["penalty"]["checks"]:
        out["verdict"] = "SELF-CHECK FAILED"
    else:
        ok = {q: out[q]["lower"] > 0.5 for q in ("spread", "penalty")}
        out["verdict"] = "PASS" if all(ok.values()) else "FAIL (" + ", ".join(
            f"{q} interval reaches 0.5" for q, good in ok.items() if not good) + ")"
    pooled_p = np.concatenate([records[s]["p_pred"] for s in SEEDS])
    pooled_y = np.concatenate([records[s]["y1"] for s in SEEDS])
    out["ece_pooled"], out["reliability"] = ece(pooled_p, pooled_y)
    return out


def load_records(in_dir, cond, variant):
    recs = {}
    for s in SEEDS:
        f = os.path.join(in_dir, f"{model_dir_name(variant, cond, s)}.json")
        if os.path.exists(f):
            with open(f) as fh:
                recs[s] = json.load(fh)
    return recs


def cmd_gate(a, out=print):
    out("=" * 100)
    out("H3 CALIBRATION GATE (§13.10) -- BNN uncertainty vs held-out preference errors, seeds 1-10")
    out("PASS = exact 95% seed-bootstrap interval of mean AUROC above 0.5, for spread AND penalty")
    out("=" * 100)
    fails = 0
    for cond in a.conds.split():
        out(f"\n[{cond}]")
        for v in a.variants.split():
            g = gate_unit(load_records(a.in_dir, cond, v))
            out(f"  {v:15s} {g['verdict']}")
            fails += g["verdict"] != "PASS"
            if "seeds" not in g:
                continue
            for q in ("spread", "penalty"):
                if g.get(q):
                    out(f"      AUROC {q:8s} {g[q]['mean']:.3f}  95% exact [{g[q]['lower']:.3f}, "
                        f"{g[q]['upper']:.3f}]")
            st = g["seeds"]
            acc = np.mean([st[s]["acc"] for s in SEEDS])
            out(f"      reported, not gated: accuracy {acc:.3f} (CVaR "
                f"{np.mean([st[s]['acc_cvar'] for s in SEEDS]):.3f}); CE "
                f"{np.mean([st[s]['ce'] for s in SEEDS]):.3f} (CVaR "
                f"{np.mean([st[s]['ce_cvar'] for s in SEEDS]):.3f}, log 2 = 0.693); ECE per seed "
                f"{np.mean([st[s]['ece'] for s in SEEDS]):.3f} (CVaR "
                f"{np.mean([st[s]['ece_cvar'] for s in SEEDS]):.3f}); ECE pooled {g['ece_pooled']:.3f}; "
                f"wrong pairs per seed {min(st[s]['n_wrong'] for s in SEEDS)}-"
                f"{max(st[s]['n_wrong'] for s in SEEDS)}")
            out("      per seed AUROC spread/penalty: " + "  ".join(
                f"s{s} {st[s]['auroc_spread']:.2f}/{st[s]['auroc_penalty']:.2f}"
                if st[s]["auroc_spread"] is not None and st[s]["auroc_penalty"] is not None
                else f"s{s} --" for s in SEEDS))
            if a.reliability:
                out("      reliability (pooled): " + "  ".join(
                    f"[{lo:.1f},{hi:.1f}) n{n} p{mp:.2f} obs{ob:.2f}"
                    for lo, hi, n, mp, ob in g["reliability"]))
    out(f"\n{fails} unit(s) not PASS.  A FAIL or UNDEFINED unit's policy numbers are conditional "
        f"(H3); an INCOMPLETE unit is not read yet.")
    return 0


def cmd_check(a, out=print):
    """End-to-end: each measurement's predictive CE, CVaR CE and CVaR accuracy at
    0.95 against the values the training run itself logged on the same split
    (val_cvar_predictive_ce, val_cvar_ce_c0p95, val_cvar_acc_c0p95)."""
    import production_readout as PR
    prod, _ = PR.fetch("bnn", a.since)
    by_out = {}
    for r in prod:
        if r["state"] == "finished":
            by_out.setdefault(os.path.basename(os.path.normpath(str(r["config"].get("OUT_DIR")))),
                              []).append(r)
    n = bad = 0
    for cond in a.conds.split():
        for v in a.variants.split():
            for s, rec in sorted(load_records(a.in_dir, cond, v).items()):
                name = model_dir_name(v, cond, s)
                runs = by_out.get(name, [])
                if len(runs) != 1:
                    out(f"  {name}: {len(runs)} finished wandb runs -- not checked")
                    bad += 1
                    continue
                sm, st = runs[0]["summary"], seed_stats(rec)
                rows = (("predictive CE", st["ce"], sm.get("val_cvar_predictive_ce")),
                        ("CVaR CE 0.95", st["ce_cvar"], sm.get("val_cvar_ce_c0p95")),
                        ("CVaR acc 0.95", st["acc_cvar"], sm.get("val_cvar_acc_c0p95")))
                diffs = [(k, x, y) for k, x, y in rows if y is None or abs(x - y) > a.tol]
                n += 1
                bad += bool(diffs)
                out(f"  {name}: " + ("matches the logged values" if not diffs else
                                     "DIFFERS " + "; ".join(f"{k} {x:.6f} vs logged {y}"
                                                            for k, x, y in diffs)))
    out(f"\n{n} measurement(s) checked against wandb, {bad} problem(s)")
    return 1 if bad else 0


# --------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(0)
    # AUROC: perfect, inverted, uninformative, ties, empty class
    assert auroc([3, 4, 1, 2], [True, True, False, False]) == 1.0
    assert auroc([1, 2, 3, 4], [True, True, False, False]) == 0.0
    assert auroc([1, 1, 1, 1], [True, False, True, False]) == 0.5
    assert auroc([1, 2], [False, False]) is None and auroc([1, 2], [True, True]) is None
    sc, wr = rng.normal(size=60), rng.random(60) < 0.3
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in sc[wr] for b in sc[~wr]])
    assert abs(auroc(sc, wr) - brute) < 1e-12
    # ECE: perfectly calibrated in expectation, and a known miscalibration
    p = np.full(1000, 0.8)
    y = np.r_[np.ones(800), np.zeros(200)]
    assert abs(ece(p, y)[0]) < 1e-12
    assert abs(ece(np.full(10, 0.9), np.r_[np.ones(5), np.zeros(5)])[0] - 0.4) < 1e-12
    # pair_arrays on synthetic draws: the CVaR penalty is >= 0, spread > 0, and a
    # pair whose draws disagree in sign is the uncertain one
    S, B, T = 400, 6, 5
    am = np.ones((B, T))
    base = rng.normal(size=(B, T))
    noise = np.r_[np.full(3, 0.05), np.full(3, 2.0)][:, None]
    P1 = base[None] + rng.normal(size=(S, B, T)) * noise[None]
    P2 = np.zeros((S, B, T)) + rng.normal(size=(S, B, T)) * 0.05
    try:
        from optbnn.utils.util import bt_pool_logit_np  # noqa: F401  the real one
    except ImportError:                                  # a stand-in, same formula
        import types
        u = types.ModuleType("optbnn.utils.util")

        def bt_pool_logit_np(pred_masked, mask, mode="mean"):
            s = np.nansum(pred_masked, axis=1)
            return s if mode == "sum" else s / np.clip(np.nansum(mask, axis=1), 1.0, None)
        u.bt_pool_logit_np = bt_pool_logit_np
        sys.modules["optbnn.utils"] = types.ModuleType("optbnn.utils")
        sys.modules["optbnn.utils.util"] = u
    r = pair_arrays(P1, P2, am, am, np.ones(B))
    pen, spr = np.array(r["penalty"]), np.array(r["spread"])
    assert (pen >= -1e-12).all() and (spr > 0).all()
    assert spr[3:].min() > spr[:3].max() and pen[3:].min() > pen[:3].max()
    assert r["k_tail"] == 20 and r["n_draws"] == S
    # a constant offset added to every point of a draw changes nothing (centring)
    shift = rng.normal(size=S)[:, None, None] * 10
    r2 = pair_arrays(P1 + shift, P2 + shift, am, am, np.ones(B))
    for k in ("p_pred", "spread", "penalty", "d_mean", "d_cvar"):
        assert np.allclose(r[k], r2[k], atol=1e-9), k

    # gate_unit end to end: informative uncertainty passes, uninformative fails
    def fake(informative, n=80):
        recs = {}
        for s in SEEDS:
            y = (rng.random(n) < 0.5).astype(float)
            unc = rng.gamma(2.0, 0.5, n)
            p_right = 1.0 / (1.0 + np.exp(-(2.5 - (1.2 * unc if informative else 0.0))))
            right = rng.random(n) < p_right
            conf = 0.5 + 0.4 * rng.random(n)
            p = np.where(right == (y > 0.5), conf, 1 - conf)
            recs[s] = {"y1": y.tolist(), "p_pred": p.tolist(), "spread": unc.tolist(),
                       "penalty": (unc * 0.8 + rng.normal(0, 0.05, n)).tolist(),
                       "d_cvar": np.log(p / (1 - p)).tolist()}
        return recs
    assert gate_unit(fake(True))["verdict"] == "PASS"
    assert gate_unit(fake(False))["verdict"].startswith("FAIL")
    nine = fake(True)
    del nine[7]
    assert gate_unit(nine)["verdict"].startswith("INCOMPLETE")
    one = fake(True)
    one[4]["p_pred"] = [1.0 if y > 0.5 else 0.0 for y in one[4]["y1"]]   # no wrong pair
    assert gate_unit(one)["verdict"].startswith("UNDEFINED")
    # names
    assert model_dir_name("large_play", ANCHOR, 3) == "antmaze_large_play_bnn_eval_3"
    assert model_dir_name("large_play", "r16", 3) == "antmaze_large_play_bnn_r16_eval_3"
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("measure", help="on the box: per-pair arrays for each model")
    m.add_argument("--cond", default=ANCHOR, help='"anchor" or e.g. "r16" (space-separated ok)')
    m.add_argument("--variants", default=" ".join(E.VARIANTS))
    m.add_argument("--seeds", default=" ".join(str(s) for s in SEEDS))
    m.add_argument("--models-root",
                   default=os.path.join(os.path.dirname(ROOT), "exp", "reward_learning"))
    m.add_argument("--out-dir", default=DEFAULT_OUT)
    m.add_argument("--device", default="cpu")
    m.add_argument("--force", action="store_true")
    g = sub.add_parser("gate", help="apply the registered rule to the measurements")
    g.add_argument("--conds", default=ANCHOR)
    g.add_argument("--variants", default=" ".join(E.VARIANTS))
    g.add_argument("--in-dir", default=DEFAULT_OUT)
    g.add_argument("--reliability", action="store_true")
    c = sub.add_parser("check", help="measurements against the training runs' logged metrics")
    c.add_argument("--conds", default=ANCHOR)
    c.add_argument("--variants", default=" ".join(E.VARIANTS))
    c.add_argument("--in-dir", default=DEFAULT_OUT)
    c.add_argument("--since", default="2026-09-24T19:00:00")
    c.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "check":
        return cmd_check(a)
    if a.cmd == "measure":
        return cmd_measure(a)
    if a.cmd == "gate":
        return cmd_gate(a)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
