#!/usr/bin/env python
"""ties_readout.py -- the ties ablation, read on the reward models (handoff §14.2 item 3,
§14.21).  DESCRIPTIVE: intervals, no verdict.

Two arms per (family, variant, seed), sharing every non-tie pair:
    tie-free  the anchor model           exp/reward_learning/antmaze_<v>_<fam>_eval_<s>
    ties      the ties-kept model        exp/reward_learning/antmaze_<v>_<fam>_ties_eval_<s>
Metrics, both arms scored on the SAME held-out pairs of that seed:
    acc_tiefree   preference accuracy on the base (tie-free) validation split
    ce_tiefree    cross-entropy on it
    ce_ties       cross-entropy on the tied validation pairs (ties/..._valties_<s>.hdf5),
                  target [0.5, 0.5]; its floor is log 2 = 0.693, reached only by a
                  model that is exactly indifferent on every tied pair
    dev_ties      mean |P(first preferred) - 0.5| on the tied pairs: how far from
                  indifferent the model is where the labeller was
The pooled logit is the masked MEAN of per-step rewards (bt_pool "mean"), as in
training.  Families: MR, PT, NMR (§14.2); variants: large_play, medium_diverse.

    measure (on the box; needs the models and iql_eval's loaders):
        cd ~/iqlpref && python gp_reward-priors/scripts_bnn/ties_readout.py measure
    report (anywhere, from the JSON):
        python gp_reward-priors/scripts_bnn/ties_readout.py report
    python gp_reward-priors/scripts_bnn/ties_readout.py --selftest
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.dirname(HERE)                                   # gp_reward-priors
ROOT = os.path.dirname(SUB)                                   # ~/iqlpref
sys.path.insert(0, HERE)
sys.path.insert(0, SUB)

import experiment_conditions as E                             # noqa: E402

FAMILIES = E.TIES_FAMILIES
VARIANTS = E.TIES_VARIANTS
SEEDS = E.EVAL_SEEDS
ARMS = ("tiefree", "ties")
OUT = os.path.join(SUB, "calibration_results", "ties", "ties_measurements.json")


def model_dir(models_root, family, variant, seed, arm):
    return os.path.join(models_root, f"{E.model_base(variant, family, 'ties' if arm == 'ties' else None)}_{seed}")


def pooled_logits(step_rewards_1, step_rewards_2, mask_1, mask_2):
    """(N, 2) logits from per-step rewards (N, T) and 0/1 masks: masked means."""
    p1 = (step_rewards_1 * mask_1).sum(1) / np.clip(mask_1.sum(1), 1.0, None)
    p2 = (step_rewards_2 * mask_2).sum(1) / np.clip(mask_2.sum(1), 1.0, None)
    return np.stack([p1, p2], axis=1)


def metrics(logits, labels):
    """Soft-label cross-entropy (mean), accuracy on non-tie pairs, |P - 0.5| (mean)."""
    m = logits.max(1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(1, keepdims=True))
    ce = float(-(labels * logp).sum(1).mean())
    p_first = np.exp(logp[:, 0])
    tie = np.isclose(labels[:, 0], 0.5)
    acc = float((logits[~tie].argmax(1) == labels[~tie].argmax(1)).mean()) if (~tie).any() else float("nan")
    return {"ce": ce, "acc": acc, "dev": float(np.abs(p_first - 0.5).mean()), "n": int(len(labels))}


def step_rewards(family, model, data, torch, device):
    """Per-step rewards (N, T) for both segments of every pair in a Pref_H5Dataset."""
    out1, out2 = [], []
    with torch.no_grad():
        for i in range(0, len(data), 256):
            sl = slice(i, i + 256)
            for seg, out in ((("states", "actions", "timesteps", "attn_mask"), out1),
                             (("states_2", "actions_2", "timesteps_2", "attn_mask_2"), out2)):
                s, a, t, m = (torch.as_tensor(getattr(data, k)[sl]).to(device) for k in seg)
                B, T, _ = s.shape
                if family == "mr":
                    r = model(torch.cat([s, a], -1).reshape(B * T, -1)).reshape(B, T)
                else:                                                # pt, nmr
                    r = model(s, a, t.to(torch.int), m)[0]["weighted_sum"].reshape(B, T)
                out.append(r.float().cpu().numpy())
    return np.concatenate(out1), np.concatenate(out2)


def cmd_measure(a):
    import torch
    sys.path.insert(0, os.path.join(ROOT, "algorithms", "offline"))
    import iql_eval as Q
    from optbnn.utils import util
    res = json.load(open(a.out)) if os.path.exists(a.out) and not a.force else {}
    for fam in FAMILIES:
        for v in VARIANTS:
            env = E.env_of(v)
            for s in SEEDS:
                sd = os.path.join(a.data_root, env, "eval", f"seed_{s}")
                sets = {"tiefree": util.Pref_H5Dataset(os.path.join(sd, f"{env}_pref_val_{s}.hdf5")),
                        "ties": util.Pref_H5Dataset(os.path.join(sd, "ties", f"{env}_pref_valties_{s}.hdf5"))}
                for arm in ARMS:
                    key = f"{fam}|{v}|{s}|{arm}"
                    d = model_dir(a.models_root, fam, v, s, arm)
                    if key in res:
                        continue
                    if not os.path.isfile(os.path.join(d, "best_model.pt")):
                        print(f"  {key}: no model at {d}")
                        continue
                    if fam == "mr":
                        m = Q.load_mlp_reward_model(d, device=a.device)
                    elif fam == "pt":
                        m = Q.load_pt_reward_model(d, device=a.device)
                    else:
                        sdim, adim = sets["tiefree"].states.shape[2], sets["tiefree"].actions.shape[2]
                        m = Q.load_nmr_reward_model(d, sdim, adim, device=a.device)
                    row = {}
                    for split, data in sets.items():
                        r1, r2 = step_rewards(fam, m, data, torch, a.device)
                        row[split] = metrics(pooled_logits(r1, r2, data.attn_mask, data.attn_mask_2), data.labels)
                    res[key] = {"acc_tiefree": row["tiefree"]["acc"], "ce_tiefree": row["tiefree"]["ce"],
                                "ce_ties": row["ties"]["ce"], "dev_ties": row["ties"]["dev"],
                                "n_tiefree": row["tiefree"]["n"], "n_ties": row["ties"]["n"], "dir": d}
                    print(f"  {key}: acc {res[key]['acc_tiefree']:.3f}  ce {res[key]['ce_tiefree']:.4f}  "
                          f"ce_ties {res[key]['ce_ties']:.4f}  dev_ties {res[key]['dev_ties']:.3f}")
                    del m
                    os.makedirs(os.path.dirname(a.out), exist_ok=True)
                    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"\n  {len(res)} measurement(s) in {a.out}")
    return 0


def cmd_report(a, out=print):
    sys.path.insert(0, os.path.join(ROOT, "results"))
    import eval_readout as R
    res = json.load(open(a.out))
    out("=" * 100)
    out("TIES ABLATION, reward models (handoff §14.2 item 3, §14.21).  DESCRIPTIVE.")
    out("Both arms on the same held-out pairs; ties - tie-free paired by seed; 95% exact bootstrap.")
    out("ce_ties floor: log 2 = 0.6931 (exactly indifferent on every tied pair).")
    out("=" * 100)
    for fam in FAMILIES:
        for v in VARIANTS:
            out(f"\n  {fam.upper():4s} {v}")
            for k in ("acc_tiefree", "ce_tiefree", "ce_ties", "dev_ties"):
                got = {arm: {s: res.get(f"{fam}|{v}|{s}|{arm}", {}).get(k) for s in SEEDS} for arm in ARMS}
                have = {arm: [s for s in SEEDS if got[arm][s] is not None] for arm in ARMS}
                cells = "   ".join(
                    f"{arm} {np.mean([got[arm][s] for s in have[arm]]):.4f} (n={len(have[arm])})" if have[arm]
                    else f"{arm} --" for arm in ARMS)
                both = [s for s in SEEDS if got["ties"][s] is not None and got["tiefree"][s] is not None]
                line = f"    {k:12s} {cells}"
                if len(both) == len(SEEDS):
                    d = np.array([got["ties"][s] - got["tiefree"][s] for s in SEEDS])
                    bs = R.exact_bootstrap(d)
                    line += f"   ties - tie-free {d.mean():+.4f} [{bs['lower']:+.4f}, {bs['upper']:+.4f}]"
                else:
                    line += f"   (paired difference when all {len(SEEDS)} seeds of both arms are in)"
                out(line)
    return 0


def selftest():
    rng = np.random.default_rng(0)
    N, T = 50, 7
    r1, r2 = rng.normal(size=(N, T)), rng.normal(size=(N, T))
    m1, m2 = np.ones((N, T)), np.ones((N, T))
    m2[:, 5:] = 0
    lg = pooled_logits(r1, r2, m1, m2)
    assert np.allclose(lg[:, 0], r1.mean(1)) and np.allclose(lg[:, 1], r2[:, :5].mean(1))
    # hard labels agreeing with the logits: accuracy 1; flipped: 0
    lab = np.eye(2)[lg.argmax(1)]
    assert metrics(lg, lab)["acc"] == 1.0 and metrics(lg, lab[:, ::-1])["acc"] == 0.0
    # ties: zero logits give CE exactly log 2 and deviation 0; ties never count in accuracy
    tie = np.full((N, 2), 0.5)
    z = metrics(np.zeros((N, 2)), tie)
    assert abs(z["ce"] - math.log(2)) < 1e-12 and z["dev"] == 0.0 and math.isnan(z["acc"])
    mix = np.concatenate([lab, tie])
    assert metrics(np.concatenate([lg, lg]), mix)["acc"] == 1.0
    # soft-label CE equals torch's CrossEntropyLoss with probability targets
    import torch
    t = torch.nn.CrossEntropyLoss()(torch.tensor(lg), torch.tensor(mix[:N] * 0.5 + 0.25)).item()
    assert abs(metrics(lg, mix[:N] * 0.5 + 0.25)["ce"] - t) < 1e-9, t
    assert model_dir("/m", "pt", "large_play", 3, "ties") == "/m/antmaze_large_play_pt_ties_eval_3"
    assert model_dir("/m", "mr", "medium_diverse", 10, "tiefree") == "/m/antmaze_medium_diverse_mr_eval_10"
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("measure")
    m.add_argument("--models-root", default=os.path.join(ROOT, "exp", "reward_learning"))
    m.add_argument("--data-root", default=os.path.join(SUB, "data", "antmaze"))
    m.add_argument("--out", default=OUT)
    m.add_argument("--device", default="cpu")
    m.add_argument("--force", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "measure":
        return cmd_measure(a)
    if a.cmd == "report":
        return cmd_report(a)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
