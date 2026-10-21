"""POST-HOC diagnostic of the anchor H3 gate (not part of the registered rule).

1. Sign sanity: plug-in accuracy from d_mean against the logged val_cvar_acc_c0.
2. Is raw spread scale-confounded?  Spearman(spread, |d_mean|), and the AUROC of
   the margin |d_mean| for errors (small margin should mean error: AUROC < 0.5).
3. A scale-free uncertainty: spread / |d_mean| (coefficient of variation of the
   pair's logit difference) and penalty / |d_mean|.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results"))                 # hold.py (HANDOFF §14.1)
import hold
hold.refuse("calibration_posthoc.py", "a diagnostic of the BNN's H3 gate")
sys.path.insert(0, sys.argv[1])
import calibration_gate as G

in_dir = os.path.join(sys.argv[1], "..", "calibration_results", "measurements")
for v in G.E.VARIANTS:
    rows = []
    for s in G.SEEDS:
        r = json.load(open(os.path.join(in_dir, f"{G.model_dir_name(v, 'anchor', s)}.json")))
        y = np.array(r["y1"]) > 0.5
        p = np.array(r["p_pred"])
        dm = np.array(r["d_mean"])
        sp, pen = np.array(r["spread"]), np.array(r["penalty"])
        wrong = (p > 0.5) != y
        rows.append({
            "acc_pred": (~wrong).mean(), "acc_plug": ((dm > 0) == y).mean(),
            "agree": ((dm > 0) == (p > 0.5)).mean(),
            "rho_spread_margin": spearmanr(sp, np.abs(dm))[0],
            "auc_margin": G.auroc(np.abs(dm), wrong),
            "auc_cv": G.auroc(sp / np.abs(dm), wrong),
            "auc_pen_rel": G.auroc(pen / np.abs(dm), wrong),
            "auc_spread": G.auroc(sp, wrong), "auc_pen": G.auroc(pen, wrong)})
    m = {k: np.mean([x[k] for x in rows]) for k in rows[0]}
    print(f"{v:15s} acc pred {m['acc_pred']:.3f} plug {m['acc_plug']:.3f} (signs agree "
          f"{m['agree']:.3f}) | rho(spread,|margin|) {m['rho_spread_margin']:+.2f} | AUROC: "
          f"margin {m['auc_margin']:.2f}  spread {m['auc_spread']:.2f}  spread/|margin| "
          f"{m['auc_cv']:.2f}  penalty {m['auc_pen']:.2f}  penalty/|margin| {m['auc_pen_rel']:.2f}")
r = json.load(open(os.path.join(in_dir, "antmaze_large_play_bnn_eval_3.json")))
print("large_play seed 3 plug-in accuracy",
      np.mean((np.array(r["d_mean"]) > 0) == (np.array(r["y1"]) > 0.5)),
      "(logged val_cvar_acc_c0 = 0.8148)")
