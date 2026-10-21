#!/usr/bin/env python
"""Wall-clock of ONE IQL labelling pass (gp_reward-priors HANDOFF §14.2 item 1: the NMR
cost measurement).  Runs on the box (needs d4rl).  For each model given, labels the full
D4RL dataset exactly as iql_eval.train() does, and reports seconds and transitions per
second:
    NMR  load_nmr_reward_model + qlearning_dataset_pt (the fixed window labeller)
    PT   load_pt_reward_model  + qlearning_dataset_pt (same labeller: the comparison)
    MR   load_mlp_reward_model + qlearning_dataset_mr (per transition)

    cd ~/iqlpref && python gp_reward-priors/scripts_nmr/time_labelling.py \\
        --env antmaze-medium-play-v2 --nmr-dir <dir> [--pt-dir <dir>] [--mr-dir <dir>]
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # ~/iqlpref
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gp_reward-priors"))
sys.path.insert(0, os.path.join(ROOT, "algorithms", "offline"))


def timed(label, fn, n_ref, sync):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    dt = time.perf_counter() - t0
    n = len(out["rewards"])
    print(f"  {label:4s} {dt:8.1f} s   {n:,} transitions labelled   {n / dt:,.0f} transitions/s   "
          f"reward range [{out['rewards'].min():.4g}, {out['rewards'].max():.4g}]")
    if n != n_ref:
        print(f"       note: {label} kept {n:,} rows, the sequence labeller {n_ref:,}")
    return dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="antmaze-medium-play-v2")
    ap.add_argument("--nmr-dir", required=True)
    ap.add_argument("--pt-dir")
    ap.add_argument("--mr-dir")
    ap.add_argument("--query-length", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import d4rl  # noqa: F401
    import gym
    import torch
    import iql_eval as Q

    sync = torch.cuda.synchronize if a.device.startswith("cuda") else (lambda: None)
    env = gym.make(a.env)
    sdim, adim = env.observation_space.shape[0], env.action_space.shape[0]
    t0 = time.perf_counter()
    data = env.get_dataset()
    print(f"{a.env}: {len(data['rewards']):,} rows loaded in {time.perf_counter() - t0:.1f} s; "
          f"device {a.device}; query_length {a.query_length}")
    n_ref = len(Q.qlearning_dataset_pt(env, _Zero(torch), 1, dataset=data)["rewards"])   # rows kept

    res = {}
    m = Q.load_nmr_reward_model(os.path.expanduser(a.nmr_dir), sdim, adim, device=a.device)
    res["NMR"] = timed("NMR", lambda: Q.qlearning_dataset_pt(env, m, a.query_length, dataset=data), n_ref, sync)
    del m
    if a.pt_dir:
        m = Q.load_pt_reward_model(os.path.expanduser(a.pt_dir), device=a.device)
        res["PT"] = timed("PT", lambda: Q.qlearning_dataset_pt(env, m, a.query_length, dataset=data), n_ref, sync)
        del m
    if a.mr_dir:
        m = Q.load_mlp_reward_model(os.path.expanduser(a.mr_dir), device=a.device)
        res["MR"] = timed("MR", lambda: Q.qlearning_dataset_mr(env, m, dataset=data), n_ref, sync)
        del m
    print("\n  as a share of one IQL run (~5.7 h = 20,520 s): "
          + ", ".join(f"{k} {v / 20520:.2%}" for k, v in res.items()))
    return 0


class _Zero:
    """A stand-in model for counting the rows the labeller keeps (no inference cost)."""
    def __init__(self, torch):
        self.torch = torch
        self.p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self.p])

    def __call__(self, s, a, t, m):
        return {"value": self.torch.zeros(s.shape[0], 1, s.shape[1], 1, device=s.device)}, None


if __name__ == "__main__":
    sys.exit(main())
