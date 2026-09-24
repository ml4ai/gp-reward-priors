#!/usr/bin/env python
"""Regenerate a BNN production config from a round-5 winner (handoff to-do 10, §4.3.130).

The production config and the sweep's BASE config are the same file,
`scripts_bnn/antmaze_<variant>_bnn_antmaze_eval.yaml`: `train_rewards.sh` trains
from it, and the sweep yaml points at it and overrides fields per trial.  So the
file on disk is NOT what the winning trial ran.  It still holds whatever it held
before the sweep (round-2 values), and the sweep supplied the rest.

The configuration the trial ACTUALLY ran is its wandb config.  This tool makes
the file equal to it, so that the §3.2.9 escalation (= the seed-0 production
model, §4.3.129) runs the selected configuration exactly, and its first 32
chains reproduce the sweep trial bit-for-bit.  Only these differ, by design:

  num_chains / chains_per_gpu   the escalated budget (default 128 @ 32/GPU)
  EXEMPT keys                   per-run identity or paths the launcher derives
                                (name, seed, OUT_DIR, data paths ...)
  None-valued keys              not written: they are the code defaults, and
                                `burn_in_lr` in particular must NEVER appear --
                                launch_hp_sweeps.sh's preflight rejects any
                                base config that mentions it (§3.7).

Lines whose value does not change are left byte-identical.  That matters:
the preflight's `centre_draws: true` check forbids a trailing comment on
that line.

Dry-run by default (prints the diff and the verification); --write applies it.

Usage:
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/make_production_config.py \\
        large_play q45qbz8h [--num-chains 128 --chains-per-gpu 32] [--write]
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/make_production_config.py --selftest
"""

import argparse
import difflib
import os
import re
import sys

import yaml

ENTITY, PROJECT = "champlin-university-of-arizona", "BNN-training"
HERE = os.path.dirname(os.path.abspath(__file__))

# Per-run identity, or paths the launcher/eval script derive per seed.  Pinning
# the *_dataset paths would make seeds 1-10 train on seed 0's split.
EXEMPT = {"name", "seed", "OUT_DIR", "config_path", "data_root", "project",
          "group", "train_dataset", "val_dataset", "test_dataset"}
NEVER_WRITE = {"burn_in_lr"}          # preflight-enforced absence (§3.7)
PREFLIGHT_CENTRE = re.compile(
    r"^[ \t]*centre_draws[ \t]*:[ \t]*(true|True)[ \t]*$", re.M)


def fmt(v):
    """A YAML scalar that yaml.safe_load reads back as the SAME type and value."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = repr(v)
        # PyYAML (YAML 1.1) reads '5e-05' as a STRING: the mantissa needs a dot.
        if "e" in s and "." not in s.split("e")[0]:
            m, e = s.split("e")
            s = f"{m}.0e{e}"
        return s
    s = str(v)
    return s if re.fullmatch(r"[A-Za-z0-9_./,:+-]+", s) and \
        yaml.safe_load(s) == s else repr(s)


def field_types(src_path):
    """{field: annotation} parsed from TrainConfig's SOURCE, no import needed.

    wandb stores an integral float as an int (1.0 -> 1), so its config cannot
    be trusted for types.  pyrallis is broken on the Mac's Python 3.14, so the
    dataclass cannot be imported here -- read the annotations as text instead.
    """
    src = open(src_path).read()
    body = re.search(r"class TrainConfig.*?\n(?=\S)", src, re.S).group(0)
    return {k: t.strip() for k, t in
            re.findall(r"^\s{4}(\w+)\s*:\s*([^=\n]+?)\s*=", body, re.M)}


def coerce(k, v, ann):
    """Cast wandb's value to the field's declared type (int<->float only)."""
    t = (ann or {}).get(k, "")
    if isinstance(v, bool) or v is None:
        return v
    if "float" in t and isinstance(v, int):
        return float(v)
    if re.fullmatch(r"(Optional\[)?int\]?", t) and isinstance(v, float) \
            and v.is_integer():
        return int(v)
    return v


def _same(a, b):
    """Equal as the TRAINING SCRIPT will see them.  An int in the file for a
    float field decodes to the same float (the trial ran from this very file),
    so 6838 and 6838.0 are the same value; bools are never numbers here."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return a == b and type(a) is type(b)


def build(text, ran, rid, num_chains, cpg, ann=None):
    """(new_text, changed, added).  Pure function of its inputs."""
    old = yaml.safe_load(text)
    target = {k: coerce(k, v, ann) for k, v in ran.items()
              if k not in EXEMPT and k not in NEVER_WRITE and v is not None}
    target["num_chains"], target["chains_per_gpu"] = num_chains, cpg

    lines = text.split("\n")
    changed, seen = [], set()
    for i, ln in enumerate(lines):
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(\s*:\s*)([^#]*?)(\s*)(#.*)?$", ln)
        if not m or m.group(1) not in target:
            continue
        k = m.group(1)
        seen.add(k)
        if _same(old.get(k), target[k]):
            continue                                     # byte-identical
        why = (f"round-5 escalation, §4.3.129 (was {fmt(old.get(k))})"
               if k in ("num_chains", "chains_per_gpu")
               else f"round-5 winner {rid} (was {fmt(old.get(k))})")
        if k == "width":
            why = "log2 exponent; " + why
        lines[i] = f"{k}: {fmt(target[k])}   # {why}"
        changed.append((k, old.get(k), target[k]))

    added = [k for k in sorted(target) if k not in seen]
    if added:
        lines += ["", "# ---- Pinned explicitly for production (handoff §4.3.130) ----",
                  f"# These were code defaults when {rid} ran.  Written out so the",
                  "# production run cannot drift from the selected trial if a",
                  "# default ever changes."]
        lines += [f"{k}: {fmt(target[k])}" for k in added]

    # Replace the leading comment header with a round-5 provenance block.
    j = next(i for i, ln in enumerate(lines) if ln.strip() and not ln.startswith("#"))
    header = [
        f"# Production config: ROUND-5 BNN winner {rid} (handoff §4.3.130).",
        "# Generated by scripts_bnn/make_production_config.py from the winning",
        "# trial's recorded wandb config, so every value below is what that trial",
        "# RAN, except by design:",
        f"#   num_chains {num_chains} / chains_per_gpu {cpg} -- the §3.2.9 escalation",
        "#     (§4.3.129).  Chains are deterministic in (seed, index), so chains",
        f"#     0-31 of a seed-0 run reproduce trial {rid} exactly.",
        "#   seed / OUT_DIR / data paths -- set per seed by the launcher.",
        "#   burn_in_lr -- deliberately ABSENT (burn-in inherits sghmc_lr, §3.7).",
        "# This file is also the sweep's base config; the sweep overrides every",
        "# field it varies, so editing it does not change a sweep re-run.",
        "# The previous (round-2 HYBRID) header and values are in git history.",
        "#",
    ]
    return "\n".join(header + lines[j:]), changed, added


def verify(new_text, old_text, ran, num_chains, cpg, ann=None):
    """List of failures (empty = pass)."""
    new, old = yaml.safe_load(new_text), yaml.safe_load(old_text)
    bad = []
    for k, v in ran.items():
        if k in EXEMPT or k in NEVER_WRITE or v is None:
            continue
        want = {"num_chains": num_chains, "chains_per_gpu": cpg}.get(
            k, coerce(k, v, ann))
        if k not in new:
            bad.append(f"{k}: missing (ran {v!r})")
        elif not _same(new[k], want):
            bad.append(f"{k}: file {new[k]!r} ({type(new[k]).__name__}) "
                       f"!= want {want!r} ({type(want).__name__})")
    for k in EXEMPT:
        if k in old and new.get(k) != old[k]:
            bad.append(f"{k}: exempt key changed {old[k]!r} -> {new.get(k)!r}")
    if "burn_in_lr" in new or re.search(r"^[ \t]*burn_in_lr[ \t]*:", new_text, re.M):
        bad.append("burn_in_lr present -- launch preflight would refuse")
    if not PREFLIGHT_CENTRE.search(new_text):
        bad.append("centre_draws line no longer matches the launch preflight")
    if "SUPERSEDED-ROUND1" in new_text:
        bad.append("SUPERSEDED-ROUND1 marker present -- train_rewards.sh would refuse")
    return bad


def selftest():
    text = ("# old header\n# more\nwidth: 9                 # log2 exponent\n"
            "num_chains: 8   # stage 3\nsghmc_lr: 0.0001\ncentre_draws: true\n"
            "seed: 1\nOUT_DIR: ./exp/x\n")
    ran = {"width": 5, "num_chains": 32, "sghmc_lr": 5e-05, "centre_draws": True,
           "seed": 0, "OUT_DIR": "./exp/x_0", "burn_in_lr": None,
           "cvar_ce_conservatism": 0.75, "use_cyclical_lr": True,
           "meas_sampling": "random", "train_dataset": "data/seed_0/t.hdf5"}
    new, changed, added = build(text, ran, "abc123", 128, 32)
    y = yaml.safe_load(new)
    assert y["width"] == 5 and y["num_chains"] == 128 and y["chains_per_gpu"] == 32, y
    assert y["sghmc_lr"] == 5e-05 and isinstance(y["sghmc_lr"], float), y  # dot guard
    assert y["seed"] == 1 and y["OUT_DIR"] == "./exp/x", y                  # exempt kept
    assert "burn_in_lr" not in y and "train_dataset" not in y, y            # never / exempt
    assert y["cvar_ce_conservatism"] == 0.75 and y["use_cyclical_lr"] is True, y
    assert y["meas_sampling"] == "random", y
    assert "\ncentre_draws: true\n" in new, "unchanged line must stay byte-identical"
    assert not verify(new, text, ran, 128, 32), verify(new, text, ran, 128, 32)
    assert "old header" not in new and "round-5 winner abc123" in new
    # verify() catches the failures it exists for
    assert verify(new + "\nburn_in_lr: 0.1\n", text, ran, 128, 32)
    assert verify(new.replace("centre_draws: true", "centre_draws: true  # x"),
                  text, ran, 128, 32)
    assert fmt(5e-05) == "5.0e-05" and yaml.safe_load(fmt(5e-05)) == 5e-05
    assert yaml.safe_load(fmt("0.0,0.5,0.75")) == "0.0,0.5,0.75"
    # wandb's int-for-integral-float artefact must NOT rewrite a float field
    t2 = ("map_eta: 1.0   # heat kernel\nclip: 100\n"
          "map_amp2: 6838           # CORRECTED note\n")
    ann = {"map_eta": "float", "clip_grad_norm_value": "Optional[float]",
           "cycle_length": "int"}
    ann["map_amp2"] = "float"
    new2, ch2, add2 = build(t2, {"map_eta": 1, "clip_grad_norm_value": 100,
                                 "cycle_length": 2000.0, "map_amp2": 6838.0},
                            "r", 128, 32, ann)
    assert "map_amp2: 6838           # CORRECTED note" in new2, new2   # comment kept
    y2 = yaml.safe_load(new2)
    assert "map_eta: 1.0   # heat kernel" in new2, new2          # untouched
    assert not any(k == "map_eta" for k, *_ in ch2), ch2
    assert y2["clip_grad_norm_value"] == 100.0 and \
        isinstance(y2["clip_grad_norm_value"], float), y2
    assert y2["cycle_length"] == 2000 and isinstance(y2["cycle_length"], int), y2
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant", nargs="?")
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--num-chains", type=int, default=128)
    ap.add_argument("--chains-per-gpu", type=int, default=32)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.variant and a.run_id):
        ap.error("variant and run_id are required")

    import wandb
    run = wandb.Api(timeout=60).run(f"{ENTITY}/{PROJECT}/{a.run_id}")
    ran = {k: v for k, v in dict(run.config).items() if not k.startswith("_")}
    if ran.get("antmaze_variant", "").replace("-", "_").find(a.variant) < 0:
        sys.exit(f"run {a.run_id} is {ran.get('antmaze_variant')!r}, not {a.variant}")

    path = os.path.join(HERE, f"antmaze_{a.variant}_bnn_antmaze_eval.yaml")
    old_text = open(path).read()
    ann = field_types(os.path.join(HERE, "run_bnn_training_antmaze_eval.py"))
    new_text, changed, added = build(old_text, ran, a.run_id,
                                     a.num_chains, a.chains_per_gpu, ann)

    print(f"=== {a.variant}: {path}")
    print(f"    winner {a.run_id} -> {len(changed)} value(s) changed, "
          f"{len(added)} key(s) pinned explicitly")
    for k, o, n in changed:
        print(f"    CHANGED {k:24s} {o!r} -> {n!r}")
    for k in added:
        print(f"    PINNED  {k:24s} {ran[k]!r}")
    bad = verify(new_text, old_text, ran, a.num_chains, a.chains_per_gpu, ann)
    print("    VERIFY: " + ("PASS -- every behaviour-relevant key equals what the "
                            "trial ran, except num_chains/chains_per_gpu"
                            if not bad else f"FAIL ({len(bad)})"))
    for b in bad:
        print(f"      !! {b}")
    if a.write and not bad:
        open(path, "w").write(new_text)
        print("    WRITTEN.")
    elif a.write:
        print("    NOT written: verification failed.")
        return 1
    else:
        sys.stdout.writelines(difflib.unified_diff(
            old_text.splitlines(True), new_text.splitlines(True),
            "before", "after", n=0))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
