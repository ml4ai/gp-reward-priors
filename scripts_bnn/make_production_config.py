#!/usr/bin/env python
"""Regenerate a production config from its sweep winner (to-do 10/10b, §4.3.130/§4.3.137).

`--family bnn` (default) | `mr` | `pt`.  The BNN description below applies to all
three with two differences for MR/PT (§4.3.137): there is no chain budget to set
(seed 0 is simply retrained by train_rewards.sh), and PT's derived fields
(`pref_attn_embd_dim`, `intermediate_dim`, `num_heads`) are left null for
`__post_init__` to compute, exactly as the trial read them.

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
    /opt/anaconda3/envs/irl/bin/python scripts_bnn/make_production_config.py \\
        --family pt medium_play giab551o [--write]
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
          "group", "train_dataset", "val_dataset", "test_dataset",
          "checkpoints_path"}          # MR/PT: gets _{seed} appended per run
NEVER_WRITE = {"burn_in_lr"}          # preflight-enforced absence (§3.7)
PREFLIGHT_CENTRE = re.compile(
    r"^[ \t]*centre_draws[ \t]*:[ \t]*(true|True)[ \t]*$", re.M)

# Per-family facts (handoff 4.3.137).  `derived` fields are computed in
# TrainConfig.__post_init__ when left None, and wandb logs the DERIVED value --
# the trial itself read None from the file.  Writing them would replace a
# derivation with a constant, so they are never written and never compared.
# `exponents` are log2 fields: a sweep trial logs the exponent (the agent pre-sets
# sweep keys), a direct launch logs 2**x after __post_init__.
FAMILIES = {
    "bnn": {"project": "BNN-training", "dir": "scripts_bnn", "suffix": "bnn",
            "script": "run_bnn_training_antmaze_eval.py", "budget": True,
            "derived": set(), "exponents": {"width"}},
    "mr": {"project": "MR-training", "dir": "scripts_mr", "suffix": "mr",
           "script": "run_mr_training_antmaze_eval.py", "budget": False,
           "derived": set(), "exponents": {"width"}},
    "pt": {"project": "PT-training", "dir": "scripts_pt", "suffix": "pt",
           "script": "run_pt_training_antmaze_eval.py", "budget": False,
           "derived": {"pref_attn_embd_dim", "intermediate_dim", "num_heads"},
           "exponents": {"embd_dim", "head_dim"}},
}


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


def _targets(ran, num_chains, cpg, ann, family):
    """What the file must hold: the recorded config, minus per-run identity,
    never-write keys, None values and derived fields; plus the BNN budget."""
    fam = FAMILIES[family]
    t = {k: coerce(k, v, ann) for k, v in ran.items()
         if k not in EXEMPT and k not in NEVER_WRITE and v is not None
         and k not in fam["derived"]}
    if fam["budget"]:
        t["num_chains"], t["chains_per_gpu"] = num_chains, cpg
    return t


def build(text, ran, rid, num_chains, cpg, ann=None, family="bnn"):
    """(new_text, changed, added).  Pure function of its inputs."""
    old = yaml.safe_load(text)
    target = _targets(ran, num_chains, cpg, ann, family)

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
        who = ("round-5 winner" if family == "bnn"
               else "round-2 baseline winner")
        why = (f"round-5 escalation, §4.3.129 (was {fmt(old.get(k))})"
               if k in ("num_chains", "chains_per_gpu")
               else f"{who} {rid} (was {fmt(old.get(k))})")
        if k in FAMILIES[family]["exponents"]:
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

    # Replace the leading comment header with a provenance block.
    j = next(i for i, ln in enumerate(lines) if ln.strip() and not ln.startswith("#"))
    if family != "bnn":
        header = [
            f"# Production config: round-2 {family.upper()} BASELINE winner {rid}",
            "# (handoff §4.3.108 winners table; regenerated §4.3.137).",
            "# Generated by scripts_bnn/make_production_config.py from the winning",
            "# trial's recorded wandb config, so every value below is what that trial",
            "# RAN, except by design:",
            "#   seed / checkpoints_path / data paths -- set per seed by the launcher.",
            "#   derived fields left null -- TrainConfig.__post_init__ computes them,",
            "#     exactly as it did for the trial." if FAMILIES[family]["derived"]
            else "#   (no derived fields for this family)",
            "# No escalation for this family: seed 0 is retrained by train_rewards.sh.",
            "# This file is also the sweep's base config; the sweep overrides every",
            "# field it varied, so editing it does not change a sweep re-run.",
            "# The previous header and values (pre-§3.2.16 winners) are in git history.",
            "#",
        ]
        return "\n".join(header + lines[j:]), changed, added
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


def verify(new_text, old_text, ran, num_chains, cpg, ann=None, family="bnn"):
    """List of failures (empty = pass)."""
    new, old = yaml.safe_load(new_text), yaml.safe_load(old_text)
    fam = FAMILIES[family]
    bad = []
    for k, v in ran.items():
        if k in EXEMPT or k in NEVER_WRITE or v is None:
            continue
        if k in fam["derived"]:
            # must be left to __post_init__ exactly as the trial read it
            if new.get(k) != old.get(k):
                bad.append(f"{k}: derived field changed {old.get(k)!r} -> "
                           f"{new.get(k)!r}; __post_init__ must compute it")
            continue
        budget = ({"num_chains": num_chains, "chains_per_gpu": cpg}
                  if fam["budget"] else {})
        want = budget.get(k, coerce(k, v, ann))
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
    if family == "bnn" and not PREFLIGHT_CENTRE.search(new_text):
        bad.append("centre_draws line no longer matches the launch preflight")
    if "SUPERSEDED-ROUND1" in new_text:
        bad.append("SUPERSEDED-ROUND1 marker present -- train_rewards.sh would refuse")
    return bad


# Differences between a LAUNCHED run's wandb config and its sweep winner's that
# are logging artefacts, not configuration differences (handoff 4.3.131):
#   width        a sweep run records the EXPONENT (the agent pre-sets it, and
#                wandb.init does not overwrite sweep keys); a direct launch
#                records the value AFTER __post_init__'s 2**width.  Same network.
#   config_path  a sweep passes it as a parameter; a direct --config_path is
#                consumed by pyrallis and never logged.
#   name         per-run uuid suffix.
EXPECTED_RUN_DIFFS = {"num_chains", "chains_per_gpu", "name", "config_path",
                      "seed", "OUT_DIR", "train_dataset", "val_dataset",
                      "test_dataset"}


PATH_KEYS = {"data_root", "measurement_dataset", "train_dataset", "val_dataset",
             "test_dataset", "OUT_DIR", "checkpoints_path"}
# MR/PT write to checkpoints_path (+ _{seed}); BNN to OUT_DIR (+ _{seed}).
SEED_KEYS = {"seed", "OUT_DIR", "checkpoints_path", "train_dataset",
             "val_dataset", "test_dataset"}


def _same_path(a, b):
    """Same file written differently: train_rewards.sh passes ABSOLUTE
    data_root/measurement_dataset, a sweep leaves them repo-relative."""
    if not (isinstance(a, str) and isinstance(b, str)):
        return False
    a2, b2 = a.lstrip("./"), b.lstrip("./")
    return a2 == b2 or a2.endswith("/" + b2) or b2.endswith("/" + a2)


def _norm_width(w):
    """Exponent 4-7 and expanded 16-128 never overlap, so this is unambiguous."""
    return int(round(__import__("math").log2(w))) if w and w >= 16 else w


# Largest exponent each log2 field can take under §3.2.16's ranges; anything
# above is already expanded.  MR/BNN width: exponent 4-7 vs expanded >= 16.
# PT embd/head: exponent 3-5 vs expanded >= 8.  Unambiguous for these ranges.
_EXP_MAX = {"width": 12, "embd_dim": 7, "head_dim": 7}


def _expanded(family, key, cfg):
    """The value __post_init__ would produce, whichever form cfg holds.
    PT clamps head_dim to embd_dim, so a trial with head 5 / embd 3 runs head 8."""
    v = cfg.get(key)
    if v is None or isinstance(v, bool):
        return v
    e = 2 ** v if v <= _EXP_MAX.get(key, -1) else v
    if family == "pt" and key == "head_dim":
        return min(e, _expanded(family, "embd_dim", cfg))
    return e


def check_run(run_cfg, winner_cfg, num_chains, cpg, same_seed=None,
              family="bnn"):
    """[(key, winner, run, verdict)] for every differing key; verdict is
    'expected', 'artefact' or 'UNEXPECTED'.

    same_seed=None infers it: a seed-0 escalation must match the winner's seed,
    output dir and data split exactly; a seeds 1-10 production run legitimately
    differs in exactly those, and in nothing else.
    """
    if same_seed is None:
        same_seed = run_cfg.get("seed") == winner_cfg.get("seed")
    out = []
    for k in sorted(set(run_cfg) | set(winner_cfg)):
        a, b = winner_cfg.get(k), run_cfg.get(k)
        if k in FAMILIES[family]["exponents"] and \
                _expanded(family, k, winner_cfg) == _expanded(family, k, run_cfg):
            if a != b:
                out.append((k, a, b, "artefact"))
            continue
        if _same(a, b):
            continue
        if k in PATH_KEYS and _same_path(a, b):
            out.append((k, a, b, "artefact"))          # same file, other spelling
            continue
        if k == "num_chains" and FAMILIES[family]["budget"]:
            ok, tag = b == num_chains, "expected"
        elif k == "chains_per_gpu" and FAMILIES[family]["budget"]:
            ok, tag = b == cpg, "expected"
        elif k in SEED_KEYS:
            ok, tag = not same_seed, "expected (other seed)"
        else:
            ok, tag = k in EXPECTED_RUN_DIFFS, "artefact"
        out.append((k, a, b, tag if ok else "UNEXPECTED"))
    return out


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
    # check_run: the observed large_diverse launch (4.3.131) must read clean...
    win = {"width": 4, "num_chains": 32, "chains_per_gpu": 32, "seed": 0,
           "config_path": "scripts_bnn/x.yaml", "name": "a", "mdecay": 0.03}
    run = {"width": 16, "num_chains": 128, "chains_per_gpu": 32, "seed": 0,
           "config_path": None, "name": "b", "mdecay": 0.03}
    assert all(v != "UNEXPECTED" for *_, v in check_run(run, win, 128, 32))
    # ...and a real difference, a wrong budget, or a double-expanded width must not
    for bad in ({"mdecay": 0.3}, {"num_chains": 96}, {"width": 65536}):
        assert any(v == "UNEXPECTED" for *_, v in
                   check_run(dict(run, **bad), win, 128, 32)), bad
    # a seed-0 run whose OUT_DIR differs is wrong...
    assert any(v == "UNEXPECTED" for *_, v in check_run(
        dict(run, OUT_DIR="./x_1"), dict(win, OUT_DIR="./x_0"), 128, 32))
    # ...but a seeds 1-10 production run (train_rewards.sh: other seed, other
    # split, ABSOLUTE data paths) must read clean, while a sampler change in it
    # must still be caught
    w0 = dict(win, OUT_DIR="./exp/m_0", data_root="data/antmaze",
              measurement_dataset="data/antmaze/v/t.hdf5",
              val_dataset="data/antmaze/v/eval/seed_0/v_pref_val_0.hdf5")
    r3 = dict(run, seed=3, OUT_DIR="./exp/m_3", data_root="/home/u/g/data/antmaze",
              measurement_dataset="/home/u/g/data/antmaze/v/t.hdf5",
              val_dataset="/home/u/g/data/antmaze/v/eval/seed_3/v_pref_val_3.hdf5")
    assert all(v != "UNEXPECTED" for *_, v in check_run(r3, w0, 128, 32))
    assert any(v == "UNEXPECTED" for *_, v in
               check_run(dict(r3, mdecay=0.3), w0, 128, 32))
    assert any(v == "UNEXPECTED" for *_, v in check_run(
        dict(r3, measurement_dataset="/home/u/g/data/OTHER.hdf5"), w0, 128, 32))

    # ---- MR / PT (4.3.137) ----------------------------------------------------
    # PT: derived fields stay null even though wandb logged their derived values
    pt_text = ("# old header\nembd_dim: 7\nhead_dim: 7\nnum_layers: 1\n"
               "lr: 0.0085\npref_attn_embd_dim: null\nintermediate_dim: null\n"
               "epochs: 5000\nseed: 1\ncheckpoints_path: ./exp/pt\n")
    pt_ran = {"embd_dim": 5, "head_dim": 3, "num_layers": 4, "lr": 2.0e-05,
              "pref_attn_embd_dim": 32, "intermediate_dim": 128, "num_heads": 4,
              "epochs": 5000, "seed": 0, "checkpoints_path": "./exp/pt_0",
              "select_split": "test"}
    pt_ann = {"embd_dim": "int", "head_dim": "int", "num_layers": "int",
              "lr": "float", "epochs": "int", "select_split": "str"}
    new_pt, ch_pt, add_pt = build(pt_text, pt_ran, "giab551o", 128, 32,
                                  pt_ann, family="pt")
    ypt = yaml.safe_load(new_pt)
    assert ypt["embd_dim"] == 5 and ypt["head_dim"] == 3 and ypt["num_layers"] == 4
    assert ypt["pref_attn_embd_dim"] is None and ypt["intermediate_dim"] is None
    assert "num_heads" not in ypt and "num_chains" not in ypt, ypt
    assert ypt["select_split"] == "test" and ypt["seed"] == 1, ypt
    assert ypt["checkpoints_path"] == "./exp/pt", "per-seed path must be exempt"
    assert not verify(new_pt, pt_text, pt_ran, 128, 32, pt_ann, family="pt"), \
        verify(new_pt, pt_text, pt_ran, 128, 32, pt_ann, family="pt")
    assert verify(new_pt.replace("pref_attn_embd_dim: null", "pref_attn_embd_dim: 32"),
                  pt_text, pt_ran, 128, 32, pt_ann, family="pt"), \
        "writing a derived field must fail verification"
    assert "BASELINE winner giab551o" in new_pt and "escalation" not in new_pt.split("\n")[0]
    assert "round-2 baseline winner giab551o" in new_pt and "round-5" not in new_pt
    # MR: no budget keys, no centre_draws requirement
    mr_text = "# h\nwidth: 8\ndepth: 5\nlr: 0.0062\nseed: 1\n"
    mr_ran = {"width": 7, "depth": 4, "lr": 0.00041, "seed": 0}
    new_mr, _, _ = build(mr_text, mr_ran, "a4qo4g4i", 128, 32,
                         {"width": "int", "depth": "int", "lr": "float"}, family="mr")
    ymr = yaml.safe_load(new_mr)
    assert ymr["width"] == 7 and "num_chains" not in ymr, ymr
    assert not verify(new_mr, mr_text, mr_ran, 128, 32,
                      {"width": "int", "depth": "int", "lr": "float"}, family="mr")
    # check_run PT: sweep-logged exponents (embd 3, head 5 -> clamped to 8)
    # against a direct launch's expanded values must read clean...
    ptw = {"embd_dim": 3, "head_dim": 5, "num_heads": 1, "pref_attn_embd_dim": 8,
           "intermediate_dim": 32, "lr": 0.0089, "seed": 0}
    ptr = {"embd_dim": 8, "head_dim": 8, "num_heads": 1, "pref_attn_embd_dim": 8,
           "intermediate_dim": 32, "lr": 0.0089, "seed": 0}
    assert all(v != "UNEXPECTED" for *_, v in check_run(ptr, ptw, 128, 32,
                                                        family="pt"))
    # ...and a genuinely different head must not
    assert any(v == "UNEXPECTED" for *_, v in check_run(
        dict(ptr, head_dim=4, num_heads=2), ptw, 128, 32, family="pt"))
    # MR seeds 1-10 via train_rewards.sh: checkpoints_path is per seed, width is
    # logged expanded; neither is a mismatch, but a different lr is
    mrw = {"width": 7, "lr": 0.00041, "seed": 0,
           "checkpoints_path": "./exp/reward_learning/m_mr_eval_0"}
    mrr = {"width": 128, "lr": 0.00041, "seed": 4,
           "checkpoints_path": "./exp/reward_learning/m_mr_eval_4"}
    assert all(v != "UNEXPECTED" for *_, v in check_run(mrr, mrw, 128, 32,
                                                        family="mr"))
    assert any(v == "UNEXPECTED" for *_, v in check_run(
        dict(mrr, lr=0.001), mrw, 128, 32, family="mr"))
    # ...and on seed 0 a changed checkpoints_path IS a mismatch
    assert any(v == "UNEXPECTED" for *_, v in check_run(
        dict(mrr, seed=0), mrw, 128, 32, family="mr"))
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
    ap.add_argument("--check-run", metavar="RUN_ID", default=None,
                    help="instead of writing: diff a LAUNCHED run's wandb config "
                         "against the winner, with known logging artefacts "
                         "(width exponent vs expanded, config_path, name) "
                         "classified rather than flagged")
    ap.add_argument("--family", choices=sorted(FAMILIES), default="bnn",
                    help="bnn (default) | mr | pt -- selects the wandb project, "
                         "config directory, TrainConfig source and derived fields")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    fam = FAMILIES[a.family]
    project = fam["project"]
    root = os.path.join(os.path.dirname(HERE), fam["dir"])
    if a.check_run:
        if not a.run_id:
            ap.error("--check-run needs the winner's run_id as well")
        import wandb
        api = wandb.Api(timeout=60)
        g = lambda rid: {k: v for k, v in dict(api.run(
            f"{ENTITY}/{project}/{rid}").config).items() if not k.startswith("_")}
        r = api.run(f"{ENTITY}/{project}/{a.check_run}")
        rows = check_run(g(a.check_run), g(a.run_id), a.num_chains,
                         a.chains_per_gpu, family=a.family)
        print(f"launched run {a.check_run} ({r.state}) vs winner {a.run_id}:")
        for k, w, v, verdict in rows:
            print(f"   {k:28s} winner={w!r:40.40s} run={v!r:40.40s} {verdict}")
        bad = [k for k, *_, v in rows if v == "UNEXPECTED"]
        print("VERDICT: " + ("the launched run IS the winner's configuration, "
                             "apart from the chain budget and logging artefacts"
                             if not bad else f"MISMATCH on {bad} -- stop the run"))
        return 1 if bad else 0
    if not (a.variant and a.run_id):
        ap.error("variant and run_id are required")

    import wandb
    run = wandb.Api(timeout=60).run(f"{ENTITY}/{project}/{a.run_id}")
    ran = {k: v for k, v in dict(run.config).items() if not k.startswith("_")}
    if ran.get("antmaze_variant", "").replace("-", "_").find(a.variant) < 0:
        sys.exit(f"run {a.run_id} is {ran.get('antmaze_variant')!r}, not {a.variant}")

    path = os.path.join(root, f"antmaze_{a.variant}_{fam['suffix']}_antmaze_eval.yaml")
    old_text = open(path).read()
    ann = field_types(os.path.join(root, fam["script"]))
    new_text, changed, added = build(old_text, ran, a.run_id, a.num_chains,
                                     a.chains_per_gpu, ann, family=a.family)

    print(f"=== {a.family.upper()} {a.variant}: {path}")
    print(f"    winner {a.run_id} -> {len(changed)} value(s) changed, "
          f"{len(added)} key(s) pinned explicitly")
    for k, o, n in changed:
        print(f"    CHANGED {k:24s} {o!r} -> {n!r}")
    for k in added:
        print(f"    PINNED  {k:24s} {ran[k]!r}")
    bad = verify(new_text, old_text, ran, a.num_chains, a.chains_per_gpu, ann,
                 family=a.family)
    print("    VERIFY: " + ("PASS -- every behaviour-relevant key equals what the "
                            "trial ran" + (", except num_chains/chains_per_gpu"
                                           if fam["budget"] else "")
                            + ("; derived fields left to __post_init__"
                               if fam["derived"] else "")
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
