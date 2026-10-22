#!/usr/bin/env python
"""experiment_conditions.py -- the ONE definition of an ablation-experiment condition.

Experiments 1 (data sparsity) and 2 (label noise) train every reward model again
on altered TRAINING data, with everything selected frozen at its anchor value
(handoff §11, §13; dissertation Ch. 3, Experimental Design).  A condition changes
exactly two things about a run: which split files it reads, and where it writes.
Both are defined here and nowhere else, so that the launcher (train_rewards.sh),
the verifier, and the IQL sweep generator cannot disagree about them.

CONDITIONS
  r16 r32 r64 r128        Experiment 1.  Train on the nested N-pair subset
                          reduction/<env>_pref_train_<s>_<N>.hdf5.
                          Validation and test are the base files: they never
                          shrink.
  n01 n02 n03 n04 n045    Experiment 2, PRIMARY arm ("noisy selection").  Train
                          on noise/<env>_pref_train_<s>_<pp>.hdf5.  For MR and PT
                          the checkpoint-selection split (test) is corrupted at
                          the same level, noise/<env>_pref_test_<s>_<pp>.hdf5.
                          The BNN selects no checkpoint and keeps the base test.
  n01o ... n045o          Experiment 2, SENSITIVITY arm ("oracle selection").
                          Same training labels, UNCORRUPTED test.  MR and PT on
                          medium_play only (scope fixed 2026-10-02, §11.3).

  Validation is NEVER overridden.  It is a measuring instrument and stays the
  base, uncorrupted file in every condition (§11.2).  The corrupted validation
  files on disk belong to a superseded design; this module cannot name them.

OUTPUT DIRECTORIES
  antmaze_<variant>_<family>_<cond>_eval_<seed>      (the script appends _<seed>)
  The condition is in the BASENAME because the BNN label cache keys an entry on
  the model directory's basename (reward_label_cache.make_key).  A model written
  to exp/.../antmaze_<variant>_bnn_eval_<seed> under some other parent would share
  the anchor model's cache key (§13.2).  It also means no two conditions, and
  neither arm, can overwrite another's model (§11.5).

SEEDS  1-10 only.  Nothing is re-selected under a condition, so there is no
       seed-0 lineage in either experiment (§11.1).

Usage:
    python scripts_bnn/experiment_conditions.py args bnn large_play 3 r32 --data-root data/antmaze
    python scripts_bnn/experiment_conditions.py preflight bnn --conds "r16 r32" \\
        --variants "medium_play large_play" --seeds "1 2 3" \\
        --data-root data/antmaze --models-root ../exp/reward_learning --num-chains 128
    python scripts_bnn/experiment_conditions.py jobs mr --conds "r16 r32"
    python scripts_bnn/experiment_conditions.py --selftest
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FAMILIES = ("bnn", "mr", "pt", "nmr")                  # NMR added 2026-10-22 (handoff 14.17)
VARIANTS = ("medium_play", "medium_diverse", "large_play", "large_diverse")
EVAL_SEEDS = tuple(range(1, 11))
REDUCTION_SIZES = (16, 32, 64, 128)                    # decided 2026-10-12, §13.5
NOISE_LEVELS = ("01", "02", "03", "04", "045")         # p = 0.1 ... 0.45
SPARSITY = tuple(f"r{n}" for n in REDUCTION_SIZES)
NOISE_PRIMARY = tuple(f"n{p}" for p in NOISE_LEVELS)
NOISE_ORACLE = tuple(f"n{p}o" for p in NOISE_LEVELS)
ALL_CONDS = SPARSITY + NOISE_PRIMARY + NOISE_ORACLE
# §11.3: the SENSITIVITY arm's families.  NMR joined on 2026-10-22 (author, handoff
# 14.18): it picks its checkpoint on the test split, as MR and PT do.
ORACLE_FAMILIES = ("mr", "pt", "nmr")
# Families that pick their checkpoint on the TEST split, so in the noise PRIMARY arm
# their test split is corrupted like the training labels.  The same three today, but
# kept as its own list: the two roles are different questions (handoff 14.17).
SELECTS_CHECKPOINT = ("mr", "pt", "nmr")
ORACLE_VARIANTS = ("medium_play",)                     # §11.3, fixed in advance
# The key each training script takes its output directory under.
OUT_KEY = {"bnn": "OUT_DIR", "mr": "checkpoints_path", "pt": "checkpoints_path", "nmr": "checkpoints_path"}
# What a finished model directory contains, used to tell an empty dir from a model.
MODEL_MARKERS = {"bnn": "sampling_f", "mr": "best_model.pt", "pt": "best_model.pt", "nmr": "best_model.pt"}


def parse(cond):
    """dict(kind, level, arm) for a condition token; ValueError if it is not one."""
    m = re.fullmatch(r"r(\d+)", cond or "")
    if m and int(m.group(1)) in REDUCTION_SIZES:
        return {"kind": "reduction", "level": int(m.group(1)), "arm": None}
    m = re.fullmatch(r"n(\d+)(o?)", cond or "")
    if m and m.group(1) in NOISE_LEVELS:
        return {"kind": "noise", "level": m.group(1),
                "arm": "oracle" if m.group(2) else "primary"}
    raise ValueError(f"unknown condition {cond!r}; choose from {' '.join(ALL_CONDS)}")


def env_of(variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return f"antmaze-{variant.replace('_', '-')}-v2"


def model_base(variant, family, cond=None):
    """Directory basename WITHOUT the seed.  cond=None is the anchor model."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    env_of(variant)
    if cond is None:
        return f"antmaze_{variant}_{family}_eval"
    parse(cond)
    return f"antmaze_{variant}_{family}_{cond}_eval"


def check_allowed(family, variant, seed, cond):
    """Raise ValueError if this (family, variant, seed, condition) is not in the design."""
    c = parse(cond)
    env_of(variant)
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if int(seed) not in EVAL_SEEDS:
        raise ValueError(f"seed {seed}: the experiments run at seeds 1-10 only; nothing is "
                         "re-selected under a condition, so there is no seed-0 lineage (§11.1)")
    if c["arm"] == "oracle":
        if family not in ORACLE_FAMILIES:
            raise ValueError(f"{cond}: the sensitivity arm is MR, PT and NMR only; the {family} "
                             "selects no checkpoint and is one model in both arms (§11.3)")
        if variant not in ORACLE_VARIANTS:
            raise ValueError(f"{cond}: the sensitivity arm runs on medium_play only, a scope "
                             "fixed before any primary-arm result (§11.3)")


def split_overrides(family, variant, seed, cond, data_root):
    """{"train_dataset": path[, "test_dataset": path]}.  Never a val_dataset."""
    check_allowed(family, variant, seed, cond)
    c = parse(cond)
    env = env_of(variant)
    sd = os.path.join(data_root, env, "eval", f"seed_{seed}")
    if c["kind"] == "reduction":
        return {"train_dataset": os.path.join(
            sd, "reduction", f"{env}_pref_train_{seed}_{c['level']}.hdf5")}
    out = {"train_dataset": os.path.join(sd, "noise", f"{env}_pref_train_{seed}_{c['level']}.hdf5")}
    if c["arm"] == "primary" and family in SELECTS_CHECKPOINT:
        # MR/PT pick their checkpoint on the test split, so in the PRIMARY arm it
        # is corrupted like the training labels.  The BNN has no such decision.
        out["test_dataset"] = os.path.join(
            sd, "noise", f"{env}_pref_test_{seed}_{c['level']}.hdf5")
    return out


def cli_args(family, variant, seed, cond, data_root, out_parent="./exp/reward_learning"):
    """The extra command-line arguments for one training run, as a flat list.

    out_parent stays RELATIVE by default, exactly as the production configs have
    it: the training scripts chdir to the parent repo at import, so
    ./exp/reward_learning resolves to ~/iqlpref/exp/reward_learning, where the
    IQL stage reads (handoff §4.3.135).
    """
    ov = split_overrides(family, variant, seed, cond, data_root)
    args = []
    for k in ("train_dataset", "test_dataset"):
        if k in ov:
            args += [f"--{k}", ov[k]]
    args += [f"--{OUT_KEY[family]}", f"{out_parent}/{model_base(variant, family, cond)}"]
    return args


def expected_run_paths(family, variant, seed, cond):
    """What a finished run's wandb config must show (handoff §11.6), as path
    TAILS, so that the check does not depend on where the data root is mounted.

    {"train_dataset": tail, "val_dataset": tail, "test_dataset": tail, "out": basename}
    """
    env = env_of(variant)
    base = f"{env}/eval/seed_{seed}/{env}_pref_{{}}_{seed}.hdf5"
    ov = split_overrides(family, variant, seed, cond, "")
    want = {"val_dataset": base.format("val")}          # ALWAYS the base file
    for part in ("train", "test"):
        k = f"{part}_dataset"
        want[k] = ov[k].lstrip("/") if k in ov else base.format(part)
    want["out"] = f"{model_base(variant, family, cond)}_{seed}"
    return want


def check_run_paths(run_cfg, family, variant, seed, cond):
    """[problems] comparing a run's logged paths with expected_run_paths."""
    want = expected_run_paths(family, variant, seed, cond)
    probs = []
    for k in ("train_dataset", "val_dataset", "test_dataset"):
        got = str(run_cfg.get(k) or "")
        if not got.endswith("/" + want[k]) and got != want[k]:
            probs.append(f"{k} is {got!r}, expected .../{want[k]}")
    val = str(run_cfg.get("val_dataset") or "")
    if "/noise/" in val or "/reduction/" in val:
        probs.append(f"val_dataset is an ALTERED file ({val!r}); validation must be the base split")
    out = os.path.basename(os.path.normpath(str(run_cfg.get(OUT_KEY[family]) or "")))
    if out != want["out"]:
        probs.append(f"{OUT_KEY[family]} basename is {out!r}, expected {want['out']!r}")
    if run_cfg.get("seed") != seed:
        probs.append(f"seed is {run_cfg.get('seed')!r}, expected {seed}")
    return probs


# The IQL evaluation reads a BNN model at two conservatism levels, CVaR at 0.95
# and the posterior mean at 0.0, and alpha is cache-key material (§3.2.9).  Both
# are cached in one pass, at the evaluation's own n_samples and centring.
LABEL_ALPHAS = "0.95,0.0"


def label_commands(variants, seeds, conds, gpu=0, models_parent="exp/reward_learning"):
    """One caching command per (condition, variant), to be run from ~/iqlpref.

    Each covers that cell's seeds, which is one IQL sweep's worth of models, and
    names every run dir explicitly (no glob: `_eval_1*` would also match seed
    10).  One command at a time per GPU: labelling peaks near 61 GB of host RAM
    per model at 128 chains (§4.3.129)."""
    out = []
    for c in conds:
        parse(c)
        for v in variants:
            for s in seeds:
                check_allowed("bnn", v, s, c)
            dirs = " ".join(f"--run-dir {models_parent}/{model_base(v, 'bnn', c)}_{s}" for s in seeds)
            out.append(
                f"CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 "
                f"python gp_reward-priors/precompute_labels.py --env {env_of(v)} "
                f"--alphas {LABEL_ALPHAS} --n-samples -1 --centre-draws --device cuda "
                f"{dirs} > exp/precompute_{v}_{c}.log 2>&1")
    return out


def config_num_chains(family, variant):
    """num_chains from the BNN production config, or None for MR/PT."""
    if family != "bnn":
        return None
    path = os.path.join(HERE, f"antmaze_{variant}_bnn_antmaze_eval.yaml")
    m = re.search(r"^num_chains:\s*(\d+)", open(path).read(), flags=re.M)
    return int(m.group(1)) if m else None


def jobs(family, variants, seeds, conds):
    """[(cond, variant, seed)] in launch order: condition, then variant, then seed.

    That order completes one (condition, variant) cell -- ten seeds, which is one
    IQL sweep -- before starting the next, so evaluation can begin as early as
    possible.  Sensitivity-arm conditions silently cover only their own scope.
    """
    out = []
    for c in conds:
        for v in variants:
            for s in seeds:
                try:
                    check_allowed(family, v, s, c)
                except ValueError:
                    if parse(c)["arm"] == "oracle" and int(s) in EVAL_SEEDS:
                        continue                      # outside the arm's scope: not a job
                    raise
                out.append((c, v, int(s)))
    return out


def preflight(family, variants, seeds, conds, data_root, models_root,
              num_chains=None, allow_existing=False):
    """(job list, [problems]).  Nothing is launched unless problems is empty."""
    probs = []
    try:
        js = jobs(family, variants, seeds, conds)
    except ValueError as e:
        return [], [str(e)]
    if not js:
        probs.append("no jobs: the conditions given do not apply to this family/variants")
    for c, v, s in js:
        for k, p in split_overrides(family, v, s, c, data_root).items():
            if not os.path.isfile(p):
                probs.append(f"{c} {v} seed {s}: {k} missing: {p}")
        d = os.path.join(models_root, f"{model_base(v, family, c)}_{s}")
        if os.path.isdir(d) and os.listdir(d) and not allow_existing:
            what = "a model" if os.path.exists(os.path.join(d, MODEL_MARKERS[family])) else "files"
            probs.append(f"{c} {v} seed {s}: {d} already holds {what}.  A run writes its "
                         "weights only at the end, so a crashed run would leave the OLD "
                         "model looking complete.  Remove it, or set ALLOW_EXISTING=1.")
    if family == "bnn":
        for v in sorted({v for _, v, _ in js}):
            want = config_num_chains(family, v)
            if num_chains is None or int(num_chains) != want:
                probs.append(f"{v}: NUM_CHAINS={num_chains} but the production config says "
                             f"{want}.  The chain budget is frozen at the anchor (§11.1); "
                             f"launch with NUM_CHAINS={want}.")
    return js, probs


def selftest():
    import tempfile
    assert parse("r16") == {"kind": "reduction", "level": 16, "arm": None}
    assert parse("n045") == {"kind": "noise", "level": "045", "arm": "primary"}
    assert parse("n02o") == {"kind": "noise", "level": "02", "arm": "oracle"}
    for bad in ("r256", "r8", "n05", "n1", "x16", "", None, "r16o", "n045oo"):
        try:
            parse(bad)
            raise AssertionError(f"accepted {bad!r}")
        except ValueError:
            pass
    assert len(ALL_CONDS) == 14 and len(set(ALL_CONDS)) == 14
    assert env_of("large_diverse") == "antmaze-large-diverse-v2"
    # names: the condition is in the basename, the anchor is untouched, all distinct
    assert model_base("large_play", "bnn") == "antmaze_large_play_bnn_eval"
    assert model_base("large_play", "bnn", "r32") == "antmaze_large_play_bnn_r32_eval"
    names = {model_base(v, f, c) for v in VARIANTS for f in FAMILIES for c in (None,) + ALL_CONDS}
    assert len(names) == len(VARIANTS) * len(FAMILIES) * 15
    # splits: reduction touches train only
    R = "/d"
    o = split_overrides("mr", "medium_play", 4, "r64", R)
    assert o == {"train_dataset":
                 "/d/antmaze-medium-play-v2/eval/seed_4/reduction/"
                 "antmaze-medium-play-v2_pref_train_4_64.hdf5"}
    assert split_overrides("bnn", "large_play", 1, "r16", R).keys() == {"train_dataset"}
    # noise PRIMARY: MR/PT get a corrupted test split, the BNN does not
    on = split_overrides("nmr", "large_diverse", 7, "n03", R)          # NMR selects on test too
    assert "test_dataset" in on and on["test_dataset"].endswith("_7_03.hdf5"), on
    check_allowed("nmr", "medium_play", 3, "n03o")                       # in the sensitivity arm (14.18)
    try:
        check_allowed("nmr", "large_play", 3, "n03o")                    # medium_play only
        raise AssertionError("nmr accepted off medium_play in the sensitivity arm")
    except ValueError:
        pass
    assert model_base("large_play", "nmr", "r16") == "antmaze_large_play_nmr_r16_eval"
    o = split_overrides("pt", "large_diverse", 7, "n03", R)
    assert o["train_dataset"].endswith("/noise/antmaze-large-diverse-v2_pref_train_7_03.hdf5")
    assert o["test_dataset"].endswith("/noise/antmaze-large-diverse-v2_pref_test_7_03.hdf5")
    assert split_overrides("bnn", "large_diverse", 7, "n03", R).keys() == {"train_dataset"}
    # noise SENSITIVITY: same training file, NO test override
    op = split_overrides("mr", "medium_play", 2, "n045", R)
    oo = split_overrides("mr", "medium_play", 2, "n045o", R)
    assert oo == {"train_dataset": op["train_dataset"]} and "test_dataset" in op
    # validation is never overridden, in any condition, for any family
    for f in FAMILIES:
        for c in ALL_CONDS:
            for v in VARIANTS:
                try:
                    ov = split_overrides(f, v, 5, c, R)
                except ValueError:
                    continue
                assert "val_dataset" not in ov
                assert not any("_pref_val_" in p for p in ov.values())
    # scope: seed 0 never; the sensitivity arm only MR/PT on medium_play
    for args in (("bnn", "medium_play", 0, "r16"), ("mr", "medium_play", 11, "n01"),
                 ("bnn", "medium_play", 3, "n01o"), ("mr", "large_play", 3, "n01o")):
        try:
            check_allowed(*args)
            raise AssertionError(f"allowed {args}")
        except ValueError:
            pass
    check_allowed("pt", "medium_play", 10, "n045o")
    # cli args: flat, in a fixed order, with the family's own output key
    a = cli_args("bnn", "large_play", 3, "r32", R)
    assert a == ["--train_dataset",
                 "/d/antmaze-large-play-v2/eval/seed_3/reduction/"
                 "antmaze-large-play-v2_pref_train_3_32.hdf5",
                 "--OUT_DIR", "./exp/reward_learning/antmaze_large_play_bnn_r32_eval"]
    a = cli_args("mr", "medium_play", 3, "n02", R)
    assert a[0] == "--train_dataset" and a[2] == "--test_dataset" and a[4] == "--checkpoints_path"
    assert a[5] == "./exp/reward_learning/antmaze_medium_play_mr_n02_eval"
    # job order: condition, then variant, then seed; oracle arm keeps to its scope
    js = jobs("bnn", ["medium_play", "large_play"], [1, 2], ["r16", "r32"])
    assert js == [("r16", "medium_play", 1), ("r16", "medium_play", 2), ("r16", "large_play", 1),
                  ("r16", "large_play", 2), ("r32", "medium_play", 1), ("r32", "medium_play", 2),
                  ("r32", "large_play", 1), ("r32", "large_play", 2)]
    assert jobs("mr", list(VARIANTS), [1, 2], ["n01o"]) == [("n01o", "medium_play", 1),
                                                           ("n01o", "medium_play", 2)]
    assert jobs("bnn", list(VARIANTS), [1], ["n01o"]) == []
    try:
        jobs("bnn", ["medium_play"], [0, 1], ["r16"])
        raise AssertionError("seed 0 accepted")
    except ValueError:
        pass
    assert config_num_chains("bnn", "large_play") == 128 and config_num_chains("mr", "large_play") is None
    # what a run's logged config must show
    w = expected_run_paths("bnn", "large_play", 3, "r32")
    assert w == {"val_dataset": "antmaze-large-play-v2/eval/seed_3/antmaze-large-play-v2_pref_val_3.hdf5",
                 "train_dataset": "antmaze-large-play-v2/eval/seed_3/reduction/"
                                  "antmaze-large-play-v2_pref_train_3_32.hdf5",
                 "test_dataset": "antmaze-large-play-v2/eval/seed_3/antmaze-large-play-v2_pref_test_3.hdf5",
                 "out": "antmaze_large_play_bnn_r32_eval_3"}
    B = "/home/x/gp/data/antmaze/"
    good = {"seed": 3, "train_dataset": B + w["train_dataset"], "val_dataset": B + w["val_dataset"],
            "test_dataset": B + w["test_dataset"],
            "OUT_DIR": "./exp/reward_learning/antmaze_large_play_bnn_r32_eval_3"}
    assert check_run_paths(good, "bnn", "large_play", 3, "r32") == []
    base_train = B + "antmaze-large-play-v2/eval/seed_3/antmaze-large-play-v2_pref_train_3.hdf5"
    for k, v in (("train_dataset", base_train),                              # full data, not reduced
                 ("train_dataset", B + w["train_dataset"].replace("_32.", "_64.")),  # wrong level
                 ("val_dataset", B + w["val_dataset"].replace("seed_3/", "seed_3/noise/")),
                 ("test_dataset", B + w["test_dataset"].replace("seed_3", "seed_4")),
                 ("OUT_DIR", "./exp/reward_learning/antmaze_large_play_bnn_eval_3"),  # the anchor's dir
                 ("OUT_DIR", "./exp/reward_learning/antmaze_large_play_bnn_r64_eval_3"),
                 ("seed", 4)):
        assert check_run_paths(dict(good, **{k: v}), "bnn", "large_play", 3, "r32"), (k, v)
    # noise: PRIMARY wants a corrupted test for MR, the base test for the BNN and for the oracle arm
    wp = expected_run_paths("mr", "medium_play", 2, "n045")
    wo = expected_run_paths("mr", "medium_play", 2, "n045o")
    wb = expected_run_paths("bnn", "medium_play", 2, "n045")
    assert "/noise/" in wp["test_dataset"] and "/noise/" not in wo["test_dataset"]
    assert "/noise/" not in wb["test_dataset"] and wp["train_dataset"] == wo["train_dataset"]
    assert all("/noise/" not in x["val_dataset"] for x in (wp, wo, wb))
    assert wp["out"] != wo["out"]
    # preflight: missing files, an occupied directory, and the chain-budget trap
    with tempfile.TemporaryDirectory() as d:
        dr, mr = os.path.join(d, "data"), os.path.join(d, "models")
        js, probs = preflight("bnn", ["large_play"], [1], ["r16"], dr, mr, num_chains=128)
        assert len(js) == 1 and len(probs) == 1 and "train_dataset missing" in probs[0]
        f = split_overrides("bnn", "large_play", 1, "r16", dr)["train_dataset"]
        os.makedirs(os.path.dirname(f)); open(f, "w").close()
        assert preflight("bnn", ["large_play"], [1], ["r16"], dr, mr, num_chains=128)[1] == []
        _, probs = preflight("bnn", ["large_play"], [1], ["r16"], dr, mr, num_chains=8)
        assert len(probs) == 1 and "NUM_CHAINS=8" in probs[0]
        _, probs = preflight("bnn", ["large_play"], [1], ["r16"], dr, mr)
        assert len(probs) == 1 and "NUM_CHAINS=None" in probs[0]
        tgt = os.path.join(mr, "antmaze_large_play_bnn_r16_eval_1")
        os.makedirs(tgt)                                             # empty dir is fine
        assert preflight("bnn", ["large_play"], [1], ["r16"], dr, mr, num_chains=128)[1] == []
        os.makedirs(os.path.join(tgt, "sampling_f"))
        _, probs = preflight("bnn", ["large_play"], [1], ["r16"], dr, mr, num_chains=128)
        assert len(probs) == 1 and "already holds a model" in probs[0]
        assert preflight("bnn", ["large_play"], [1], ["r16"], dr, mr, num_chains=128,
                         allow_existing=True)[1] == []
        # the anchor directory, same variant and seed, is a different name: not flagged
        os.makedirs(os.path.join(mr, "antmaze_large_play_bnn_eval_1", "sampling_f"))
        assert preflight("bnn", ["large_play"], [1], ["r16"], dr, mr, num_chains=128,
                         allow_existing=True)[1] == []
        _, probs = preflight("bnn", ["medium_play"], [1], ["n01o"], dr, mr, num_chains=128)
        assert probs and "no jobs" in probs[0]
        _, probs = preflight("mr", ["large_play"], [0], ["r16"], dr, mr)
        assert probs and "seeds 1-10 only" in probs[0]

    # ---- label caching ----
    lc = label_commands(VARIANTS, EVAL_SEEDS, SPARSITY, gpu=3)
    assert len(lc) == 16 and len(set(lc)) == 16
    one = lc[0]
    assert one.startswith("CUDA_VISIBLE_DEVICES=3 ")
    assert "--alphas 0.95,0.0 --n-samples -1 --centre-draws" in one
    assert "--env antmaze-medium-play-v2" in one and one.endswith("> exp/precompute_medium_play_r16.log 2>&1")
    assert one.count("--run-dir ") == 10 and "--glob" not in one
    for s in EVAL_SEEDS:                                  # every seed's dir, by its §13.2 name
        assert f"--run-dir exp/reward_learning/antmaze_medium_play_bnn_r16_eval_{s} " in one + " "
    assert "_bnn_eval_" not in "".join(lc)                # never an anchor model
    # the dirs named are exactly the ones the launcher writes
    for c in SPARSITY:
        for v in VARIANTS:
            for s in EVAL_SEEDS:
                want = expected_run_paths("bnn", v, s, c)["out"]
                assert sum(f"/{want} " in x + " " for x in lc) == 1, want
    for bad in (lambda: label_commands(VARIANTS, (0,), ("r16",)),          # seed 0
                lambda: label_commands(VARIANTS, EVAL_SEEDS, ("r256",)),   # unknown level
                lambda: label_commands(VARIANTS, EVAL_SEEDS, ("n01o",))):  # BNN has no sensitivity arm
        try:
            bad()
            raise AssertionError("label_commands accepted a bad request")
        except ValueError:
            pass
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    a1 = sub.add_parser("args", help="print one run's extra CLI arguments, one per line")
    a1.add_argument("family"); a1.add_argument("variant"); a1.add_argument("seed", type=int)
    a1.add_argument("cond")
    a1.add_argument("--data-root", default=os.path.join(ROOT, "data", "antmaze"))
    a2 = sub.add_parser("preflight", help="check a whole launch; exit 1 on any problem")
    a2.add_argument("family")
    a2.add_argument("--variants", default=" ".join(VARIANTS))
    a2.add_argument("--seeds", default=" ".join(str(s) for s in EVAL_SEEDS))
    a2.add_argument("--conds", required=True)
    a2.add_argument("--data-root", default=os.path.join(ROOT, "data", "antmaze"))
    a2.add_argument("--models-root",
                    default=os.path.join(os.path.dirname(ROOT), "exp", "reward_learning"))
    a2.add_argument("--num-chains", type=int, default=None)
    a2.add_argument("--allow-existing", action="store_true")
    a3 = sub.add_parser("jobs", help="print the job list, one 'cond variant seed' per line")
    a3.add_argument("family")
    a3.add_argument("--variants", default=" ".join(VARIANTS))
    a3.add_argument("--seeds", default=" ".join(str(s) for s in EVAL_SEEDS))
    a3.add_argument("--conds", required=True)
    a4 = sub.add_parser("labels", help="print the BNN label-caching commands, one per "
                                       "(condition, variant); run them from ~/iqlpref")
    a4.add_argument("--variants", default=" ".join(VARIANTS))
    a4.add_argument("--seeds", default=" ".join(str(s) for s in EVAL_SEEDS))
    a4.add_argument("--conds", required=True)
    a4.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "labels":
        try:
            print("\n".join(label_commands(a.variants.split(), [int(s) for s in a.seeds.split()],
                                           a.conds.split(), a.gpu)))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "args":
        try:
            print("\n".join(cli_args(a.family, a.variant, a.seed, a.cond, a.data_root)))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "jobs":
        try:
            for c, v, s in jobs(a.family, a.variants.split(),
                                [int(s) for s in a.seeds.split()], a.conds.split()):
                print(c, v, s)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "preflight":
        js, probs = preflight(a.family, a.variants.split(), [int(s) for s in a.seeds.split()],
                              a.conds.split(), a.data_root, a.models_root,
                              a.num_chains, a.allow_existing)
        by = {}
        for c, v, s in js:
            by.setdefault((c, v), []).append(s)
        for (c, v), ss in by.items():
            print(f"  {c:6s} {v:15s} seeds {ss[0]}..{ss[-1]} ({len(ss)})  -> "
                  f"{model_base(v, a.family, c)}_<seed>")
        print(f"  {len(js)} job(s), in that order")
        for x in probs:
            print(f"  !! {x}", file=sys.stderr)
        return 1 if probs else 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
