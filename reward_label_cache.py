"""reward_label_cache.py — cache the CVaR reward labels so the CHAINS can be deleted.

Handoff §3.2.9, to-do item 17.  Retaining 11 chain sets per variant after the
sweep is up to 11 × 458 GB ≈ **5 TB**; the reward labels those chains exist to
produce are one float32 per transition, **~4 MB**.  Caching them also removes the
re-labelling cost from each of stage 4's **8 normalization indices**, which today
repeat the entire forward pass over every posterior draw to arrive at the same
numbers.

WHAT IS CACHED
--------------
`penalized_r`: the (N-1,) float32 CVaR reward, **before** the keep mask, **before**
`gauge_reward()` and **before** `modify_reward()`.  Pre-gauge is the reusable
point — gauge mode and normalization index are config knobs applied downstream,
so all 8 stage-4 indices and every gauge mode share ONE cache entry.

CORRECTNESS IS THE WHOLE PROBLEM
--------------------------------
A cache that silently serves the wrong labels puts wrong rewards into the paper's
results, and nothing downstream would notice.  So:

* the key covers **every** input that determines the labels;
* `LOGIC_VERSION` must be bumped whenever the labelling maths changes.  It would
  have had to be bumped for item F (§4.3.114), which changed every label;
* when the source artefacts are still present the stored inventory is **verified**
  on every read, so retraining into the same `OUT_DIR` — the hazard §10.3 records
  as having destroyed evidence twice — is caught;
* when the artefacts are gone (the point of the cache) verification is necessarily
  partial, and every such read says so **loudly** rather than silently.

The key is deliberately computable WITHOUT the source artefacts — otherwise
deleting them would make the entry unfindable.  The artefact inventory is stored
as verification data, not as key material.

DEVICE AND DETERMINISM
----------------------
`device` is recorded but is NOT key material.  A GPU and a CPU forward pass can
differ in the last bits, so including it would produce two entries for what is
meant to be one labelling.  The cache therefore **fixes a canonical labelling**:
first writer wins, and the device that produced it is recorded in the metadata.
That is a reproducibility improvement over re-labelling per consumer, but it is a
deliberate choice and is stated here so it is not mistaken for an oversight.

Usage (module):
    import reward_label_cache as rlc
    key, payload = rlc.make_key("bnn", model_dir, env_name, n, params)
    r, status = rlc.load(key, payload, expect_len=n, inventory=inv)
    if r is None:
        r = ...expensive...
        rlc.save(key, payload, r, inventory=inv, meta={...})

Usage (CLI):
    /opt/anaconda3/envs/irl/bin/python reward_label_cache.py --selftest
    /opt/anaconda3/envs/irl/bin/python reward_label_cache.py --list
    /opt/anaconda3/envs/irl/bin/python reward_label_cache.py --emit-rm
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

# Bump on ANY change to the labelling maths.  History:
#   1 — centred CVaR (item F, §4.3.114) + np.partition tail mean, 2026-09-20.
LOGIC_VERSION = 1

ENV_VAR = "REWARD_LABEL_CACHE_DIR"
_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "reward_label_cache")

# Statuses returned by load().  Only HIT* return an array.
HIT = "HIT"
HIT_UNVERIFIED = "HIT_UNVERIFIED"      # sources gone — expected after deletion
MISS = "MISS"
MISS_STALE = "MISS_STALE"              # sources changed under a live entry
MISS_CORRUPT = "MISS_CORRUPT"


def cache_root():
    return os.environ.get(ENV_VAR) or _DEFAULT


def _entry_path(key):
    return os.path.join(cache_root(), f"{key}.npz")


def inventory(paths, root):
    """[[relpath, size_bytes], ...] sorted — identifies the source artefacts.

    `mtime` is deliberately EXCLUDED: it changes under rsync/copy, which would
    make every transferred cache read as stale.  Path + size catches added,
    removed, truncated and regrown files; a rewrite to a byte-identical size is
    caught instead by the weights digest recorded at write time.
    """
    out = []
    for p in sorted(paths):
        try:
            size = os.path.getsize(p)
        except OSError:
            size = -1
        out.append([os.path.relpath(p, root), int(size)])
    return out


def make_key(kind, model_dir, env_name, n_transitions, params):
    """(key, payload).  Computable WITHOUT the source artefacts, by design.

    `model_dir` enters as its basename, not its full path, so a cache stays valid
    when the tree is moved or the work is done on a different machine.  The
    basename is the run name and is what distinguishes seeds and variants.
    """
    payload = {
        "logic_version": LOGIC_VERSION,
        "kind": kind,
        "model": os.path.basename(os.path.normpath(model_dir)),
        "env_name": env_name,
        "n_transitions": int(n_transitions),
        "params": {k: params[k] for k in sorted(params)},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32], payload


def digest_weights(all_weights, cap=8):
    """A cheap, order-sensitive digest of the loaded posterior draws.

    Computed on the miss path, where the weights are in hand anyway, and stored
    so that an entry read back after its artefacts are gone can still be checked
    if they are ever restored.  Only the first `cap` draws are hashed: enough to
    detect a different training run, and O(1) in the draw count.
    """
    h = hashlib.sha256()
    h.update(f"n={len(all_weights)}".encode())
    for w in all_weights[:cap]:
        for a in w:
            arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()[:32]


def load(key, payload, expect_len, inventory_now=None):
    """(rewards | None, status).  Verifies everything it can, loudly."""
    path = _entry_path(key)
    if not os.path.exists(path):
        return None, MISS
    try:
        with np.load(path, allow_pickle=False) as z:
            rewards = z["rewards"]
            meta = json.loads(str(z["meta"]))
    except Exception as e:  # noqa: BLE001 — a bad entry must never be trusted
        print(f"[label-cache] CORRUPT entry {path} ({type(e).__name__}: {e}); "
              f"recomputing.")
        return None, MISS_CORRUPT

    if meta.get("payload") != payload:
        # Should be unreachable: the key IS the hash of the payload.  If it
        # happens, something has collided or an entry was hand-edited.
        print(f"[label-cache] PAYLOAD MISMATCH under key {key}; recomputing. "
              f"stored={meta.get('payload')!r}")
        return None, MISS_STALE

    if rewards.ndim != 1 or rewards.shape[0] != expect_len:
        print(f"[label-cache] length mismatch: cached {rewards.shape}, "
              f"expected ({expect_len},); recomputing.")
        return None, MISS_CORRUPT
    if not np.all(np.isfinite(rewards)):
        print("[label-cache] cached labels contain non-finite values; "
              "recomputing.")
        return None, MISS_CORRUPT

    stored_inv = meta.get("inventory")
    if inventory_now:
        if stored_inv != inventory_now:
            print(f"[label-cache] STALE: the source artefacts under this run dir "
                  f"have changed since the labels were cached "
                  f"({len(stored_inv or [])} files stored, "
                  f"{len(inventory_now)} now); recomputing and overwriting.")
            return None, MISS_STALE
        print(f"[label-cache] HIT {key} — {expect_len} labels, inventory "
              f"VERIFIED against {len(inventory_now)} source file(s).")
        return rewards.astype(np.float32, copy=False), HIT

    print(f"[label-cache] HIT {key} — {expect_len} labels.")
    print(f"[label-cache] !! the source artefacts are GONE, so the inventory "
          f"could NOT be verified.  This is expected after 3.2.9's chain "
          f"deletion.  Provenance: cached {meta.get('cached_at')} from "
          f"{meta.get('source_dir')} on {meta.get('device')}, "
          f"{meta.get('n_draws')} draws, weights digest "
          f"{meta.get('weights_digest')}.")
    return rewards.astype(np.float32, copy=False), HIT_UNVERIFIED


def save(key, payload, rewards, inventory_now, meta=None):
    """Write one entry.  Atomic: temp file then rename."""
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got {rewards.shape}")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("refusing to cache non-finite reward labels")

    root = cache_root()
    os.makedirs(root, exist_ok=True)
    full = dict(meta or {})
    full.update({
        "payload": payload,
        "inventory": inventory_now,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_labels": int(rewards.shape[0]),
    })
    path = _entry_path(key)
    # The temp name must itself end in `.npz`: np.savez_compressed APPENDS the
    # extension when it is absent, so `foo.npz.tmp123` would land on disk as
    # `foo.npz.tmp123.npz` and the rename would miss it.
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, rewards=rewards,
                        meta=np.array(json.dumps(full, sort_keys=True)))
    os.replace(tmp, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"[label-cache] wrote {path} ({size_mb:.1f} MB, "
          f"{rewards.shape[0]} labels) — the source chains may now be deleted "
          f"(reward_label_cache.py --emit-rm).")
    return path


def entries():
    """[(key, meta), ...] for every entry in the cache root."""
    root = cache_root()
    if not os.path.isdir(root):
        return []
    out = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".npz"):
            continue
        try:
            with np.load(os.path.join(root, fn), allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
            out.append((fn[:-4], meta))
        except Exception as e:  # noqa: BLE001
            out.append((fn[:-4], {"ERROR": f"{type(e).__name__}: {e}"}))
    return out


# ---------------------------------------------------------------- CLI ----- #

def _cmd_list(emit_rm=False):
    rows = entries()
    if not rows:
        print(f"no cache entries under {cache_root()}")
        return 0
    print(f"{len(rows)} entry(ies) under {cache_root()}\n")
    deletable = []
    for key, meta in rows:
        if "ERROR" in meta:
            print(f"  {key}  !! {meta['ERROR']}")
            continue
        p = meta.get("payload", {})
        src = meta.get("source_dir", "?")
        alive = os.path.isdir(src) if src != "?" else False
        print(f"  {key}")
        print(f"    model    {p.get('model')}  ({p.get('kind')}, "
              f"logic v{p.get('logic_version')})")
        print(f"    env      {p.get('env_name')}   labels {meta.get('n_labels')}")
        print(f"    params   {p.get('params')}")
        print(f"    cached   {meta.get('cached_at')} on {meta.get('device')}, "
              f"{meta.get('n_draws')} draws")
        print(f"    source   {src}   [{'PRESENT' if alive else 'deleted'}]")
        if alive:
            deletable.append(src)
    if emit_rm and deletable:
        print("\n# Sources whose labels are cached.  REVIEW EACH LINE, then run\n"
              "# them yourself — this tool never deletes anything.")
        for d in deletable:
            print(f"rm -rf {d}/sampling_f")
    elif emit_rm:
        print("\n# nothing to emit: no cached entry has its source dir present.")
    return 0


def _cmd_selftest():
    import tempfile
    root = tempfile.mkdtemp(prefix="rlc_selftest_")
    os.environ[ENV_VAR] = root
    try:
        md = os.path.join(root, "run_seed0")
        os.makedirs(md)
        f1 = os.path.join(md, "a.bin")
        open(f1, "wb").write(b"x" * 100)
        inv = inventory([f1], md)
        assert inv == [["a.bin", 100]], inv

        params = {"alpha": 0.95, "n_samples": 500, "centre_draws": True}
        k, pay = make_key("bnn", md, "antmaze-large-diverse-v2", 5, params)

        # 1. cold miss
        r, st = load(k, pay, 5, inv)
        assert r is None and st == MISS, st

        # 2. save + verified hit, bit-exact
        vals = np.array([1.5, -2.0, 0.0, 3.25, -0.125], dtype=np.float32)
        save(k, pay, vals, inv, meta={"source_dir": md, "device": "cpu",
                                      "n_draws": 7, "weights_digest": "deadbeef"})
        r, st = load(k, pay, 5, inv)
        assert st == HIT and np.array_equal(r, vals) and r.dtype == np.float32

        # 3. any param change must MISS, not silently reuse
        for bad in ({"alpha": 0.75}, {"n_samples": 400}, {"centre_draws": False}):
            p2 = dict(params); p2.update(bad)
            k2, pay2 = make_key("bnn", md, "antmaze-large-diverse-v2", 5, p2)
            assert k2 != k, bad
            assert load(k2, pay2, 5, inv)[0] is None, bad
        # ...and so must a different env, model name, N, kind or logic version
        assert make_key("bnn", md, "antmaze-medium-play-v2", 5, params)[0] != k
        assert make_key("bnn", os.path.join(root, "run_seed1"), "antmaze-large-diverse-v2", 5, params)[0] != k
        assert make_key("bnn", md, "antmaze-large-diverse-v2", 6, params)[0] != k
        assert make_key("mr_ensemble", md, "antmaze-large-diverse-v2", 5, params)[0] != k
        saved_v = globals()["LOGIC_VERSION"]
        globals()["LOGIC_VERSION"] = saved_v + 1
        assert make_key("bnn", md, "antmaze-large-diverse-v2", 5, params)[0] != k
        globals()["LOGIC_VERSION"] = saved_v

        # 4. sources CHANGED under a live entry -> stale, not a silent hit.
        #    This is the 10.3 hazard: retraining into the same OUT_DIR.
        open(f1, "wb").write(b"y" * 200)
        r, st = load(k, pay, 5, inventory([f1], md))
        assert r is None and st == MISS_STALE, st

        # 5. sources GONE -> hit, flagged unverified (the point of the cache)
        os.remove(f1)
        r, st = load(k, pay, 5, [])
        assert st == HIT_UNVERIFIED and np.array_equal(r, vals), st
        r, st = load(k, pay, 5, None)
        assert st == HIT_UNVERIFIED, st

        # 6. wrong expected length must never be served
        assert load(k, pay, 4, None)[0] is None

        # 7. non-finite labels are refused at write time
        for bad in (np.array([1.0, np.nan], dtype=np.float32),
                    np.array([1.0, np.inf], dtype=np.float32)):
            try:
                save(k + "x", pay, bad, [])
                raise AssertionError("accepted non-finite labels")
            except ValueError:
                pass

        # 8. digest is order-sensitive and length-sensitive
        w1 = [[np.array([1.0, 2.0]), np.array([3.0])]]
        w2 = [[np.array([2.0, 1.0]), np.array([3.0])]]
        assert digest_weights(w1) != digest_weights(w2)
        assert digest_weights(w1) != digest_weights(w1 * 2)
        assert digest_weights(w1) == digest_weights(list(w1))

        # 9. entries() round-trips
        assert [e[0] for e in entries()] == [k], entries()
        print("selftest OK")
        return 0
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
        os.environ.pop(ENV_VAR, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show cache entries")
    ap.add_argument("--emit-rm", action="store_true",
                    help="with --list, PRINT (never run) deletion commands for "
                         "sources whose labels are cached")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _cmd_selftest()
    if a.list or a.emit_rm:
        return _cmd_list(emit_rm=a.emit_rm)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
