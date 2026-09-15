"""Within-run checkpoint selection for the MR and PT reward models (handoff 4.3.107).

Two splits, two jobs:

  * the SELECTION split (``select_split``, default the seed's **test** file)
    picks the checkpoint that is saved as ``best_model.pt`` -- the model IQL
    loads and that the reward model is evaluated with downstream;
  * the VALIDATION split scores that checkpoint, and that score
    (``eval_loss_at_selected``) is the stage-1 sweep objective.

Before 2026-09-15 the validation split did both, so the sweep objective was the
minimum of a noisy validation curve that the same split had also picked -- an
optimistic score that large, fast-memorising models exploited (4.3.106).
``select_split="val"`` reproduces that old rule.

Tie-break: the primary metric is ``criteria_key``; on an exact tie the secondary
metric is compared against the value AT THE SELECTED CHECKPOINT.  (The old
inline code compared against the best secondary seen at any epoch; the two
differ only on exact float ties.)
"""

import math


def is_better(loss, acc, best_loss, best_acc, criteria_key):
    """True if (loss, acc) should replace the selected checkpoint's (best_loss, best_acc)."""
    if criteria_key == "loss":
        if math.isnan(loss):
            return False
        return loss < best_loss or (loss == best_loss and acc > best_acc)
    if criteria_key == "acc":
        if math.isnan(acc):
            return False
        return acc > best_acc or (acc == best_acc and loss < best_loss)
    raise ValueError(f"criteria_key must be 'loss' or 'acc', got {criteria_key!r}")


class CheckpointSelector:
    """Track the checkpoint chosen on the selection split and the validation score at it."""

    def __init__(self, criteria_key):
        if criteria_key not in ("loss", "acc"):
            raise ValueError(f"criteria_key must be 'loss' or 'acc', got {criteria_key!r}")
        self.criteria_key = criteria_key
        self.best_epoch = None
        self.select_loss = math.inf
        self.select_acc = -math.inf
        self.val_loss = math.nan
        self.val_acc = math.nan

    def update(self, epoch, select_loss, select_acc, val_loss, val_acc):
        """Offer one evaluation; returns True if it becomes the selected checkpoint."""
        if not is_better(select_loss, select_acc, self.select_loss, self.select_acc, self.criteria_key):
            return False
        self.best_epoch = epoch
        self.select_loss, self.select_acc = select_loss, select_acc
        self.val_loss, self.val_acc = val_loss, val_acc
        return True

    def log_dict(self):
        if self.best_epoch is None:
            return {}
        return {
            "best_epoch": self.best_epoch,
            "eval_loss_at_selected": self.val_loss,
            "eval_acc_at_selected": self.val_acc,
            "select_loss_at_selected": self.select_loss,
            "select_acc_at_selected": self.select_acc,
        }


def _old_inline_rule(evals, criteria_key):
    """The pre-2026-09-15 inline logic, verbatim in structure, for the self-test."""
    best_acc, best_loss, best_epoch = -math.inf, math.inf, 0
    for epoch, loss, acc in evals:
        if criteria_key == "acc":
            if acc > best_acc:
                best_epoch, best_acc = epoch, acc
                if loss < best_loss:
                    best_loss = loss
            elif acc == best_acc:
                if loss < best_loss:
                    best_epoch, best_loss = epoch, loss
            elif loss < best_loss:
                best_loss = loss
        else:
            if loss < best_loss:
                best_epoch, best_loss = epoch, loss
                if acc > best_acc:
                    best_acc = acc
            elif loss == best_loss:
                if acc > best_acc:
                    best_epoch, best_acc = epoch, acc
            elif acc > best_acc:
                best_acc = acc
    return best_epoch


def _selftest():
    import random

    rng = random.Random(0)
    # 1. With continuous values (no exact ties), select_split="val" reproduces the old rule.
    for key in ("loss", "acc"):
        for _ in range(2000):
            evals = [(e, rng.random(), rng.random()) for e in range(0, 200, 5)]
            sel = CheckpointSelector(key)
            for e, l, a in evals:
                sel.update(e, l, a, l, a)
            assert sel.best_epoch == _old_inline_rule(evals, key), key
    # 2. Selection and scoring are decoupled: the val score is taken AT the selected epoch.
    sel = CheckpointSelector("loss")
    sel.update(0, 0.9, 0.5, 0.30, 0.6)
    sel.update(5, 0.5, 0.7, 0.80, 0.4)   # selection split improves, val worsens
    sel.update(10, 0.6, 0.9, 0.10, 0.9)  # val improves, selection split does not
    assert sel.best_epoch == 5 and sel.val_loss == 0.80 and sel.select_loss == 0.5
    # 3. Exact-tie semantics: secondary compared at the selected checkpoint.
    sel = CheckpointSelector("loss")
    sel.update(0, 0.5, 0.6, 0.0, 0.0)
    sel.update(5, 0.4, 0.5, 0.0, 0.0)    # selected; acc at selection 0.5
    sel.update(10, 0.4, 0.55, 0.0, 0.0)  # tie on loss, better acc than AT selection
    assert sel.best_epoch == 10          # the old rule kept epoch 5 (best-ever acc 0.6)
    # 4. NaN never selected; a run of NaNs selects nothing.
    sel = CheckpointSelector("loss")
    sel.update(0, math.nan, 0.5, 0.1, 0.5)
    assert sel.best_epoch is None and sel.log_dict() == {}
    sel.update(5, 0.7, 0.5, 0.2, 0.5)
    assert sel.best_epoch == 5
    print("checkpoint_selection self-test: PASS (4 checks, 4000 randomized old-rule comparisons)")


if __name__ == "__main__":
    _selftest()
