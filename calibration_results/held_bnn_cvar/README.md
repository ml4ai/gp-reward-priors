# Held: outputs covered by the BNN/CVaR hold

Moved here on 2026-10-21 under the BNN/CVaR hold (`HANDOFF_HP_SELECTION.md` §14.1, decided
by the author on 2026-10-20). Not quoted anywhere until the author lifts the hold. Nothing
was deleted. The raw per-pair measurements (`../measurements/`) and the measurement log stay
where they are, as raw data.

| file | covered content |
|---|---|
| `anchor_gate.txt`, `anchor_check.txt`, `anchor_posthoc.txt` | the BNN's H3 calibration gate, its check and its post-hoc diagnostic |
| `goal_structure_run1.txt`, `goal_structure_run2.txt` | reward maps for MR, PT **and** BNN (mixed; `results/goal_structure.py` now defaults to MR and PT) |
| `heuristic_floor.txt` | trivial-heuristic floor for MR, PT **and** BNN (mixed; `scripts_bnn/heuristic_floor.py` now prints MR and PT only) |

The parent repo's outputs are stashed in `~/iqlpref/results/held_bnn_cvar/`.
