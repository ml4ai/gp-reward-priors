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
| `mechanism_stage_a.txt` | why the H3 gate fails, stage A (desk; `scripts_bnn/calibration_mechanism.py`), under the author's scoped exception of 2026-11-06 (handoff §14.92–§14.93) |
| `mechanism_stage_a2.txt` | stage A2: what the errors are (human-label proxies, novelty; `calibration_mechanism.py --a2`), handoff §14.94 |
| `mechanism_stage_a3.txt` | stage A3: novelty in the network's input space (`calibration_mechanism.py --a3`), handoff §14.95 |
| `mechanism_stage_b.txt` | stage B: temperatures of all four families, the BNN's chains, the prior's exact spread (`results/calibration_stage_b.py analyse`), handoff §14.98 |
| `mechanism_stage_b3.txt` | stage B3: positional vs within-cell parts of the BNN's and MR's per-step reward (`results/calibration_stage_b3.py analyse`), handoff §14.102 |
| `mechanism_selection.txt` | the round-5 BNN selection sweeps' trials: softness vs the selection score and the searched settings (`calibration_mechanism.py --selection`), handoff §14.103 |

The parent repo's outputs are stashed in `~/iqlpref/results/held_bnn_cvar/`.
