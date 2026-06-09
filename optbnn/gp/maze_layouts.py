"""Antmaze maze-layout recovery for the map-informed GP functional prior.

This module turns a D4RL Antmaze maze specification into the three plain,
serialisable objects the :class:`MapInformedGPPrior` needs:

    * ``free_mask`` — 2-D bool array, ``True`` where a cell is navigable.
    * ``scaling``   — world-units per grid cell (``maze_size_scaling``, ~4.0).
    * ``offset``    — ``(row_off, col_off)`` mapping world (x, y) → (row, col).

The affine map is the inverse of D4RL's authoritative ``_rowcol_to_xy``::

    x = col * scaling - init_torso_x   →   col = x / scaling + col_off
    y = row * scaling - init_torso_y   →   row = y / scaling + row_off

with ``col_off = init_torso_x / scaling`` and ``row_off = init_torso_y /
scaling``.  ``init_torso`` is the world position of the reset ('r') cell, so the
offsets are simply that cell's (row, col) indices.

Two ways to obtain a layout:

1. :func:`extract_maze_from_env` — introspect a live ``gym.make`` D4RL env.
   This is the **authoritative** path: it reads the exact maze the user's D4RL
   build ships, with the correct scaling/offset, and is the recommended default
   on a machine with D4RL installed.
2. Hardcoded :data:`ANTMAZE_MEDIUM_MAP` / :data:`ANTMAZE_LARGE_MAP` fallbacks,
   used when no env name is supplied.  **Verify these against your installed
   D4RL 1.1** with :mod:`scripts_bnn.diagnose_map_prior` before trusting them —
   maze arrays have historically differed between D4RL versions.

Goal handling: 'g'/'r' markers are treated as ordinary **free** cells.  The goal
cell is never given any special role — encoding it would leak the inferential
target (see the handoff, §10 "goal leakage").
"""

import numpy as np

# Marker symbols used by D4RL maze specifications.
RESET = "r"
GOAL = "g"

# Default antmaze world-units per cell (D4RL `maze_size_scaling` for ant envs).
ANTMAZE_SIZE_SCALING = 4.0

# ---------------------------------------------------------------------------
# Hardcoded fallback layouts (D4RL locomotion maze_env.py).
#
# 1 = wall, 0 = free, 'r' = reset (free), 'g' = goal (free).  Shared by the
# play/diverse variants of each size.  VERIFY against your installed D4RL — the
# env-extraction path (extract_maze_from_env) is preferred and authoritative.
# ---------------------------------------------------------------------------

# antmaze-medium-{play,diverse} — D4RL "BIG_MAZE".
ANTMAZE_MEDIUM_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, RESET, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, GOAL, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, GOAL, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

# antmaze-large-{play,diverse} — D4RL "HARDEST_MAZE".
ANTMAZE_LARGE_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, RESET, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, GOAL, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
]

_HARDCODED = {
    "medium": ANTMAZE_MEDIUM_MAP,
    "large": ANTMAZE_LARGE_MAP,
}


# ---------------------------------------------------------------------------
# Layout extraction
# ---------------------------------------------------------------------------

def _is_wall(cell):
    """A cell is a wall iff it equals 1 (int or '1').  Markers/0 are free."""
    if isinstance(cell, str):
        return cell.strip() == "1"
    return int(cell) == 1


def _find_reset_cell(maze_map):
    """Return (row, col) of the reset ('r') cell, or None if absent."""
    for r, row in enumerate(maze_map):
        for c, cell in enumerate(row):
            if isinstance(cell, str) and cell.strip().lower() == RESET:
                return (r, c)
    return None


def maze_map_to_arrays(maze_map, scaling=ANTMAZE_SIZE_SCALING,
                       init_torso_x=None, init_torso_y=None):
    """Convert a maze-map (list of lists) into (free_mask, scaling, offset).

    Args:
        maze_map: list-of-lists of {1, 0, 'r', 'g'} (walls / free / markers).
        scaling: world-units per cell.
        init_torso_x/init_torso_y: optional world coords of the reset cell.
            When omitted, they are derived from the reset marker's grid index
            (init_torso = reset_index * scaling), which is exactly how D4RL
            defines them.

    Returns:
        free_mask: (R, C) bool array, True where navigable.
        scaling:   float, echoed back.
        offset:    (row_off, col_off) float tuple for the world→cell map.
    """
    free_mask = np.array(
        [[not _is_wall(cell) for cell in row] for row in maze_map],
        dtype=bool,
    )

    if init_torso_x is None or init_torso_y is None:
        reset = _find_reset_cell(maze_map)
        if reset is None:
            raise ValueError(
                "maze_map has no reset ('r') marker and no explicit "
                "init_torso_x/init_torso_y were given — cannot determine the "
                "world→cell offset."
            )
        row_off, col_off = float(reset[0]), float(reset[1])
    else:
        col_off = float(init_torso_x) / float(scaling)
        row_off = float(init_torso_y) / float(scaling)

    return free_mask, float(scaling), (row_off, col_off)


def extract_maze_from_env(env_name):
    """Introspect a live D4RL env to recover its maze layout.

    Reads the maze array, the size scaling, and the reset/init-torso position
    straight from the environment object, so the result matches whatever maze
    the installed D4RL build actually ships.  Tries several attribute names to
    stay robust across D4RL versions.

    Args:
        env_name: a D4RL gym id, e.g. ``"antmaze-medium-play-v2"``.

    Returns:
        (free_mask, scaling, offset) — see maze_map_to_arrays.
    """
    import gym  # local import: only needed on a machine with D4RL installed
    import d4rl  # noqa: F401  (registers the antmaze envs)

    env = gym.make(env_name)
    u = env.unwrapped

    maze_map = None
    for attr in ("_maze_map", "maze_map", "_maze_arr", "maze_arr"):
        if hasattr(u, attr):
            maze_map = getattr(u, attr)
            break
    if maze_map is None:
        raise AttributeError(
            f"Could not find a maze-map attribute on {env_name!r}'s unwrapped "
            "env (tried _maze_map / maze_map / _maze_arr / maze_arr).  Inspect "
            "`gym.make(env_name).unwrapped` and pass the layout explicitly."
        )
    # Normalise numpy arrays to nested lists so marker dtype handling is uniform.
    if isinstance(maze_map, np.ndarray):
        maze_map = maze_map.tolist()

    scaling = ANTMAZE_SIZE_SCALING
    for attr in ("_maze_size_scaling", "maze_size_scaling"):
        if hasattr(u, attr):
            scaling = float(getattr(u, attr))
            break

    init_x = getattr(u, "_init_torso_x", None)
    init_y = getattr(u, "_init_torso_y", None)

    return maze_map_to_arrays(
        maze_map, scaling=scaling, init_torso_x=init_x, init_torso_y=init_y
    )


def get_antmaze_layout(map_size, env_name=None):
    """Return (free_mask, scaling, offset) for an antmaze map.

    Args:
        map_size: ``"medium"`` or ``"large"`` — selects the hardcoded fallback.
        env_name: optional D4RL gym id.  When given, the layout is extracted
            from the live env (authoritative) and ``map_size`` is ignored.

    Returns:
        (free_mask, scaling, offset).
    """
    if env_name is not None:
        return extract_maze_from_env(env_name)

    key = str(map_size).lower()
    if key not in _HARDCODED:
        raise ValueError(
            f"Unknown map_size={map_size!r}; expected 'medium' or 'large' "
            "(or pass env_name to extract the layout from a live D4RL env)."
        )
    return maze_map_to_arrays(_HARDCODED[key])


# ---------------------------------------------------------------------------
# Graph + heat (diffusion) kernel over the maze free-space grid
# ---------------------------------------------------------------------------

def build_maze_graph(free_mask):
    """Build the 4-connected free-space grid graph.

    Nodes are navigable cells, indexed ``0..N-1`` in row-major order.  Edges
    connect orthogonally adjacent free cells (no diagonals — 8-connectivity
    would let correlation leak around wall corners; see handoff §10).

    Args:
        free_mask: (R, C) bool array, True where navigable.

    Returns:
        adjacency: (N, N) float64 0/1 symmetric adjacency matrix.
        rc_to_idx: dict ``(row, col) -> node_idx``.
        free_rc:   (N, 2) int array of each node's (row, col).
    """
    free_mask = np.asarray(free_mask, dtype=bool)
    rc_to_idx = {}
    free_rc = []
    for r in range(free_mask.shape[0]):
        for c in range(free_mask.shape[1]):
            if free_mask[r, c]:
                rc_to_idx[(r, c)] = len(rc_to_idx)
                free_rc.append((r, c))
    n = len(rc_to_idx)
    adjacency = np.zeros((n, n), dtype=np.float64)
    for (r, c), i in rc_to_idx.items():
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = rc_to_idx.get((r + dr, c + dc))
            if j is not None:
                adjacency[i, j] = 1.0
    return adjacency, rc_to_idx, np.asarray(free_rc, dtype=np.int64)


def heat_kernel(adjacency, eta):
    """Graph heat (diffusion) kernel ``K_geo = exp(-eta L)``.

    Uses the symmetric normalised Laplacian ``L = I - D^{-1/2} A D^{-1/2}`` and
    an eigendecomposition, so ``exp(-eta L)`` is PSD by construction for any
    ``eta > 0`` (unlike plugging graph distance into an RBF; see handoff §4).

    Args:
        adjacency: (N, N) symmetric adjacency matrix.
        eta: diffusion time (> 0); larger ⇒ longer spatial correlation length.

    Returns:
        (N, N) float64 PSD heat-kernel matrix.
    """
    A = np.asarray(adjacency, dtype=np.float64)
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.clip(d, 1e-12, None))
    L = np.eye(A.shape[0]) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    w, U = np.linalg.eigh(0.5 * (L + L.T))  # symmetrise against fp drift
    Kgeo = (U * np.exp(-eta * w)) @ U.T
    return 0.5 * (Kgeo + Kgeo.T)
