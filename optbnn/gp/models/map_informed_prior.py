"""Map-informed Gaussian-process functional prior for D4RL Antmaze rewards.

A manually-designed, goal-agnostic GP prior over the scalar reward
``r(s, a) -> R`` whose informativeness lives entirely in its kernel (zero mean).
The kernel is a sum of a constant offset and a wall-respecting heat-kernel term::

    k((s,a),(s',a')) = sig_c2  +  sig_g2 * K_geo[c(s), c(s')]  +  sig_n2 * 1[i=j]

where ``c(s)`` maps a state to its discrete maze cell via the torso (x, y) and
``K_geo`` is a graph diffusion (heat) kernel on the maze free-space grid, so
reward correlations flow **along corridors, not through walls**.

This object is a drop-in alternative to :class:`optbnn.gp.models.model.LCFModel`
for the fSGHMC pipeline: it exposes the same ``functional_prior_grad(net, X_M,
aux_X, jitter)`` and ``.to(device)`` contract that :class:`FPrefNet` relies on,
so it can be toggled in ``run_bnn_training.py`` with no change to the sampler.

The prior is **manually designed** — it is never pretrained, fit to data, or
tuned on downstream return.  Its only inputs are the known maze layout (walls)
and a handful of fixed hyperparameters.  The goal cell is never used.

It is also a standalone, serialisable object (``save``/``load``) carrying only
the layout + hypers (no data), so the *same* prior can be reused verbatim as the
reference measure mu_0 in the downstream Gibbs-measure transfer method.

See ``handoff_mapinformed_gp_functional_prior_antmaze.md``.
"""

import numpy as np
import torch

from ..maze_layouts import build_maze_graph, heat_kernel


class MapInformedGPPrior(torch.nn.Module):
    """Goal-agnostic, wall-respecting GP prior over Antmaze reward.

    Args:
        free_mask: (R, C) bool array — True where a maze cell is navigable.
        scaling: world-units per grid cell (D4RL ``maze_size_scaling``, ~4.0).
        offset: ``(row_off, col_off)`` for the world→cell affine map
            ``col = x/scaling + col_off``, ``row = y/scaling + row_off``.
        eta: heat-kernel diffusion time (> 0); sets the spatial correlation
            length.  Fixed per map *size*, never tuned on reward.
        sig_c2: constant-offset variance (absorbs the Bradley-Terry additive
            constant the reward is only identifiable up to).
        sig_g2: map-informed signal variance (prior reward scale).
        sig_n2: nugget / diagonal jitter — mandatory for invertibility of K.
        xy_cols: which two columns of the input hold the torso (x, y).
        xy_source: ``"obs"`` to read (x, y) from the BNN input X_M, or
            ``"aux"`` to read them from aux_X.  Antmaze ``obs[:, :2]`` are the
            torso x, y, so the default is ``"obs"`` / ``(0, 1)``.  **Never read
            the goal columns** — that would leak the inferential target.
        device: torch device for the kernel buffers.
        name: optional label (informational only).
    """

    def __init__(self, free_mask, scaling, offset, eta, sig_c2, sig_g2, sig_n2,
                 xy_cols=(0, 1), xy_source="obs", device=None, name=None):
        super().__init__()
        self.name = name
        self.free_mask = np.asarray(free_mask, dtype=bool)
        self.scaling = float(scaling)
        self.offset = (float(offset[0]), float(offset[1]))
        self.eta = float(eta)
        self.sig_c2 = float(sig_c2)
        self.sig_g2 = float(sig_g2)
        self.sig_n2 = float(sig_n2)
        self.xy_cols = (int(xy_cols[0]), int(xy_cols[1]))
        self.xy_source = str(xy_source)
        if self.xy_source not in ("obs", "aux"):
            raise ValueError(f"xy_source must be 'obs' or 'aux', got {xy_source!r}")

        # Graph + heat kernel are fixed: compute once at construction (handoff
        # §10 "Recomputing Kgeo every step").
        self._adjacency, self.rc_to_idx, self.free_rc = build_maze_graph(self.free_mask)
        Kgeo = heat_kernel(self._adjacency, self.eta)  # (N, N) float64

        # Register as a buffer so .to(device) moves it with the module.
        self.register_buffer("Kgeo", torch.from_numpy(Kgeo).double())
        # free_rc as a buffer too, for vectorised nearest-free clamping on-device.
        self.register_buffer(
            "_free_rc_t", torch.from_numpy(self.free_rc).long()
        )
        self.n_cells = Kgeo.shape[0]

        if device is not None:
            self.to(device)

    @property
    def device(self):
        return self.Kgeo.device

    # ------------------------------------------------------------------
    # State -> cell mapping  c(s)
    # ------------------------------------------------------------------

    def _extract_xy(self, X_M, aux_X):
        """Pull the (n, 2) torso (x, y) out of the chosen input source."""
        src = aux_X if self.xy_source == "aux" else X_M
        if src is None:
            raise ValueError(
                f"xy_source={self.xy_source!r} but the corresponding input is "
                "None.  Pass aux_X when xy_source='aux'."
            )
        if isinstance(src, np.ndarray):
            src = torch.from_numpy(src)
        return src[:, [self.xy_cols[0], self.xy_cols[1]]].detach().float()

    def cell_of(self, X_M, aux_X=None):
        """Map a batch of inputs to free-cell node indices (clamped).

        Out-of-range or wall cells are clamped to the nearest free node by grid
        (row, col) distance.

        Args:
            X_M: (n, d) input tensor/array.
            aux_X: optional (n, k) aux tensor/array (used when xy_source='aux').

        Returns:
            (n,) int64 numpy array of node indices into ``Kgeo``.
        """
        xy = self._extract_xy(X_M, aux_X).cpu().numpy()  # (n, 2) = (x, y)
        x, y = xy[:, 0], xy[:, 1]
        row_off, col_off = self.offset
        col = np.rint(x / self.scaling + col_off).astype(np.int64)
        row = np.rint(y / self.scaling + row_off).astype(np.int64)

        idx = np.array(
            [self.rc_to_idx.get((int(r), int(c)), -1) for r, c in zip(row, col)],
            dtype=np.int64,
        )

        # Clamp misses (wall / out-of-range) to the nearest free node.
        miss = idx < 0
        if miss.any():
            q = np.stack([row[miss], col[miss]], axis=1)[:, None, :]  # (m, 1, 2)
            d2 = ((q - self.free_rc[None, :, :]) ** 2).sum(axis=2)     # (m, N)
            idx[miss] = np.argmin(d2, axis=1)
        return idx

    # ------------------------------------------------------------------
    # Kernel / Gram matrix
    # ------------------------------------------------------------------

    def _gram_from_idx(self, idx, extra_jitter=0.0):
        """Assemble K (with nugget) for the given node indices.

        K[i,j] = sig_c2 + sig_g2 * Kgeo[idx_i, idx_j]; diag += sig_n2 (+ extra).
        Returns an (n, n) float64 tensor on ``self.device``.
        """
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        K = self.sig_c2 + self.sig_g2 * self.Kgeo[idx_t][:, idx_t]
        n = K.shape[0]
        if n > 0:
            diag = self.sig_n2 + float(extra_jitter)
            K = K + diag * torch.eye(n, dtype=K.dtype, device=K.device)
        return K

    def gram(self, X_M, aux_X=None, extra_jitter=0.0):
        """Full Gram matrix K(X_M, X_M) for arbitrary (s, a) inputs (§5).

        Returns an (n, n) float64 tensor including the nugget.
        """
        idx = self.cell_of(X_M, aux_X)
        return self._gram_from_idx(idx, extra_jitter=extra_jitter)

    def _solve(self, K, rhs):
        """Cholesky solve K alpha = rhs (double precision).  rhs: (n,) or (n,1)."""
        n = K.shape[0]
        if n == 0:
            return rhs.new_zeros(rhs.shape)
        L = torch.linalg.cholesky(K)
        return torch.cholesky_solve(rhs.double().view(n, -1), L).view_as(rhs.double())

    # ------------------------------------------------------------------
    # Path A — functional prior gradient (drop-in for LCFModel)
    # ------------------------------------------------------------------

    def functional_prior_grad(self, net, X_M, aux_X=None, jitter=0.0):
        """fSGHMC functional GP prior gradient w.r.t. ``net``'s parameters.

            grad_w log p_GP(f(.; w)) = -J_w(X_M)^T K^{-1} (f(X_M; w) - m(X_M))

        with zero mean ``m = 0``.  The (n, n) Cholesky solve gives
        ``alpha = K^{-1} f``; a single VJP back through the BNN yields the
        weight-space gradient.  The return value is ready to be **added** to the
        likelihood gradient (it negates the VJP so it equals grad log p_GP).

        Mirrors :meth:`LCFModel.functional_prior_grad` exactly so it is a drop-in
        replacement in :class:`FPrefNet`.  ``jitter`` is added to the diagonal on
        top of the structural nugget ``sig_n2`` for extra Cholesky robustness.

        Args:
            net: the BNN whose parameters we differentiate.
            X_M: (n_M, d) measurement inputs on net's device.
            aux_X: optional aux inputs (used when xy_source='aux').
            jitter: extra absolute diagonal jitter for the solve.

        Returns:
            Tuple of per-parameter gradient tensors (zeros for unused params).
        """
        f_XM = net(X_M)  # (n_M, 1) — keep graph for the VJP

        with torch.no_grad():
            idx = self.cell_of(X_M, aux_X)
            K = self._gram_from_idx(idx, extra_jitter=jitter)
            alpha = self._solve(K, f_XM.detach())  # (n_M, 1) double

        grad_out = alpha.to(f_XM.dtype).view_as(f_XM)

        vjp = torch.autograd.grad(
            outputs=f_XM,
            inputs=list(net.parameters()),
            grad_outputs=grad_out,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        return tuple(
            (-g if g is not None else torch.zeros_like(p))
            for g, p in zip(vjp, net.parameters())
        )

    def prior_grad(self, f_M, X_M, aux_X=None, jitter=0.0):
        """d log mu0 / d f_M = -K^{-1} (f_M - m) = -K^{-1} f_M  (mean 0).

        The gradient of the GP log-density w.r.t. the function values
        themselves (Section 6), useful for direct function-space updates.
        """
        K = self.gram(X_M, aux_X, extra_jitter=jitter)
        return -self._solve(K, f_M)

    # ------------------------------------------------------------------
    # Path B — functional regulariser / negative log density
    # ------------------------------------------------------------------

    def neg_log_density(self, f_M, X_M, aux_X=None, jitter=0.0):
        """Negative GP log-density (up to constants): 0.5 f^T K^{-1} f.

        The function-space regulariser to add to an ELBO (Path B), with zero
        mean.  Returns a scalar tensor.
        """
        K = self.gram(X_M, aux_X, extra_jitter=jitter)
        f = f_M.double().view(-1, 1)
        alpha = self._solve(K, f)
        return 0.5 * (f.view(-1) @ alpha.view(-1))

    def log_prior(self, X_M, f_XM, aux_X=None, jitter=0.0):
        """Full GP log-density log p_GP(f_XM) including the log-det term.

            log p = -0.5 f^T K^{-1} f - 0.5 log|K| - n/2 log(2 pi)

        Signature mirrors :meth:`LCFModel.log_prior`.
        """
        K = self.gram(X_M, aux_X, extra_jitter=jitter)
        n = K.shape[0]
        if n == 0:
            return torch.zeros((), dtype=torch.float64, device=self.device)
        f = f_XM.double().view(-1, 1)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(f, L)
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        return (
            -0.5 * (f.view(-1) @ alpha.view(-1))
            - 0.5 * logdet
            - 0.5 * n * torch.log(torch.tensor(2.0 * np.pi, dtype=torch.float64))
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def sample_prior(self, X_M, num_samples, aux_X=None, jitter=1e-8):
        """Draw GP prior function samples at X_M (zero mean).

        Returns an (n, num_samples) tensor.  For the heatmap diagnostic, pass
        the centroids of all free cells (see :meth:`free_cell_inputs`).
        """
        K = self.gram(X_M, aux_X, extra_jitter=jitter)
        n = K.shape[0]
        L = torch.linalg.cholesky(K)
        V = torch.randn(n, num_samples, dtype=L.dtype, device=L.device)
        return L @ V

    def free_cell_inputs(self):
        """World (x, y) at the centre of every free cell, in node order.

        Returns an (N, 2) float64 array suitable as X_M (with xy_cols=(0,1)) for
        the prior-sample heatmap diagnostic.
        """
        row_off, col_off = self.offset
        rows = self.free_rc[:, 0].astype(np.float64)
        cols = self.free_rc[:, 1].astype(np.float64)
        x = (cols - col_off) * self.scaling
        y = (rows - row_off) * self.scaling
        return np.stack([x, y], axis=1)

    # ------------------------------------------------------------------
    # Serialisation  (layout + hypers only — never any data)
    # ------------------------------------------------------------------

    def to_args(self):
        """Plain-Python/numpy dict for pickling across mp.spawn and save/load."""
        return {
            "prior_type": "map_informed",
            "free_mask": self.free_mask,
            "scaling": self.scaling,
            "offset": np.asarray(self.offset, dtype=np.float64),
            "eta": self.eta,
            "sig_c2": self.sig_c2,
            "sig_g2": self.sig_g2,
            "sig_n2": self.sig_n2,
            "xy_cols": np.asarray(self.xy_cols, dtype=np.int64),
            "xy_source": self.xy_source,
        }

    @classmethod
    def from_args(cls, args, device=None):
        """Reconstruct from a :meth:`to_args` dict (rebuilds graph + kernel)."""
        return cls(
            free_mask=args["free_mask"],
            scaling=float(args["scaling"]),
            offset=tuple(np.asarray(args["offset"]).tolist()),
            eta=float(args["eta"]),
            sig_c2=float(args["sig_c2"]),
            sig_g2=float(args["sig_g2"]),
            sig_n2=float(args["sig_n2"]),
            xy_cols=tuple(np.asarray(args["xy_cols"]).tolist()),
            xy_source=str(args["xy_source"]),
            device=device,
        )

    def save(self, path):
        """Serialise the prior (layout + hypers) to a .npz file."""
        np.savez(
            path,
            free_mask=self.free_mask,
            scaling=self.scaling,
            offset=np.asarray(self.offset, dtype=np.float64),
            eta=self.eta,
            sig_c2=self.sig_c2,
            sig_g2=self.sig_g2,
            sig_n2=self.sig_n2,
            xy_cols=np.asarray(self.xy_cols, dtype=np.int64),
            xy_source=self.xy_source,
        )

    @classmethod
    def load(cls, path, device=None):
        """Load a prior saved with :meth:`save`."""
        d = np.load(path, allow_pickle=False)
        return cls(
            free_mask=d["free_mask"],
            scaling=float(d["scaling"]),
            offset=tuple(d["offset"].tolist()),
            eta=float(d["eta"]),
            sig_c2=float(d["sig_c2"]),
            sig_g2=float(d["sig_g2"]),
            sig_n2=float(d["sig_n2"]),
            xy_cols=tuple(d["xy_cols"].tolist()),
            xy_source=str(d["xy_source"]),
            device=device,
        )
