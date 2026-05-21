import os

import numpy as np
import torch
import torch.nn as nn
import wandb

from ..utils.util import ensure_dir, prepare_device


def gradient_explosion(model: nn.Module) -> bool:
    """Return True if any parameter gradient contains non-finite values."""
    return any(
        not torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.grad is not None
    )


def _sqrt_psd(K: torch.Tensor) -> torch.Tensor:
    """Compute the matrix square-root of a PSD matrix via eigendecomposition.

    PERF: factored out so it can be called from both aux and non-aux paths
    without code duplication.  Also avoids recomputing relu inside the call.
    """
    evalues, evectors = torch.linalg.eigh(K)
    sqrt_evalues = torch.relu(evalues).sqrt_()
    return evectors @ torch.diag_embed(sqrt_evalues) @ evectors.transpose(-2, -1)


def _sq_wasserstein2(bnn_K: torch.Tensor, target_K: torch.Tensor) -> torch.Tensor:
    """Squared 2-Wasserstein distance between two zero-mean Gaussians.

    W_2^2(N(0, A), N(0, B)) = tr(A + B - 2*(A^{1/2} B A^{1/2})^{1/2})

    PERF: shared implementation for both aux / non-aux branches.
    """
    sqrt_target_K = _sqrt_psd(target_K)
    # fidelity matrix: (sqrt_A @ B @ sqrt_A)^{1/2}
    M = sqrt_target_K @ bnn_K @ sqrt_target_K
    fidelity = _sqrt_psd(M)
    # PERF: compute three separate traces instead of allocating a full n×n sum
    # matrix just to trace it.  Saves one O(n²) allocation + two n² adds.
    return target_K.trace() + bnn_K.trace() - 2.0 * fidelity.trace()


class MapperWassersteinGP(object):
    def __init__(
        self,
        gp,
        bnn,
        data_generator,
        out_dir,
        input_dim=1,
        output_dim=1,
        n_data=256,
        logger=None,
        n_gpu=0,
        gpu_gp=False,
    ):
        self.gp = gp
        self.bnn = bnn
        self.data_generator = data_generator
        self.n_data = n_data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out_dir = out_dir
        self.device, device_ids = prepare_device(n_gpu)
        self.gpu_gp = gpu_gp

        if gpu_gp:
            self.gp = self.gp.to(self.device)
        self.bnn = self.bnn.to(self.device)
        if len(device_ids) > 1:
            if self.gpu_gp:
                self.gp = torch.nn.DataParallel(self.gp, device_ids=device_ids)
            self.bnn = torch.nn.DataParallel(self.bnn, device_ids=device_ids)

        self.print_info = print if logger is None else logger.info
        self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        ensure_dir(self.ckpt_dir)

    # ------------------------------------------------------------------
    # PERF: the original code had two nearly-identical copies of compute_sqw2
    # (one for has_aux, one without) defined inside optimize(), causing the
    # closure to be re-created on every call and duplicating ~30 lines.
    # We define them once here as methods.
    # ------------------------------------------------------------------

    def _compute_sqw2_aux(self, X: torch.Tensor, aux_X: torch.Tensor) -> torch.Tensor:
        # PERF: X and aux_X are already on self.device (moved once before vmap).
        bnn_module = self.bnn.module if isinstance(self.bnn, nn.DataParallel) else self.bnn
        gp_module = self.gp.module if isinstance(self.gp, nn.DataParallel) else self.gp

        bnn_K = bnn_module.compute_covariance(X.double())
        if self.gpu_gp:
            target_K = gp_module.compute_covariance(X.double(), aux_X.double())
        else:
            target_K = gp_module.compute_covariance(
                X.cpu().double(), aux_X.cpu().double()
            ).to(self.device)

        return _sq_wasserstein2(bnn_K, target_K)

    def _compute_sqw2(self, X: torch.Tensor) -> torch.Tensor:
        # PERF: X is already on self.device (moved once before vmap).
        bnn_module = self.bnn.module if isinstance(self.bnn, nn.DataParallel) else self.bnn
        gp_module = self.gp.module if isinstance(self.gp, nn.DataParallel) else self.gp

        bnn_K = bnn_module.compute_covariance(X.double())
        if self.gpu_gp:
            target_K = gp_module.compute_covariance(X.double())
        else:
            target_K = gp_module.compute_covariance(X.cpu().double()).to(self.device)

        return _sq_wasserstein2(bnn_K, target_K)

    def optimize(
        self,
        num_iters,
        batches=10,
        lr=1e-2,
        save_ckpt_every=50,
        print_every=10,
    ):
        wdist_hist = []
        prior_optimizer = torch.optim.Adam(
            self.bnn.parameters(), lr=lr, weight_decay=0.001
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            prior_optimizer, factor=0.01
        )

        has_aux = self.data_generator.has_aux
        # PERF: pick the right vmap-able function once rather than branching
        # inside the loop.
        compute_fn = self._compute_sqw2_aux if has_aux else self._compute_sqw2

        best_wdist = np.inf
        last_it = 1

        for it in range(1, num_iters + 1):
            last_it = it

            # PERF: move the full batch to device once here rather than
            # transferring each slice inside the vmapped function (which would
            # issue one CPU→GPU transfer per batch element).
            if has_aux:
                X_batch, aux_X_batch = self.data_generator.get_batches(
                    self.n_data, batches
                )
                X_batch = X_batch.to(self.device, non_blocking=True)
                aux_X_batch = aux_X_batch.to(self.device, non_blocking=True)
                assert torch.isfinite(X_batch).all()
                assert torch.isfinite(aux_X_batch).all()
            else:
                X_batch = self.data_generator.get_batches(self.n_data, batches)
                X_batch = X_batch.to(self.device, non_blocking=True)

            # PERF: check for NaN/Inf parameters once per iteration with a
            # short-circuit any() instead of iterating all params twice.
            if any(not torch.isfinite(p).all() for p in self.bnn.parameters()):
                print("Model contains NaN or Inf!")

            # set_to_none=True avoids zeroing memory and is faster than fill_(0)
            prior_optimizer.zero_grad(set_to_none=True)

            if has_aux:
                losses = torch.vmap(compute_fn)(X_batch, aux_X_batch)
            else:
                losses = torch.vmap(compute_fn)(X_batch)

            loss = losses.sum() / X_batch.size(0)
            assert torch.isfinite(loss).all()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.bnn.parameters(), max_norm=1)
            if gradient_explosion(self.bnn):
                break

            prior_optimizer.step()

            with torch.no_grad():
                # PERF: compute wdist once from already-computed losses tensor.
                wdist = float(losses.sqrt().sum() / X_batch.size(0))
                if wdist < best_wdist:
                    best_wdist = wdist
                    torch.save(
                        self.bnn.state_dict(), os.path.join(self.ckpt_dir, "best.ckpt")
                    )

                wdist_hist.append(wdist)
                wandb.log({"avg_2_W_dist": wdist}, step=it)

                if (it % print_every == 0) or it == 1:
                    self.print_info(
                        ">>> Iteration # {:3d}: Avg 2-Wasserstein Dist {:.4f}".format(
                            it, wdist
                        )
                    )

                if (it % save_ckpt_every == 0) or (it == num_iters):
                    torch.save(
                        self.bnn.state_dict(),
                        os.path.join(self.ckpt_dir, "it-{}.ckpt".format(it)),
                    )

                scheduler.step(wdist)

        return wdist_hist, last_it