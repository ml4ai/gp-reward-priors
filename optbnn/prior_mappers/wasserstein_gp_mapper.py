import itertools
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils.util import ensure_dir, prepare_device


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

        # Move models to configured device
        if gpu_gp:
            self.gp = self.gp.to(self.device)
        self.bnn = self.bnn.to(self.device)
        if len(device_ids) > 1:
            if self.gpu_gp:
                self.gp = torch.nn.DataParallel(self.gp, device_ids=device_ids)
            self.bnn = torch.nn.DataParallel(self.bnn, device_ids=device_ids)

        # Setup logger
        self.print_info = print if logger is None else logger.info

        # Setup checkpoint directory
        self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        ensure_dir(self.ckpt_dir)

    def optimize(
        self,
        num_iters,
        batches=10,
        lr=1e-2,
        save_ckpt_every=50,
        print_every=10,
    ):
        wdist_hist = []
        prior_optimizer = torch.optim.Adam(self.bnn.parameters(), lr=lr)

        # Prior loop
        # Draw X
        if self.data_generator.has_aux:

            def compute_sqw2(X, aux_X):
                X = X.to(self.device)
                aux_X = aux_X.to(self.device)
                if not self.gpu_gp:
                    X = X.to("cpu")
                    aux_X = aux_X.to("cpu")

                bnn_K = self.bnn.compute_covariance(X.double()).to(self.device)
                target_K = self.gp.compute_covariance(X.double(), aux_X.double()).to(
                    self.device
                )

                if not self.gpu_gp:
                    X = X.to(self.device)
                    aux_X = aux_X.to(self.device)

                t_evalues, t_evectors = torch.linalg.eigh(target_K)
                sqrt_t_evalues = torch.sqrt(torch.clamp(t_evalues, min=0.0))
                sqrt_target_K = (
                    t_evectors
                    @ torch.diag_embed(sqrt_t_evalues)
                    @ t_evectors.transpose(-2, -1)
                )
                evalues, evectors = torch.linalg.eigh(
                    sqrt_target_K @ bnn_K @ sqrt_target_K
                )
                sqrt_evalues = torch.sqrt(torch.clamp(evalues, min=0.0))
                fidelity = (
                    evectors
                    @ torch.diag_embed(sqrt_evalues)
                    @ evectors.transpose(-2, -1)
                )
                loss = torch.trace(target_K + bnn_K - 2 * fidelity)
                return loss

            for it in range(1, num_iters + 1):
                X_batch, aux_X_batch = self.data_generator.get_batches(self.n_data, batches)
                prior_optimizer.zero_grad()
                losses = torch.vmap(compute_sqw2)(X_batch, aux_X_batch)
                loss = losses.sum() / X_batch.size(0)
                loss.backward()
                prior_optimizer.step()
                with torch.no_grad():
                    wdist = torch.sqrt(losses).sum() / X_batch.size(0)
                    wdist_hist.append(float(wdist))
                    if (it % print_every == 0) or it == 1:
                        self.print_info(
                            ">>> Iteration # {:3d}: "
                            "Avg 2-Wasserstein Dist {:.4f}".format(it, float(wdist))
                        )

                    # Save checkpoint
                    if ((it) % save_ckpt_every == 0) or (it == num_iters):
                        path = os.path.join(self.ckpt_dir, "it-{}.ckpt".format(it))
                        torch.save(self.bnn.state_dict(), path)

        else:

            def compute_sqw2(X):
                X = X.to(self.device)
                if not self.gpu_gp:
                    X = X.to("cpu")

                bnn_K = self.bnn.compute_covariance(X.double())
                target_K = self.gp.compute_covariance(X.double(), aux_X.double())

                if not self.gpu_gp:
                    X = X.to(self.device)

                t_evalues, t_evectors = torch.linalg.eigh(target_K)
                sqrt_t_evalues = torch.sqrt(torch.clamp(t_evalues, min=0.0))
                sqrt_target_K = (
                    t_evectors
                    @ torch.diag_embed(sqrt_t_evalues)
                    @ t_evectors.transpose(-2, -1)
                )
                evalues, evectors = torch.linalg.eigh(
                    sqrt_target_K @ bnn_K @ sqrt_target_K
                )
                sqrt_evalues = torch.sqrt(torch.clamp(evalues, min=0.0))
                fidelity = (
                    evectors
                    @ torch.diag_embed(sqrt_evalues)
                    @ evectors.transpose(-2, -1)
                )
                loss = torch.trace(target_K + bnn_K - 2 * fidelity)
                return loss

            for it in range(1, num_iters + 1):
                X_batch = self.data_generator.get_batches(self.n_data, batches)
                prior_optimizer.zero_grad()
                losses = torch.vmap(compute_sqw2)(X_batch)
                loss = losses.sum() / X_batch.size(0)
                loss.backward()
                prior_optimizer.step()
                with torch.no_grad():
                    wdist = torch.sqrt(losses).sum() / X_batch.size(0)
                    wdist_hist.append(float(wdist))
                    if (it % print_every == 0) or it == 1:
                        self.print_info(
                            ">>> Iteration # {:3d}: "
                            "Avg 2-Wasserstein Dist {:.4f}".format(it, float(wdist))
                        )

                    # Save checkpoint
                    if ((it) % save_ckpt_every == 0) or (it == num_iters):
                        path = os.path.join(self.ckpt_dir, "it-{}.ckpt".format(it))
                        torch.save(self.bnn.state_dict(), path)

        return wdist_hist
