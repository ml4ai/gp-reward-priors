import itertools
import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, TensorDataset

from ..utils.util import ensure_dir, prepare_device


class LipschitzFunction(nn.Module):
    def __init__(self, dim):
        super(LipschitzFunction, self).__init__()
        # PERF: fused Sequential instead of separate named attributes —
        # avoids repeated Python-level attribute lookups in forward().
        self.net = nn.Sequential(
            nn.Linear(dim, 200),
            nn.Softplus(),
            nn.Linear(200, 200),
            nn.Softplus(),
            nn.Linear(200, 1),
        )

    def forward(self, x):
        # PERF: cast once here rather than relying on callers remembering to cast.
        return self.net(x.float())


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)


class WassersteinDistance:
    def __init__(
        self,
        bnn,
        gp,
        lipschitz_f_dim,
        output_dim,
        use_lipschitz_constraint=True,
        lipschitz_constraint_type="gp",
        wasserstein_lr=0.01,
        device="cpu",
        gpu_gp=True,
    ):
        self.bnn = bnn
        self.gp = gp
        self.device = device
        self.output_dim = output_dim
        self.lipschitz_f_dim = lipschitz_f_dim
        self.lipschitz_constraint_type = lipschitz_constraint_type
        assert self.lipschitz_constraint_type in ["gp", "lp"]

        self.lipschitz_f = LipschitzFunction(dim=lipschitz_f_dim).to(self.device)
        self.gpu_gp = gpu_gp
        self.values_log = []

        self.optimiser = torch.optim.Adagrad(
            self.lipschitz_f.parameters(), lr=wasserstein_lr
        )
        self.use_lipschitz_constraint = use_lipschitz_constraint
        self.penalty_coeff = 10

        # PERF: pre-build a reusable ones tensor for grad_outputs so it is
        # not reallocated on every gradient-penalty call.
        self._ones_cache: dict = {}

    def _ones_like_cached(self, y: torch.Tensor) -> torch.Tensor:
        key = y.shape
        if key not in self._ones_cache:
            self._ones_cache[key] = torch.ones(key, device=self.device)
        return self._ones_cache[key]

    def calculate(self, nnet_samples, gp_samples):
        # PERF: vectorise across output dims with a single network call instead
        # of a Python loop.  Reshape so the batch dimension is (N * output_dim).
        # nnet_samples / gp_samples: [n_dim, N, output_dim]
        if self.output_dim == 1:
            f_samples = self.lipschitz_f(nnet_samples[:, :, 0].T)  # [N, 1]
            f_gp = self.lipschitz_f(gp_samples[:, :, 0].T)
            return torch.mean(f_samples - f_gp)

        # Stack all output-dim slices into one batch: [N * output_dim, n_dim]
        n_dim, N, _ = nnet_samples.shape
        ns_cat = nnet_samples.permute(1, 2, 0).reshape(N * self.output_dim, n_dim)
        gp_cat = gp_samples.permute(1, 2, 0).reshape(N * self.output_dim, n_dim)
        f_samples = self.lipschitz_f(ns_cat).view(N, self.output_dim)
        f_gp = self.lipschitz_f(gp_cat).view(N, self.output_dim)
        # mean over N then sum over output_dim  (same semantics as original loop)
        return (f_samples.mean(0) - f_gp.mean(0)).sum()

    def compute_gradient_penalty(self, samples_p, samples_q):
        # PERF: avoid repeated .to(device) inside penalty; callers already
        # ensure tensors are on device.
        eps = torch.rand(samples_p.shape[1], 1, device=samples_p.device)
        X = eps * samples_p.t().detach() + (1 - eps) * samples_q.t().detach()
        X.requires_grad_(True)
        Y = self.lipschitz_f(X)
        gradients = torch.autograd.grad(
            Y,
            X,
            grad_outputs=self._ones_like_cached(Y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        f_gradient_norm = gradients.norm(2, dim=1)

        if self.lipschitz_constraint_type == "gp":
            return ((f_gradient_norm - 1) ** 2).mean()
        else:  # "lp"
            return (torch.clamp(f_gradient_norm - 1, 0.0) ** 2).mean()

    # PERF: helper to move X (and optional aux_X) to the correct device
    # without repeating the if/else branching everywhere.
    def _to_gp_device(self, *tensors):
        dev = "cpu" if not self.gpu_gp else self.device
        return tuple(t.to(dev) for t in tensors)

    def _to_main_device(self, *tensors):
        return tuple(t.to(self.device) for t in tensors)

    def _sample_gp(self, X, n_samples, aux_X=None):
        """Draw function samples from the GP; handles device placement once."""
        if aux_X is not None:
            X_gp, aux_X_gp = self._to_gp_device(X, aux_X)
            samples = (
                self.gp.sample_functions(X_gp.double(), n_samples, aux_X_gp.double())
                .detach()
                .float()
            )
        else:
            (X_gp,) = self._to_gp_device(X)
            samples = self.gp.sample_functions(X_gp.double(), n_samples).detach().float()
        return samples.to(self.device)

    def wasserstein_optimisation(
        self, X, n_samples, aux_X=None, n_steps=10, threshold=None, debug=False
    ):
        for p in self.lipschitz_f.parameters():
            p.requires_grad_(True)

        n_samples_bag = n_samples

        # --- sample once, batch internally ---
        gp_samples_bag = self._sample_gp(X, n_samples_bag, aux_X)
        if self.output_dim > 1:
            gp_samples_bag = gp_samples_bag.squeeze()

        (X_main,) = self._to_main_device(X)
        nnet_samples_bag = (
            self.bnn.sample_functions(X_main, n_samples_bag)
            .detach()
            .float()
            .to(self.device)
        )
        if self.output_dim > 1:
            nnet_samples_bag = nnet_samples_bag.squeeze()

        # [n_dim, N, n_out] -> [N, n_dim, n_out]
        gp_samples_bag = gp_samples_bag.transpose(0, 1)
        nnet_samples_bag = nnet_samples_bag.transpose(0, 1)
        dataset = TensorDataset(gp_samples_bag, nnet_samples_bag)
        # PERF: pin_memory speeds up CPU→GPU transfers when device is CUDA.
        pin = self.device != "cpu"
        data_loader = DataLoader(
            dataset, batch_size=n_samples, num_workers=0, pin_memory=pin
        )
        batch_generator = itertools.cycle(data_loader)

        for i in range(n_steps):
            gp_samples, nnet_samples = next(batch_generator)
            gp_samples = gp_samples.transpose(0, 1).to(self.device, non_blocking=pin)
            nnet_samples = nnet_samples.transpose(0, 1).to(self.device, non_blocking=pin)

            self.optimiser.zero_grad()
            objective = -self.calculate(nnet_samples, gp_samples)
            if debug:
                self.values_log.append(-objective.item())

            if self.use_lipschitz_constraint:
                # PERF: compute penalty across all dims in fewer forward passes
                # (vectorised in calculate already; penalty still per-dim).
                penalty = sum(
                    self.compute_gradient_penalty(
                        nnet_samples[:, :, dim], gp_samples[:, :, dim]
                    )
                    for dim in range(self.output_dim)
                )
                objective = objective + self.penalty_coeff * penalty

            objective.backward()

            if threshold is not None:
                params = list(self.lipschitz_f.parameters())
                # PERF: avoid building a large intermediate tensor for the norm.
                grad_norm = torch.sqrt(
                    sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
                )

            self.optimiser.step()
            if not self.use_lipschitz_constraint:
                with torch.no_grad():
                    for p in self.lipschitz_f.parameters():
                        p.clamp_(-0.1, 0.1)

            if threshold is not None and grad_norm < threshold:
                print(
                    "WARNING: Grad norm (%.3f) lower than threshold (%.3f). "
                    "Stopping optimization at step %d" % (grad_norm, threshold, i)
                )
                if debug:
                    self.values_log += [self.values_log[-1]] * (n_steps - i - 1)
                break

        for p in self.lipschitz_f.parameters():
            p.requires_grad_(False)


class MapperWasserstein(object):
    def __init__(
        self,
        gp,
        bnn,
        data_generator,
        out_dir,
        input_dim=1,
        output_dim=1,
        n_data=256,
        wasserstein_steps=(200, 200),
        wasserstein_lr=0.01,
        wasserstein_thres=0.01,
        logger=None,
        n_gpu=0,
        gpu_gp=False,
        lipschitz_constraint_type="gp",
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

        assert lipschitz_constraint_type in ["gp", "lp"]
        self.lipschitz_constraint_type = lipschitz_constraint_type

        if type(wasserstein_steps) not in (list, tuple):
            wasserstein_steps = (wasserstein_steps, wasserstein_steps)
        self.wasserstein_steps = wasserstein_steps
        self.wasserstein_threshold = wasserstein_thres

        if gpu_gp:
            self.gp = self.gp.to(self.device)
        self.bnn = self.bnn.to(self.device)
        if len(device_ids) > 1:
            if self.gpu_gp:
                self.gp = torch.nn.DataParallel(self.gp, device_ids=device_ids)
            self.bnn = torch.nn.DataParallel(self.bnn, device_ids=device_ids)

        self.wasserstein = WassersteinDistance(
            self.bnn,
            self.gp,
            self.n_data,
            output_dim=self.output_dim,
            wasserstein_lr=wasserstein_lr,
            device=self.device,
            gpu_gp=self.gpu_gp,
            lipschitz_constraint_type=self.lipschitz_constraint_type,
        )

        self.print_info = print if logger is None else logger.info
        self.ckpt_dir = os.path.join(self.out_dir, "ckpts")
        ensure_dir(self.ckpt_dir)

    # PERF: factor out the repeated has_aux branching into a helper so the
    # main loop body is written once.
    def _draw_samples(self, n_samples, aux_X=None):
        """Return (X, gp_samples, nnet_samples), all on self.device."""
        if self.data_generator.has_aux:
            X, aux_X = self.data_generator.get(self.n_data)
        else:
            X = self.data_generator.get(self.n_data)
            aux_X = None

        X = X.to(self.device)
        if aux_X is not None:
            aux_X = aux_X.to(self.device)

        gp_samples = self.wasserstein._sample_gp(X, n_samples, aux_X)
        if self.output_dim > 1:
            gp_samples = gp_samples.squeeze()

        nnet_samples = self.bnn.sample_functions(X, n_samples).float().to(self.device)
        if self.output_dim > 1:
            nnet_samples = nnet_samples.squeeze()

        return X, aux_X, gp_samples, nnet_samples

    def optimize(
        self,
        num_iters,
        n_samples=128,
        lr=1e-2,
        save_ckpt_every=50,
        print_every=10,
        debug=False,
    ):
        wdist_hist = []
        wasserstein_steps = self.wasserstein_steps
        prior_optimizer = torch.optim.RMSprop(self.bnn.parameters(), lr=lr)

        for it in range(1, num_iters + 1):
            X, aux_X, gp_samples, nnet_samples = self._draw_samples(n_samples)

            self.wasserstein.lipschitz_f.apply(weights_init)
            self.wasserstein.wasserstein_optimisation(
                X,
                n_samples,
                aux_X=aux_X,
                n_steps=wasserstein_steps[1],
                threshold=self.wasserstein_threshold,
                debug=debug,
            )

            prior_optimizer.zero_grad()
            wdist = self.wasserstein.calculate(nnet_samples, gp_samples)
            wdist.backward()
            prior_optimizer.step()

            wdist_val = float(wdist)
            wdist_hist.append(wdist_val)
            wandb.log({"W_dist": wdist_val}, step=it)

            if (it % print_every == 0) or it == 1:
                self.print_info(
                    ">>> Iteration # {:3d}: Wasserstein Dist {:.4f}".format(
                        it, wdist_val
                    )
                )

            if (it % save_ckpt_every == 0) or (it == num_iters):
                path = os.path.join(self.ckpt_dir, "it-{}.ckpt".format(it))
                torch.save(self.bnn.state_dict(), path)

        if debug:
            values = np.array(self.wasserstein.values_log).reshape(-1, 1)
            path = os.path.join(self.out_dir, "wsr_intermediate_values.log")
            np.savetxt(path, values, fmt="%.6e")
            self.print_info("Saved intermediate wasserstein values in: " + path)

        return wdist_hist