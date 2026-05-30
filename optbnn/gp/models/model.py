# Copyright 2016 James Hensman, alexggmatthews, PabloLeon, Valentine Svensson
# Copyright 2017 Thomas Viehmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import torch
import numpy as np

from .. import mean_functions
from .. import parameter


class GPModel(torch.nn.Module):
    """
    A base class for Gaussian process models, that is, those of the form
       \begin{align}
       \theta & \sim p(\theta) \\
       f       & \sim \mathcal{GP}(m(x), k(x, x'; \theta)) \\
       f_i       & = f(x_i) \\
       y_i\,|\,f_i     & \sim p(y_i|f_i)
       \end{align}

    This class mostly adds functionality to compile predictions. To use it,
    inheriting classes must define a build_predict function, which computes
    the means and variances of the latent function. This gets compiled
    similarly to build_likelihood in the Model class.
    These predictions are then pushed through the likelihood to obtain means
    and variances of held out data, self.predict_y.
    The predictions can also be used to compute the (log) density of held-out
    data via self.predict_density.
    For handling another data (Xnew, Ynew), set the new value to
    self.X and self.Y

    >>> m.X = Xnew
    >>> m.Y = Ynew
    """

    def __init__(
        self, X, Y, kern, likelihood, mean_function, name=None, jitter_level=1e-6
    ):
        super(GPModel, self).__init__()
        self.name = name
        self.mean_function = mean_function or mean_functions.Zero()
        self.kern = kern
        self.likelihood = likelihood
        self.jitter_level = jitter_level

        if isinstance(X, np.ndarray):
            # X is a data matrix; each row represents one instance
            X = torch.from_numpy(X)
        if isinstance(Y, np.ndarray):
            # Y is a data matrix, rows correspond to the rows in X,
            # columns are treated independently
            Y = torch.from_numpy(Y)
        self.X, self.Y = X, Y

    @abc.abstractmethod
    def compute_log_prior(self):
        """Compute the log prior of the model."""
        pass

    @abc.abstractmethod
    def compute_log_likelihood(self, X=None, Y=None):
        """Compute the log likelihood of the model."""
        pass

    def objective(self, X=None, Y=None):
        pos_objective = self.compute_log_likelihood(X, Y)
        for param in self.parameters():
            if isinstance(param, parameter.ParamWithPrior):
                pos_objective = pos_objective + param.get_prior()
        return -pos_objective

    def forward(self, X=None, Y=None):
        return self.objective(X, Y)

    @abc.abstractmethod
    def predict_f(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        raise NotImplementedError

    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self.predict_f(Xnew, full_cov=True)

    def sample_functions(self, X, num_samples):
        """
        Produce samples from the prior latent functions at the points X.
        """
        X = X.reshape((-1, self.kern.input_dim))
        mu = self.mean_function(X)

        prior_params = []
        for name in dict(self.kern.named_parameters()).keys():
            param = getattr(self.kern, name)
            if param.prior is not None:
                prior_params.append(param)
        if len(prior_params) == 0:
            var = self.kern.K(X)
            jitter = (
                torch.eye(mu.size(0), dtype=mu.dtype, device=mu.device)
                * self.jitter_level
            )
            samples = []
            for i in range(self.num_latent):
                L = torch.cholesky(var + jitter, upper=False)
                V = torch.randn(L.size(0), num_samples, dtype=L.dtype, device=L.device)
                samples.append(mu + torch.matmul(L, V))
            return torch.stack(samples, dim=0).permute(1, 2, 0)
        else:
            samples = []
            for sample_idx in range(num_samples):
                for param in prior_params:
                    with torch.no_grad():
                        param.copy_(param.prior.sample())
                var = self.kern.K(X)
                jitter = (
                    torch.eye(mu.size(0), dtype=mu.dtype, device=mu.device)
                    * self.jitter_level
                )
                s = []
                for i in range(self.num_latent):
                    multiplier = 1
                    while True:
                        try:
                            L = torch.cholesky(var + multiplier * jitter, upper=False)
                            break
                        except RuntimeError as err:
                            multiplier *= 2
                    V = torch.randn(L.size(0), 1, dtype=L.dtype, device=L.device)
                    s.append(mu + torch.matmul(L, V))
                samples.append(torch.stack(s, dim=0).permute(1, 2, 0))
            return torch.cat(samples, dim=1)

    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self.predict_f(Xnew, full_cov=True)
        jitter = (
            torch.eye(mu.size(0), dtype=mu.dtype, device=mu.device) * self.jitter_level
        )
        samples = []
        for i in range(self.num_latent):  # TV-Todo: batch??
            L = torch.cholesky(var[:, :, i] + jitter, upper=False)
            V = torch.randn(L.size(0), num_samples, dtype=L.dtype, device=L.device)
            samples.append(mu[:, i : i + 1] + torch.matmul(L, V))
        return torch.stack(samples, dim=0)  # TV-Todo: transpose?

    def predict_y(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self.predict_f(Xnew, full_cov)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew
        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self.predict_f(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)

    def _repr_html_(self):
        s = "Model {}<ul>".format(type(self).__name__)
        for n, c in self.named_children():
            s += "<li>{}: {}</li>".format(n, type(c).__name__)
        s += "</ul><table><tr><th>Parameter</th><th>Value</th><th>Prior</th><th>ParamType</th></tr><tr><td>"
        s += "</td></tr><tr><td>".join(
            [
                "</td><td>".join(
                    (n, str(p.get().data.cpu().numpy()), str(p.prior), type(p).__name__)
                )
                for n, p in self.named_parameters()
            ]
        )
        s += "</td></tr></table>"
        return s


class LCFModel(torch.nn.Module):
    """
    A Gaussian Process defined as a linear combination of functions (LCF).

    The GP is parameterised by a weight distribution w ~ N(p_mean, p_covariance)
    and a feature map Φ: X → R^d (provided by `function_vect`).  The resulting
    random function is f(x) = Φ(x)^T w, which induces the GP

        m(x)         = Φ(x)^T p_mean
        K(x, x')     = Φ(x)^T p_covariance Φ(x')

    Args:
        p_covariance: (d,) or (d, d) array/tensor — weight prior covariance.
            A 1-D input is interpreted as the diagonal of a diagonal covariance.
        function_vect: callable (X, device) or (X, aux_X, device) → (n, d) tensor.
            Evaluates the feature map over a batch of inputs.  The function is
            responsible for including any intercept column.
        device: torch device for all internal tensors.
        p_mean: optional (d,) array/tensor — weight prior mean.  Defaults to zeros.
        name: optional string label (informational only).
    """

    def __init__(self, p_covariance, function_vect, device, p_mean=None, name=None):
        super(LCFModel, self).__init__()
        self.name = name
        self.function_vect = function_vect
        self.device = device

        if isinstance(p_covariance, np.ndarray):
            if p_covariance.ndim == 1:
                p_covariance = np.diag(p_covariance)
            p_covariance = torch.from_numpy(p_covariance).float().to(device)
        else:
            if p_covariance.ndim == 1:
                p_covariance = torch.diag(p_covariance).float().to(device)

        if p_mean is not None:
            if isinstance(p_mean, np.ndarray):
                p_mean = torch.from_numpy(p_mean).float().to(device)
        else:
            p_mean = torch.zeros(p_covariance.size(0)).float().to(device)

        self.weight_generator = (
            torch.distributions.multivariate_normal.MultivariateNormal(
                p_mean, p_covariance
            )
        )
        self.p_mean = p_mean
        self.p_covariance = p_covariance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _feature_matrix(self, X, aux_X=None):
        """
        Evaluate the feature map Φ(X) via function_vect.

        Returns a (n, d) double tensor where n = X.shape[0] and d = n_concepts.
        Avoids the repeated if/else scattered across public methods.
        """
        if aux_X is not None:
            return self.function_vect(X, aux_X, self.device)
        return self.function_vect(X, self.device)

    def _woodbury_solve(self, Phi, residual, jitter=1e-6):
        """
        Solve α = (K + εI)^{-1} residual for the degenerate kernel
        K = Φ Σ Φ^T, and return the regularised log-determinant log|K + εI|.

        K has rank ≤ d = n_concepts, so for n_M > d it is singular and a direct
        (n_M, n_M) Cholesky is numerically fragile — its trailing pivots rest
        entirely on the nugget and go negative once the feature scale grows.
        The Woodbury identity reduces the problem to an exact, well-conditioned
        (d, d) solve:

            (εI + ΦΣΦ^T)^{-1} = (1/ε)I - (1/ε^2) Φ (Σ^{-1} + (1/ε)Φ^TΦ)^{-1} Φ^T

        The nugget ε is scaled relative to the mean magnitude of diag(K)
        (= trace(Σ Φ^TΦ) / n_M), so it stays an effective regulariser for any
        feature scale without retuning.  This is the standard "GP + nugget":
        k_eff(x, x') = Φ(x)^T Σ Φ(x') + ε·δ(x, x'), still a valid GP.

        Args:
            Phi: (n_M, d) feature matrix.
            residual: (n_M,) tensor = f - m.
            jitter: relative nugget coefficient.

        Returns:
            alpha:  (n_M,) double tensor = (K + εI)^{-1} residual.
            logdet: scalar double tensor = log|K + εI|.
        """
        Phi = Phi.double()
        residual = residual.double().view(-1)                # (n_M,)
        Sigma = self.p_covariance.double()                   # (d, d)
        n = Phi.shape[0]

        PtP = Phi.T @ Phi                                     # (d, d)

        # Relative nugget: ε = jitter · max(mean(diag K), 1),
        # where mean(diag K) = trace(Σ Φ^TΦ) / n_M.
        k_diag_mean = torch.einsum("ij,ij->", Sigma, PtP) / n
        eps = jitter * torch.clamp(k_diag_mean, min=1.0)

        # (d, d) inner system A = Σ^{-1} + (1/ε) Φ^TΦ  (PD, well-conditioned)
        Sigma_inv = torch.linalg.inv(Sigma)
        A = Sigma_inv + PtP / eps
        LA = torch.linalg.cholesky(A)

        # alpha = (1/ε)[ residual - (1/ε) Φ A^{-1} Φ^T residual ]
        b = Phi.T @ residual                                 # (d,)
        z = torch.cholesky_solve(b.unsqueeze(1), LA).squeeze(1)
        alpha = (residual - (Phi @ z) / eps) / eps           # (n_M,)

        # Matrix determinant lemma:
        #   log|εI + ΦΣΦ^T| = log|A| + log|Σ| + n_M·log ε
        logdet_A = 2.0 * torch.log(torch.diagonal(LA)).sum()
        _, logdet_Sigma = torch.linalg.slogdet(Sigma)
        logdet = logdet_A + logdet_Sigma + n * torch.log(eps)
        return alpha, logdet

    def _alpha_from_phi(self, Phi, f, jitter=1e-6):
        """
        Given the feature matrix Phi (n, d) and function values f (n,) or
        (n, 1), compute α = (K + εI)^{-1}(f - m) where K = Φ Σ Φ^T.

        Delegates to _woodbury_solve so the degenerate (rank ≤ d) kernel is
        solved in the d-dimensional weight space rather than by an (n, n)
        Cholesky.  Returns a (n,) double tensor.
        """
        Phi = Phi.double()
        m = torch.mv(Phi, self.p_mean.double())              # (n,)
        residual = f.double().view(-1) - m                   # (n,)
        alpha, _ = self._woodbury_solve(Phi, residual, jitter)
        return alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, X=None, Y=None):
        pass

    def compute_mean(self, X, aux_X=None):
        """
        Compute the GP prior mean m(X) = Φ(X) @ p_mean.

        Args:
            X: (n, d_raw) input tensor or array.
            aux_X: optional auxiliary inputs (n, d_aux).

        Returns:
            (n,) double tensor of prior mean values.
        """
        Phi = self._feature_matrix(X, aux_X)
        return torch.mv(Phi, self.p_mean.double())

    def compute_covariance(self, X, aux_X=None):
        """
        Compute the GP prior covariance K(X, X) = Φ(X) @ p_covariance @ Φ(X)^T.

        Args:
            X: (n, d_raw) input tensor or array.
            aux_X: optional auxiliary inputs (n, d_aux).

        Returns:
            (n, n) double tensor — the prior covariance matrix over X.
        """
        Phi = self._feature_matrix(X, aux_X)
        return torch.mm(torch.mm(Phi, self.p_covariance.double()), Phi.T)

    def sample_functions(self, X, num_samples, aux_X=None):
        """
        Draw prior function samples at the inputs X.

        Each sample is drawn by sampling a weight vector w ~ N(p_mean, p_covariance)
        and computing f(X) = Φ(X) w.

        Args:
            X: (n, d_raw) input tensor or array.
            num_samples: number of independent function samples to draw.
            aux_X: optional auxiliary inputs (n, d_aux).

        Returns:
            (n, num_samples, 1) double tensor of sampled function values.
        """
        Phi = self._feature_matrix(X, aux_X)                         # (n, d)
        weights = self.weight_generator.sample(                       # (d, S)
            torch.Size([num_samples])
        ).T.double()
        return torch.mm(Phi, weights).unsqueeze(-1)                   # (n, S, 1)

    def solve_prior(self, X_M, f_XM, aux_X=None, jitter=1e-6):
        """
        Compute α = K_{X_M}^{-1}(f_{X_M} - m(X_M)) via Cholesky solve.

        This is the 'alpha' vector required by the fSGHMC functional prior
        gradient.  Pass the BNN outputs at the inducing points as f_XM
        (detached from the computational graph if gradients are not needed here).

        Args:
            X_M: (n_M, d_raw) inducing-point inputs.
            f_XM: (n_M,) or (n_M, 1) BNN function values at X_M.
            aux_X: optional auxiliary inputs (n_M, d_aux).
            jitter: scalar added to the diagonal of K before factorisation.

        Returns:
            alpha: (n_M,) double tensor = K_{X_M}^{-1}(f_{X_M} - m(X_M)).
        """
        Phi = self._feature_matrix(X_M, aux_X)
        return self._alpha_from_phi(Phi, f_XM, jitter)

    def log_prior(self, X_M, f_XM, aux_X=None, jitter=1e-6):
        """
        Evaluate the functional GP prior log-density at the given BNN outputs.

            log p_GP(f_{X_M}) = -1/2 (f-m)^T K^{-1} (f-m)
                                 - 1/2 log det K
                                 - n/2 log(2π)

        Useful for monitoring convergence and acceptance decisions.

        Args:
            X_M: (n_M, d_raw) inducing-point inputs.
            f_XM: (n_M,) or (n_M, 1) BNN function values at X_M.
            aux_X: optional auxiliary inputs (n_M, d_aux).
            jitter: scalar diagonal regularisation before Cholesky.

        Returns:
            Scalar double tensor — the log-density value.
        """
        Phi = self._feature_matrix(X_M, aux_X).double()
        m = torch.mv(Phi, self.p_mean.double())
        residual = f_XM.double().view(-1) - m                         # (n,)
        alpha, logdet = self._woodbury_solve(Phi, residual, jitter)
        n = residual.size(0)
        return (
            -0.5 * (residual @ alpha)
            - 0.5 * logdet
            - 0.5 * n * torch.log(torch.tensor(2.0 * np.pi, dtype=torch.float64))
        )

    def functional_prior_grad(self, net, X_M, aux_X=None, jitter=1e-6):
        """
        Compute the fSGHMC functional GP prior gradient w.r.t. ``net``'s parameters:

            ∇_w log p_GP(f(·; w)) = -J_w(X_M)^T K_{X_M}^{-1}(f(X_M; w) - m(X_M))

        where J_w(X_M) is the Jacobian of the BNN outputs at X_M w.r.t. the
        network weights w.  The computation uses a single vector-Jacobian product
        (VJP / reverse-mode AD), so the cost is one backward pass — independent
        of the number of parameters.

        The return value is ∇_w log p_GP, i.e. it is ready to be **added** to the
        likelihood gradient.  It replaces the weight-space Gaussian prior gradient
        −(w − μ)/σ² used in standard SGHMC.

        Note: ``net`` must be in the same dtype/device as X_M.  The internal
        Cholesky solve is performed in float64 and the VJP grad_outputs are cast
        back to match ``net``'s output dtype before the backward pass.

        Args:
            net: torch.nn.Module — the BNN whose parameters we differentiate.
            X_M: (n_M, d_raw) inducing-point inputs (must be on net's device).
            aux_X: optional auxiliary inputs (n_M, d_aux).
            jitter: scalar diagonal regularisation for the Cholesky solve.

        Returns:
            grads: tuple of tensors, one per element of ``net.parameters()``,
                   giving ∇_w log p_GP(f(·; w)).  Parameters that do not
                   contribute to f(X_M) (e.g. unused branches) receive
                   a zero gradient tensor.
        """
        # --- BNN forward pass: keep graph so autograd can propagate through it ---
        f_XM = net(X_M)                                               # (n_M,) or (n_M, 1)

        # --- Compute α = K^{-1}(f - m) purely numerically (no grad needed) ---
        Phi = self._feature_matrix(X_M, aux_X)                       # (n_M, d)
        with torch.no_grad():
            alpha = self._alpha_from_phi(Phi, f_XM.detach(), jitter) # (n_M,) double

        # Cast α to the BNN's output dtype and reshape to match f_XM for VJP
        grad_out = alpha.to(f_XM.dtype)
        if f_XM.dim() > 1:
            grad_out = grad_out.view_as(f_XM)

        # --- VJP: J_w(X_M)^T α ---
        # torch.autograd.grad(f, params, v) computes sum_i v_i * ∂f_i/∂params,
        # i.e. J^T v.  The prior log-density gradient is -J^T α, so we negate.
        vjp = torch.autograd.grad(
            outputs=f_XM,
            inputs=list(net.parameters()),
            grad_outputs=grad_out,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # Negate to get ∇_w log p_GP = -J^T α; replace None (unused params) with zeros
        grads = tuple(
            (-g if g is not None else torch.zeros_like(p))
            for g, p in zip(vjp, net.parameters())
        )
        return grads
