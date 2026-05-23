import torch
from torch.optim import Optimizer


class AdaptiveSGHMC(Optimizer):
    """Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
    procedure to adapt its own hyperparameters during the initial stages
    of sampling.

    References:
        [1] http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf
        [2] https://arxiv.org/pdf/1402.4102.pdf
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        num_burn_in_steps=3000,
        epsilon=1e-16,
        mdecay=0.05,
        scale_grad=1.0,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr,
            scale_grad=float(scale_grad),
            num_burn_in_steps=num_burn_in_steps,
            mdecay=mdecay,
            epsilon=epsilon,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # PERF: hoist scalar group values out of the inner parameter loop
            # so they are not re-looked-up on every iteration.
            mdecay = group["mdecay"]
            epsilon = group["epsilon"]
            lr = group["lr"]
            lr2 = lr**2
            lr4 = lr**4
            # PERF: build scale_grad as a Python float; avoids a tensor
            # construction on every parameter when it never changes shape.
            scale_grad = float(group["scale_grad"])
            num_burn_in_steps = group["num_burn_in_steps"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                state = self.state[parameter]

                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(parameter)
                    state["g"] = torch.ones_like(parameter)
                    state["v_hat"] = torch.ones_like(parameter)
                    state["momentum"] = torch.zeros_like(parameter)

                state["iteration"] += 1

                tau = state["tau"]
                g = state["g"]
                v_hat = state["v_hat"]
                momentum = state["momentum"]

                # PERF: multiply grad in-place by scalar instead of creating
                # a new tensor each step.
                gradient = parameter.grad.data.mul(scale_grad)

                if state["iteration"] <= num_burn_in_steps:
                    # Update tau first, then derive tau_inv from the new value.
                    # The reference algorithm (Springenberg et al. 2016, Alg. 1)
                    # computes α_t = 1/τ_t *after* the τ update, so that g and
                    # v_hat use the correct, current step size.  Computing it
                    # from tau_old + 1 (as was done before) underestimates the
                    # step size and slows down burn-in adaptation.
                    tau.addcdiv_(tau * g.pow(2), v_hat.add(epsilon), value=-1.0)
                    tau.add_(1.0)
                    # tau is a positive time-constant (≥ 1 by construction in
                    # the paper).  Floating-point rounding can drive it below 1
                    # when gradients are small (v_hat → 0 makes the subtracted
                    # term dominate).  Clamp to prevent negative tau_inv, which
                    # would reverse the g / v_hat update directions and send
                    # v_hat negative → sqrt(v_hat) = NaN.
                    tau.clamp_(min=1.0)
                    tau_inv = tau.reciprocal()

                    # g += tau_inv * (gradient - g)
                    g.add_(tau_inv * (gradient - g))

                    # v_hat += tau_inv * (gradient^2 - v_hat)
                    v_hat.add_(tau_inv * (gradient.pow(2) - v_hat))
                    # v_hat is a variance estimate and must remain positive.
                    # Clamp to epsilon so that sqrt(v_hat) and 1/sqrt(v_hat)
                    # never produce NaN or Inf.
                    v_hat.clamp_(min=epsilon)

                # Preconditioner: minv_t = 1 / (sqrt(v_hat) + eps)
                minv_t = v_hat.sqrt().add_(epsilon).reciprocal_()

                # epsilon_var = 2 * lr^2 * mdecay * minv_t - lr^4
                epsilon_var = minv_t.mul(2.0 * lr2 * mdecay).sub_(lr4)

                # PERF: clamp in-place, then sqrt in-place to avoid an extra alloc.
                sigma = epsilon_var.clamp_(min=1e-16).sqrt_()

                # Sample noise: N(0, sigma^2) = sigma * N(0, 1)
                sample_t = torch.empty_like(gradient).normal_().mul_(sigma)

                # momentum += -lr^2 * minv_t * gradient - mdecay * momentum + noise
                momentum.add_(
                    minv_t.mul(gradient)
                    .mul_(-lr2)
                    .add_(momentum.mul(-mdecay))
                    .add_(sample_t)
                )

                # parameter += momentum
                parameter.data.add_(momentum)

        return loss