# Functional HMC / NUTS: theoretical and practical issues

Written 2026-09-04. Sources: Wu et al. (2025) §3.2, Eq. 7–9, Table 1, App. B–C
(Prop. 3.1/3.2 proofs), Alg. 2; `optbnn/gp/models/map_informed_prior.py`;
`optbnn/sgmcmc_bayes_net/f_pref_net.py`; `scripts_bnn/run_bnn_training_antmaze_eval.py`;
the four `*_bnn_antmaze_eval.yaml` configs.

**Verdict.** Functional HMC is a much smaller derivation than it looks — Wu
et al.'s functional Hamiltonian dynamics reduce, in the parameter-space form
they actually implement, to ordinary HMC with a functional potential. That is
good news: almost no new theory is needed. But **the target as currently
specified is improper**, and that is fatal for HMC/NUTS in a way it is merely
untidy for fSGHMC. Fix that and three of the four variants are comfortably
within reach; the other two are not, for a reason unrelated to the sampler.

---

## 1. What "functional HMC" actually reduces to

Wu et al. define the dynamics in function space (Eq. 7) and then transform to
parameter space (Eq. 8):

```
dw/dt = -∇_z g · ∇_g log p(g)
dz/dt = -∇_w f · ∇_f U(f)
```

With the auxiliary lift taken as the identity, `g(·; z) = z` and `p(g) = N(0, M)`,
this is `dw = M⁻¹z dt`, `dz = -∇_w U(f(w)) dt` — **canonical Hamiltonian
dynamics in (w, z) with potential `U(w) = U(f(w))`**. Alg. 2 confirms it: it is
HMC with `m` leapfrog steps and momentum resampling, minus the MH correction.
Your own earlier code review reached the same conclusion ("the fSGHMC momentum
lift is unexercised"), and it is right.

So there is no separate "functional HMC" to derive. What you have is HMC on

```
U(w) = Σ_pairs CE(BT logit) + ½ f(X_M; w)ᵀ K_MM⁻¹ f(X_M; w)
```

and every genuinely *functional* design decision lives in three places: the
measurement set (§3), the metric (§4), and the U-turn criterion (§4). That is a
narrower but cleaner contribution than "a new sampler", and it is worth framing
that way rather than as a derivation.

### 1.1 A gap in Prop. 3.2 you should not inherit

App. C invokes Ma et al. (2015) Thm. 1, which requires

```
µ(X) = -[D(X) + Q(X)]∇G(X) + Γ(X),     Γ_i(X) = Σ_j ∂/∂X_j (D_ij(X) + Q_ij(X))
```

For **fSGLD** (Prop. 3.1, Eq. 17) they correctly carry `Γ(f) = H_w f`, the
second-order Fréchet term.

For the **Hamiltonian** case they do not. Eq. 19 writes
`d[f; g] = -Q(f, g)∇H(f, g) dt` with

```
Q(f, g) = [[0, -(∇_w f)ᵀ ∇_z g], [(∇_z g)ᵀ ∇_w f, 0]]
```

which is **state-dependent** — it contains the Jacobian `∇_w f`. With
state-dependent `Q`, `Γ ≠ 0` in general, and no `Γ` appears. Eq. 20 has the same
issue on the `D` block, where `D(f, g) = C(∇_z g)ᵀ ∇_z g` is also
state-dependent. (There is also a sign slip: the text reads "π(f, g) ∝ exp(H(f, g))"
where it should be `exp(-H)`.)

**Why this does not break their algorithm but does affect your write-up.** Push
the dynamics down to `(w, z)` and `Q` becomes the constant canonical
`[[0, -I], [I, 0]]`, so `Γ = 0` and the parameter-space sampler is fine. The gap
is in the *function-space* statement, which is the paper's headline claim. If
you write a functional-HMC chapter that cites Prop. 3.2, either supply the `Γ`
term for state-dependent `Q` or state the result in `(w, z)` where the curl
matrix is constant. A reviewer who knows Ma et al. will look for exactly this.

---

## 2. The blocking issue: `exp(-U(w))` is improper

These are two separate claims and they are worth keeping apart: **(2a)** the
posterior has infinite mass, which is a yes/no fact proved by one explicit set;
and **(2b)** how large the flat set is, which is a rank count and only says how
bad it is.

### 2a. `∫ exp(-U(w)) dw = ∞`

`FPrefNet` uses **no weight-space prior at all** — that is its stated design
(module docstring: "requires no weight-space prior... the only regularisation
comes from the functional GP prior"). So

```
U(w) = Σ_pairs CE(BT logit)  +  ½ f(X_M; w)ᵀ K⁻¹ f(X_M; w)
```

and **both terms see `w` only through `f`**. Nothing in `U` grows when `w` grows
in a direction that leaves `f` alone. In a standard BNN the `‖w‖²/(2σ_w²)` term
is what makes the integral converge; there is no such term here.

**Explicit witness.** Take one unit `j` in the **first** hidden layer, with
incoming weights `w_j` and bias `b_j` (`optbnn/bnn/layers/linear.py` gives every
`Linear` a bias). Its pre-activation on input `x` is `x·w_j/√37 + b_j`, and `x`
ranges over fixed, bounded data. Choose any `w_j`, then set

```
b_j  <  − max_i (x_i · w_j)/√37       over all training and measurement rows
```

The unit is now dead on every input the likelihood or the prior ever evaluates,
so `f` — and hence `U` — is exactly what it would be with the unit deleted. That
set of `(w_j, b_j)` is an open cone in R^38: unbounded, infinite Lebesgue
measure, and `U` does not depend on those 38 coordinates anywhere on it. So for
any small ball `B` in the remaining coordinates,

```
∫ exp(-U) dw  ≥  exp(-sup_B U) · vol(B) · vol(cone)  =  ∞
```

Because it is a **first**-layer unit, the cone is determined by the data alone
and does not depend on the other weights, so the argument is airtight. The exact
ReLU rescaling symmetry (`W1, b1 → cW1, cb1` with `W2 → W2/c`, valid for any
`c > 0`) gives a second unbounded family.

> **`tanh` does not fix this.** With a smooth saturating activation there is no
> dead cone, but the unit saturates: `U` tends to a *finite* limit as `‖w_j‖ → ∞`
> along most rays, so `exp(-U)` does not decay at infinity and the integral still
> diverges. Switching activation fixes the leapfrog smoothness problem (§5);
> propriety needs a weight-space prior or a constrained reparameterisation (§6).
> These are two independent fixes and both are needed.

### 2b. How large the flat set is

`U` depends on `w` through exactly two sets of scalars:

- **Likelihood.** `CrossEntropyLoss` on the two pooled logits is shift-invariant,
  so each pair contributes **one** constrained scalar — the logit difference.
  That is `N` scalars.
- **Prior.** `½ f_Mᵀ K⁻¹ f_M` constrains `f` at each of the `n_meas` measurement
  points: `n_meas` scalars.

so the count is **`N + n_meas`**.

> **Correction.** An earlier draft used the free-cell count (26/33) as the number
> of constrained functionals. That is wrong. Cells bound the *rank of the
> geometric information* in `K` (`Kgeo[idx][:,idx]` has rank ≤ `n_cells`,
> `sig_c2·J` adds one), but `sig_n2·I` makes `K` full rank, so all `n_meas`
> point-values are genuinely constrained — the `n_meas − n_cells − 1` extra
> directions are simply constrained *only* by the nugget, at strength
> `1/(amp2·sig_n2)`. That is the within-cell smoothness penalty described in §3,
> and it is why `cond(K)` grows linearly in `n_meas`. Two different quantities.

Parameter counts (`hidden_dims = [2**width] * depth`, `input_dim = 37` =
29 obs + 8 action), with `n_meas = 256` per the §4.3.46 settled recipe:

| variant | width × depth | P | N | constrained (`N`+256) | flat dims | over-param |
|---|---|---|---|---|---|---|
| medium_play | 64 × 2 | 6,657 | ≈725 | 981 | 5,676 | 6.8× |
| large_diverse | 64 × 4 | 14,977 | ≈1,042 | 1,298 | 13,679 | 11.5× |
| medium_diverse | 1024 × 2 | 1,089,537 | ≈1,009 | 1,265 | 1,088,272 | 861× |
| large_play | 512 × 6 | 1,333,249 | ≈515 | 771 | 1,332,478 | 1,729× |

"Flat dims" is `P − (N + n_meas)`: the generic dimension of the level set
`{w : f is unchanged at every constrained point}`, i.e. the null space of the
Jacobian of those functionals. **This is a local, generic rank statement** — it
says the level set through a typical `w` has that dimension, not that it is
unbounded. Unboundedness is what §2a establishes, separately and without any
counting.

**You already observe this.** §4.5 records `‖w‖² = ‖w₀‖² + c·t` and correctly
identifies it as "free diffusion along f-preserving flat directions". For
fSGHMC that is tolerable: the chain wanders in the null space forever, `f` is
unaffected, and you only ever look at `f`. **For HMC/NUTS it is fatal, in four
separate ways:**

1. **No target.** The MH correction is against a non-normalisable density.
2. **The U-turn criterion never fires.** Along a flat direction there is no
   force, so the trajectory is ballistic: `(θ⁺ − θ⁻)` grows linearly while the
   velocity is constant, and their inner product stays positive indefinitely.
   NUTS will hit `max_treedepth` on essentially every iteration.
3. **Mass-matrix adaptation diverges.** Stan-style windowed adaptation estimates
   the metric from the warm-up sample variance, which grows without bound along
   flat directions.
4. **Step-size adaptation destabilises.** Energy error along flat directions is
   zero, so dual averaging at target-accept 0.8 keeps raising ε until the stiff
   directions diverge; combined with (3) this oscillates.

Every one of these presents as "NUTS doesn't work on BNNs", which is the wrong
lesson. **This must be fixed before any other question about NUTS is
meaningful.** Four fixes, in §6.

---

## 3. The measurement set: what HMC forbids, and what your prior gives you free

**Forbidden: resampling `X_M` inside a trajectory.** Wu et al. Alg. 2 draws
`X_M` inside the leapfrog loop, and `f_pref_net.train()` resamples every step.
That is fine for SGHMC (no MH correction, and the resampled prior gradient is
still a conservative field — the implied target is `exp(-E_M[U_M])`, the
`M`-averaged energy, which is well defined). It is **not** fine for HMC: the
Hamiltonian is not conserved along a trajectory whose potential changes at every
step, the acceptance ratio is meaningless, and reversibility is gone.

So HMC needs a fixed `M`. That normally costs you prior coverage — §4.3.24/4.3.25
measured exactly that cost and killed the fixed-set route. **Here it costs
nothing**, because of a structural fact about your prior that is worth stating
explicitly:

> `map_informed_prior` builds `K` from **cell indices only**. Two measurement
> points in the same free cell produce identical rows and columns of `K`. So the
> prior carries at most `n_cells` = 26 (medium) / 33 (large) distinct pieces of
> information, and `n_meas > n_cells` adds no new prior structure — only
> near-duplicate constraints, which is precisely why `cond(K) ≥ n·sig_c2/sig_n2`
> grows linearly in `n_meas` (§4.3.42).

**Therefore: use one representative point per free cell, fixed, all of them.**
`n_M` = 26 or 33, no subsetting, no resampling, no coverage loss, and the target
is unambiguous — `exp(-E_M[U_M])` and `exp(-U_M)` coincide when `M` is the whole
pool. This also makes the Cholesky **constant**, so `L = chol(K)` is precomputed
once and each leapfrog step costs one triangular solve on a 26×26 matrix
(`map_informed_prior._solve` currently re-factorises every step, in float64).

One thing this loses: `n_meas > n_cells` currently enforces *within-cell*
smoothness of `f` at rate `1/(amp2·sig_n2)`, since duplicate cells with the
nugget act like repeated noisy observations. If you want that, keep `k` points
per cell with `k` fixed and identical across cells — still a fixed set, still one
precomputed Cholesky.

You will also need the **potential's value**, not just its gradient.
`functional_prior_grad` returns only `∇_w log p_GP`; the value is
`0.5 · f_Mᵀ α` with `α = K⁻¹f_M`, which the function already computes. Two lines.

---

## 4. Geometry, and the metric that makes NUTS's criterion functional

### 4.1 The nugget sets the step size, directly

The prior contributes `JᵀK⁻¹J` to the Hessian of `U`. Its largest eigenvalue is
`1/λ_min(K) = 1/(amp2·sig_n2)`, and HMC's stable step size goes as
`1/√λ_max(∇²U)`. So **`sig_n2` is not a numerical convenience here — it is the
parameter that sets `ε`.** Your §4.3.42–44 found the same thing from the SGHMC
side (`cond(K) ≥ n·sig_c2/sig_n2`, the nugget fixed large_play where `sig_c2`
did nothing); in HMC terms it is exact and quantitative.

NUTS cost per effective sample scales roughly as `√cond(∇²U)`. Two easy wins:

- **One point per cell** takes `n` from 256 to 26/33, and with
  `cond ≈ n·sig_c2/sig_n2` that is `2.6e4` instead of `2.3e5` — a ~3× reduction
  in trajectory length (tree depth ~7.5 instead of ~9).
- **The offset.** The BT likelihood is exactly invariant to `f → f + c`, and the
  rank-1 `sig_c2·J` term carries `λ_max(K)`, so `K⁻¹` penalises the offset
  *least* (§4.3.43). That is one near-flat direction with an enormous prior
  variance sitting inside the condition number. Since §5.1 establishes that IQL
  is **not** invariant to the offset, you cannot drop it — but you should
  **reparameterise it out**: sample the centred shape on the cells plus an
  explicit scalar offset with its own prior. Standard non-centred practice, and
  it removes the worst direction from the metric.

### 4.2 The GGN metric, and why it gives you a function-space U-turn for free

§4.3.41 named "better preconditioning" as a remaining lever and §4.3.68–4.3.72
closed the diagonal `V̂^{-1/2}` route. HMC offers the metric that the diagonal
preconditioner cannot express:

```
M = Jᵀ K⁻¹ J  +  σ_w⁻² I
```

the pullback of the function-space metric (Gauss–Newton of the prior term) plus
the weight prior from §6. It is **low-rank + diagonal** — rank ≤ `n_M` = 26 — so
Woodbury gives `M⁻¹` exactly at negligible cost, and it is estimated once at the
MAP and held fixed (Riemannian HMC with a state-dependent metric needs the
implicit generalised leapfrog and is not worth it here).

The part worth writing up: **under this metric, NUTS's existing U-turn criterion
is already a function-space criterion.** The generalised criterion tests
`(θ⁺ − θ⁻)ᵀ M⁻¹ p`, i.e. `(θ⁺ − θ⁻)ᵀ v` with `v` the velocity. Linearising `f`
(which is the same approximation the GGN metric already makes),
`f⁺ − f⁻ ≈ J(θ⁺ − θ⁻)` and `ḟ = Jv`, so

```
(θ⁺ − θ⁻)ᵀ (Jᵀ K⁻¹ J) v  ≈  (f⁺ − f⁻)ᵀ K⁻¹ ḟ
```

— a U-turn in `f`, measured in the prior's own norm, plus a `σ_w⁻²` weight-space
term. So you do **not** need a novel termination rule, and you avoid the
correctness hazard that comes with one (NUTS's proof needs the criterion to be a
reversal-symmetric function of the trajectory; ad-hoc criteria break it). You
need the right metric, and the criterion becomes functional on its own. That is
a clean, defensible contribution and it is the strongest reason to frame this as
"functional HMC" rather than "HMC on a BNN".

Worth verifying carefully before claiming it: the identity is exact only under
the linearisation, and the sub-tree criteria in multinomial NUTS need the same
treatment.

---

## 5. ReLU is the wrong activation for HMC

This is an **independent** problem from §2 and fixing it does not fix that one:
a tanh network's posterior is still improper (§2a). Both fixes are needed.

`run_bnn_training_antmaze_eval.py:454` sets `transfer_fn = "relu"`. `U(w)` is
then only **piecewise** smooth: `∇U` is discontinuous across every activation
boundary. Leapfrog loses its second-order energy conservation on non-smooth
Hamiltonians, so you get spurious divergences, an unstable acceptance rate, and
step-size adaptation that collapses toward zero.

Wu et al.'s own experiments use **tanh** networks ("2 × 100 fully connected tanh
neural networks", App. E). Switch to `tanh` (already in `MLP`'s options dict) or
add GELU/SiLU before attempting any of this. This is a one-line change and it
should be made *first*, because a ReLU network will make every other problem
look like a NUTS problem.

Note this also changes the fSGHMC results — smooth activations generally mix
differently — so if you make the switch, it is a variant to re-measure, not a
free change.

---

## 6. Four ways to make the target proper, ranked

**A. Add a weak Gaussian weight prior.** `-log p_0(w) = ‖w‖²/(2σ_w²)`. Minimal
change: one term in `U`, one term in the gradient. It compactifies every level
set, kills the ReLU rescaling degeneracy, and gives the metric in §4.2 its
diagonal. **Cost, and it must be disclosed:** the `f`-marginal is now the
pushforward of a hybrid prior, not the pure functional prior, which is exactly
the property `FPrefNet`'s design advertises. Defensible as `σ_w → ∞` recovers
the functional-only posterior and a large finite `σ_w` is non-informative on `f`
— but that is an argument to make explicitly, ideally with a sensitivity check
across `σ_w`.

**B. Sample `f` on the cells directly — and use it as ground truth.** Because
the prior sees only cells, you can drop the network and put a
`MultivariateNormal(0, K)` prior on a 26- or 33-dimensional vector `u` of
per-cell rewards, with the BT likelihood on cell-averaged returns. That is a
**proper, low-dimensional, near-log-concave posterior** that NUTS will sample
essentially exactly in seconds, with ESS/iteration near 1.

It is not the dissertation's model — the deployed reward must generalise over
the full 37-dim observation — but it is something the project has never had and
badly needs: **a reference posterior to check fSGHMC against.** Every diagnostic
in §4.2–4.5 is an internal-consistency check; none of them can answer "is the
75-draw ensemble actually the right posterior?" This can. Given the eval set
occupies only 13/26 and 18/33 cells anyway (§4.3.69), the cell model is close to
a sufficient statistic for what those diagnostics measure. **If you do one thing
from this note, do this one** — it is a day of work and it converts the whole
drift investigation from internal consistency to comparison against truth.

**C. Last-layer / linearised BNN.** Fix the hidden layers at the MAP and sample
only the output layer (64 or 512 parameters) under a Gaussian prior. Proper,
low-dimensional, NUTS-friendly, and a well-established approximation with its own
literature. A reasonable middle ground and a legitimate ablation.

**D. Shrink the network.** medium_play is already ~9:1 over-parameterised
against ~758 constrained directions. Nothing in the method requires
`width × depth` = 64 × 2, and the two 1M-parameter configs
(medium_diverse, large_play) were chosen by a CE sweep, not by anything
principled. Even with (A), NUTS at 1.09M and 1.33M dimensions is a research
project in itself; at 6.7k and 15k it is routine.

---

## 7. What NUTS does and does not buy you

**Buys:** an exact target (MH correction, no discretisation bias — the whole
`ε`/`mdecay`/`v_hat_min` search in §4.3.16–4.3.72 disappears); automatic step
size and trajectory length, so no `sghmc_lr`/`sghmc_lr_max`/`cycle_length`/
`fraction_cool` to sweep at all; divergence counts, E-BFMI and tree-depth
saturation as *actionable* diagnostics with known failure signatures; and ESS
per iteration near 1 instead of ~1/35, which resolves the entire §4.3.67
"steps per independent sample" problem by construction.

**Does not buy:** mode hopping. NUTS explores one basin. §4.5 found localised
multimodality in a few maze regions, and the cyclical schedule exists precisely
to jump between modes — that is the one thing cSGHMC does that NUTS does not.
If the multimodality is real and matters, the honest answer is multiple NUTS
chains from dispersed inits, disclosed as such, or a mixture (NUTS within
cyclically-restarted basins). Weight-space symmetry modes are harmless — they
give identical `f` — but §3.6.2's rule still stands: judge on function-space
diagnostics, never on weight-space R-hat.

**Does not buy:** minibatching. Betancourt (2015) shows naive subsampling is
incompatible with HMC — the energy error from a stochastic gradient destroys the
acceptance probability. You need full-batch gradients, which is exactly why this
is worth trying *here* and would not be elsewhere.

---

## 8. Feasibility numbers

Dataset sizes inferred from HDF5 file sizes (h5py is not available in this
sandbox — **check these against the actual arrays**), assuming float32 storage
and the documented shape `(N, 2, T=100, d_dim=38)`:

| variant | train | val | test | note |
|---|---|---|---|---|
| medium_play | ≈725 | ≈156 | ≈156 | ratio 4.645 ≈ 70/15 ✓ |
| medium_diverse | ≈1,010 | ≈217 | ≈217 | |
| large_play | ≈515 | ≈110 | ≈110 | |
| large_diverse | ≈1,042 | ≈223 | ≈223 | |

Per full-batch gradient, medium_play: 725 pairs × 2 segments × 100 timesteps =
145,000 rows through a 6,657-parameter MLP (a few ms on an A6000), plus one
26×26 triangular solve and one VJP. Call it **~3 ms**.

At tree depth 7–9 (from the `√cond` estimate in §4.1) that is 128–512 gradients
per draw, i.e. **0.4–1.5 s/draw**, so 2,000 draws (1,000 warm-up) is
**15–50 minutes per chain**. Four chains on four GPUs: under an hour. That is
comparable to one current fSGHMC run and gives ~1,000 effective draws instead of
~2 per chain.

Same arithmetic for large_diverse (14,977 params) is ~2× that. For the two
million-parameter variants it is not the wall-clock that fails, it is §2.

---

## 9. What I would actually do

1. **Switch `transfer_fn` to `tanh`** and re-measure the current fSGHMC finals.
   One line, and it is a prerequisite for HMC — but note it addresses only the
   smoothness problem (§5), not propriety (§2a), which needs §6A regardless.
2. **Build the cell-space reference posterior (§6B) in NumPyro.** ~26 dimensions,
   exact NUTS, a day of work. Then compare the fSGHMC function-space ensemble
   against it. This is the highest-value item in this note and it does not
   depend on any of the rest working.
3. **Only then** decide whether to pursue BNN NUTS. If you do: medium_play and
   large_diverse only, weight prior added (§6A), one fixed measurement point per
   cell with a precomputed Cholesky (§3), explicit offset reparameterisation
   (§4.1), fixed GGN + diagonal metric (§4.2), NumPyro or BlackJAX rather than
   this codebase.
4. **Write the theory up in `(w, z)`, not in `(f, g)`**, or supply the `Γ` term
   (§1.1). The functional content of the contribution is the metric and the
   U-turn identity (§4.2), not the dynamics.

## Sources

- Wu, Xuan & Lu (2025), *Functional Stochastic Gradient MCMC for Bayesian Neural
  Networks*, AISTATS — §3.2, Eq. 7–9, Table 1/6, App. B–E, Alg. 1–2.
- Ma, Chen & Fox (2015), *A Complete Recipe for Stochastic Gradient MCMC*,
  NeurIPS — Thm. 1 and the `Γ` correction.
- Betancourt (2015), *The Fundamental Incompatibility of Scalable Hamiltonian
  Monte Carlo and Naive Data Subsampling*, ICML.
  <https://proceedings.mlr.press/v37/betancourt15.html>
- Hoffman & Gelman (2014), *The No-U-Turn Sampler*, JMLR 15; Betancourt (2017),
  *A Conceptual Introduction to Hamiltonian Monte Carlo* — the generalised
  termination criterion under a non-identity metric.
- Izmailov, Vikram, Hoffman & Wilson (2021), *What Are Bayesian Neural Network
  Posteriors Really Like?*, ICML — full-batch HMC on BNNs, and what it costs.
