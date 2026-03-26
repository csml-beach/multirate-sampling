# Adaptive Multirate SVGD: Ideas for Robustness vs Cost

Below are concrete strategies to make multirate SVGD more adaptive while keeping
computational cost and stability in mind.

## A. Event-triggered repulsion (cheap, robust)
- **Rule:** Always drift each step; apply repulsion only when particles get too
  close (e.g., min pairwise distance < τ or kernel bandwidth < τ).
- **Why:** Repulsion is O(N^2); skip it when diversity is already good.
- **Stability:** Prevents collapse while limiting unnecessary repulsion.
- **Cost:** Low; extra kernel evals only during crowding.

## B. Error-controlled drift substeps (stable, modest cost)
- **Rule:** Compare one drift step of size `dt` vs two half-steps; if the error
  exceeds a tolerance, increase drift substeps.
- **Why:** ODE-style control reacts directly to stiffness.
- **Stability:** Strong; less heuristic than fixed thresholds.
- **Cost:** 1 extra grad eval when you test; can be amortized every K steps.

## C. Repulsion frequency schedule (simple and predictable)
- **Rule:** Apply repulsion every `k` steps; update `k` on a slow timescale using
  KSD or ESS slope.
- **Why:** Avoids per-step jitter, easy to analyze and compare.
- **Stability:** Good if `k` is capped.
- **Cost:** Predictable and easy to budget.

## D. Scale-invariant stiffness ratio
- **Rule:** Use a ratio `r = rms(drift_step) / (rms(rep_step) + eps)` or normalize
  by an EMA of its own scale before mapping to `m`.
- **Why:** Avoids saturation due to absolute scale.
- **Stability:** More portable across targets.
- **Cost:** 1 kernel + 1 grad per step (already computed).

## E. Mixed-accuracy repulsion
- **Rule:** Use approximate repulsion (low-rank or mini-batch kernel) in
  substeps; do a full kernel once per iteration.
- **Why:** Keeps stabilizing effect without O(N^2) each substep.
- **Stability:** Good if approximation is unbiased or conservative.
- **Cost:** Lower kernel cost when `m` is large.

## F. Per-particle adaptivity (targeted robustness)
- **Rule:** Apply extra substeps only to crowded particles; leave the rest alone.
- **Why:** Crowding is local; no need for global cost.
- **Stability:** Avoids particle collapse.
- **Cost:** More bookkeeping; can still be O(N^2) without approximation.

## G. Adaptive bandwidth instead of adaptive m
- **Rule:** Keep `m` fixed, adapt kernel bandwidth based on distance statistics.
- **Why:** Repulsion strength depends heavily on bandwidth.
- **Stability:** Often better than aggressive m changes.
- **Cost:** No extra evals.

## H. Guard-triggered control
- **Rule:** Change substepping only when nonfinite_frac spikes or mean_logp
  decreases consistently.
- **Why:** Conservative; reacts only to clear failure signals.
- **Cost:** Minimal.

## Benchmark Extensions (Beyond Gaussian)
Implemented in this repo:
- **Bayesian logistic regression (standard):** UCI datasets (e.g., breast cancer, ionosphere, a5a) with a Gaussian prior. Report posterior predictive accuracy, log-loss, and calibration. This is a classic benchmark with well-defined likelihood and gradients.
- **Small Bayesian neural net (UCI classification):** 1-hidden-layer BNN with Gaussian prior on UCI datasets (`breast_cancer`, `ionosphere`, `a5a`), using the same sampler interface and reporting Accuracy/NLL/ECE/ESS/KSD.
- **Mixture models:** Moderate-dimensional Gaussian mixture with known modes to test mode coverage and repulsion behavior.
- **Hierarchical logistic regression (synthetic, large-scale):** grouped random-effects benchmark with `longtail` and `uniform` group modes, reporting Accuracy/NLL/ECE/ESS and cost metrics, with NLL-based early stopping.

Additional candidate extensions:
- **Bayesian linear regression (closed form check):** Synthetic data with known posterior to validate correctness and scaling. Useful for sanity checks and step-size tuning.
- **Hierarchical / funnel targets:** Neal’s funnel (in addition to implemented HLR) to stress geometry and anisotropy.
- **Robust regression:** Student-t likelihood with Gaussian prior (heavier tails, outlier sensitivity). Tests stability under non-Gaussian likelihoods.
- **Small Bayesian neural net (regression):** 1–2 hidden layers on a small UCI regression dataset (e.g., Boston, Yacht). Use a Gaussian prior and compare predictive NLL and RMSE.

### UCI Logistic Regression (detail)
- **Why:** Standard, light-weight, gradients easy, no GPU needed.
- **Datasets:** breast cancer, ionosphere, a5a (Adult), wdbc.
- **Metrics:** test accuracy, NLL, calibration (ECE); plus KSD on posterior samples if desired.
- **Implementation:**
  - Standardize features; add intercept.
  - Gaussian prior on weights.
  - Log-posterior = log-likelihood + log-prior.
  - Use the same sampler interface as the 50D benchmark.

### UCI BNN (detail, implemented)
- **Why:** Nonlinear posterior benchmark that is still lightweight enough for CPU runs.
- **Datasets:** `breast_cancer`, `ionosphere`, `a5a`.
- **Metrics:** test accuracy, NLL, ECE, ESS, KSD, mean log-prob.
- **Implementation:**
  - 1-hidden-layer BNN with tanh activations.
  - Gaussian prior on parameters.
  - Posterior predictive probabilities computed by averaging sigmoid outputs over sampled particles/states.
  - Uses the same sampler interface as the 50D and UCI logistic benchmarks.
