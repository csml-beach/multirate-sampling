
# LaTeX Labels Reference

This file tracks all \label{} keys used in equations, figures, and tables in the paper, with a brief description for each. Update this file whenever you add a new label.

## Equations

- eq:svgd-update — Discrete SVGD update equation (method section)
- eq:svgd-velocity — SVGD velocity field (method section)
- eq:svgd-rbf-kernel — RBF kernel definition for SVGD (method section)
- eq:svgd-splitting — SVGD splitting into drift and repulsion (method section)
- eq:svgd-drift-term — SVGD drift component $f_{\\text{drift}}$ definition
- eq:svgd-rep-term — SVGD repulsion component $f_{\\text{rep}}$ definition
- eq:strang-split-svgd — Strang-split SVGD macro-step update
- eq:mr-svgd — Fixed multirate SVGD (MR-SVGD) macro-step update
- eq:adapt-mr-step — Adapt-MR-SVGD macro-step with adaptive $m$
- eq:adapt-mr-error — Adapt-MR-SVGD step-doubling error indicator
- eq:adapt-mr-m — Adapt-MR-SVGD drift substep selection rule
- eq:sgld-update — SGLD baseline update used in experimental protocol
- eq:sghmc-update — SGHMC-style baseline update used in experimental protocol
- eq:ksd-stein-kernel — Score-Stein kernel used in the KSD definition
- eq:ksd-metric — Empirical kernel Stein discrepancy used for evaluation
- eq:mean-logp-metric — Mean log-density (log-score) evaluation metric
- eq:hlr-likelihood — Hierarchical logistic-regression likelihood for grouped binary outcomes
- eq:hlr-priors — Non-centered hierarchical priors for global and group-level effects
- eq:hlr-posterior — Posterior density (up to normalization) for the large-scale HLR benchmark

## Figures

- fig:gauss50_pareto — 50D Gaussian Pareto plot (moment-error space)
- fig:gauss50_summary — 50D Gaussian summary metric panels
- fig:gauss50_walltime — 50D Gaussian wall-time overview
- fig:2d_demo_panel — 2D target visualization panel (banana, squiggly, ring, two-moons demos)
- fig:2d_demo_banana — Banana demo subpanel (initial vs short-run final particles)
- fig:2d_demo_squiggly — Squiggly demo subpanel (initial vs short-run final particles)
- fig:2d_demo_ring — Ring demo subpanel (initial vs short-run final particles)
- fig:2d_demo_two_moons — Two-moons demo subpanel (initial vs short-run final particles)
- fig:mix8_summary — Mixture2D (mix8) summary metric panels
- fig:uci_summary — UCI logistic regression summary panels (all datasets)
- fig:uci_breast_cancer — UCI breast\_cancer subpanel
- fig:uci_ionosphere — UCI ionosphere subpanel
- fig:uci_spambase — UCI spambase subpanel
- fig:uci_a5a — UCI a5a subpanel
- fig:bnn_summary — Bayesian neural network (BNN) summary panels
- fig:bnn_spambase — BNN spambase subpanel
- fig:bnn_a5a — BNN a5a subpanel

## Tables

- tab:2d_summary — 2D targets quality-cost summary table (median final metrics)
- tab:mix8_summary — Mixture2D (mix8) mean final-metric summary table over five seeds
- tab:uci_nll_summary — UCI logistic-regression particle-method summary by dataset using best finite-NLL checkpoints
- tab:bnn_nll_summary — BNN particle-method summary by dataset using best finite-NLL checkpoints
- tab:hlr_longtail_summary — HLR long-tail quality-cost summary table (best finite-NLL checkpoints)
- tab:hlr_uniform_summary — HLR uniform-group quality-cost summary table (best finite-NLL checkpoints)

## Algorithms

- alg:mr_svgd — Fixed multirate SVGD (MR-SVGD) pseudocode
- alg:adapt_mr_svgd — Adaptive multirate SVGD (Adapt-MR-SVGD) pseudocode

## Sections (optional)

- sec:introduction — Introduction section
- sec:method — Method section
- sec:experiments — Experiments and Results section
- sec:conclusions — Conclusions section
