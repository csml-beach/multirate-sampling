# Paper Plan: Multirate Sampling (SVGD-focused)

Status key: [ ] planned  [~] drafting  [x] approved  [!] needs input

## Current milestone

- [x] Introduction is approved (streamlined narrative and citations aligned with current scope).
- [x] Method is approved (Strang split framing, fixed/adaptive multirate derivations, algorithm placement, and implementation remark updated).
- [~] Experiments and Results drafted with HLR section/table; final polish in progress.
- [x] Abstract revised for statistics/data-science audience and aligned with current benchmark scope.
- [x] Conclusions revised to complement (not paraphrase) the abstract.

## Agreed sections

1) Abstract
- Goal: Summarize the multirate SVGD idea, main experimental findings, and contributions.
- Sources: `README.md`, `jax/samplers.py`, benchmark outputs in `metrics/` and figures in `figures/`.
- Open questions: [ ] none for this pass.
- Status: [x] approved

2) Introduction (includes Related Work subsection)
- Goal: Background on SVGD, stiffness in repulsive vs drift terms, and motivation for multirate integration; position vs SGLD/SGHMC; state contributions; include a Related Work subsection.
- Sources: `README.md`, `docs/ideas.md`, `jax/samplers.py`, `paper/Bib/sarshar.bib`.
- Open questions: [ ] none for this pass.
- Status: [x] approved

3) Method
- Goal: Formalize SVGD dynamics, define repulsion/drift split, describe fixed multirate, Strang splitting, and adaptive error-controlled multirate variants; implementation details (bandwidth scaling, substeps, adaptive m).
- Sources: `jax/samplers.py`, `jax/benchmarks/*/benchmark_*.py`.
- Open questions: [ ] none for this pass.
- Status: [x] approved

4) Experiments (includes Results)
- Goal: Describe benchmarks, protocols, metrics, and evaluation settings, plus Results in the same section.
- Candidate subsections:
  - 50D Gaussian (metrics: mu/cov error, ESS, KSD, mean logp)
  - 2D targets (banana/ring/squiggly/two_moons/funnel; KSD primary, mean logp secondary, ESS supplemental, grad/kernel evals + wall time for cost; mu/cov and grid L1 diagnostic only)
  - UCI logistic regression (accuracy/NLL/ECE/ESS; dataset splits)
  - UCI BNN (1-hidden-layer BNN on breast_cancer/ionosphere/a5a; accuracy/NLL/ECE/ESS)
  - Mixture2D (mode coverage/entropy/min mass, KSD, grid L1)
  - HLR (longtail and uniform grouped random-effects logistic regression; NLL/ECE/accuracy/ESS/KSD plus timing)
- Sources: `README.md`, `jax/benchmarks/gauss50/benchmark_gauss50.py`, `jax/benchmarks/2d/benchmark_2d.py`, `jax/benchmarks/uci/benchmark_logreg.py`, `jax/benchmarks/mixture2d/benchmark_mixture2d.py`, `jax/benchmarks/bnn/benchmark_bnn.py`, `jax/benchmarks/hlr/benchmark_hlr.py`.
- Open questions: [!] replace remaining 2D placeholder main figure with final consolidated panel; tighten consistency/wording pass.
- Status: [~] drafting (substantive content present; publication-polish pending)

5) Conclusions
- Goal: Summarize findings, practical guidance, and future work (e.g., adaptive schedules, approximate kernels).
- Sources: `docs/ideas.md`, experimental outcomes.
- Open questions: [ ] none for this pass.
- Status: [x] approved

6) Acknowledgments
- Goal: Keep funding and compute acknowledgments synchronized with final manuscript text.
- Sources: `paper/acknowledgments.tex`.
- Open questions: [ ] none for this pass.
- Status: [x] approved

7) References
- Goal: Maintain BibTeX entries and consistent citation usage.
- Sources: papers cited in intro/method/experiments, `paper/Bib/references.bib`, `paper/Bib/sarshar.bib`, `paper/Bib/sandu.bib`, `paper/Bib/ode_multirate.bib`, `paper/Bib/ode_imex.bib`, `paper/Bib/ode_general.bib`.
- Open questions: [!] list of required citations and preferred keys; which of your papers to emphasize.
- Status: [~] drafting (bibliography files created)

## Paper table DNA (LaTeX style guide)

Use this as the default table style for `paper/experiments.tex` unless there is a strong reason to deviate.

- Use:
  - `\begin{table}[t]`
  - `\centering`
  - `\scriptsize`
  - `\setlength{\tabcolsep}{4pt}`
  - `\toprule / \midrule / \bottomrule` (`booktabs`)
- Header conventions:
  - Include optimization direction markers in metric columns, e.g. `NLL$\downarrow$`, `Accuracy$\uparrow$`.
  - Keep headers on one line when possible (prefer concise wording over forced line breaks).
- Method naming conventions (keep consistent across all tables/figures/text):
  - `SVGD`, `Strang-SVGD`, `MR-SVGD`, `Adapt-MR-SVGD`, `SGLD`, `SGHMC`.
- Ordering conventions:
  - If a table has a primary metric, sort rows by that metric (best first).
  - For winner-scorecard tables, keep dataset/metric winner layout and arrows in headers.
- Caption conventions:
  - State aggregation rule (mean/median, across what axis) and what each entry represents.
  - Keep abbreviations minimal; avoid extra legend text if method names are already explicit.
- Labeling conventions:
  - Use `\label{tab:<short_name>}` and add/update the entry in `paper/labels.md`.

Minimal template:

```tex
\begin{table}[t]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{lccc}
    \toprule
    Method & Metric A$\downarrow$ & Metric B$\uparrow$ & Wall (s)$\downarrow$ \\
    \midrule
    Adapt-MR-SVGD & ... & ... & ... \\
    MR-SVGD & ... & ... & ... \\
    SVGD & ... & ... & ... \\
    \bottomrule
  \end{tabular}
  \caption{...}
  \label{tab:example}
\end{table}
```

## Manuscript structure

- Main entrypoint: `paper/main.tex` (Elsevier `elsarticle` class).
- Section files included from `paper/main.tex`:
  - `paper/abstract.tex`
  - `paper/introduction.tex`
  - `paper/method.tex`
  - `paper/experiments.tex`
  - `paper/conclusions.tex`
  - `paper/acknowledgments.tex`
- Bibliography in `paper/main.tex`:
  - `\bibliography{Bib/references,Bib/sarshar,Bib/sandu,Bib/ode_multirate,Bib/ode_imex,Bib/ode_general}`
- Figures are mostly referenced from `../figures/...` in section files. `paper/main.tex` defines `\safeincludegraphics` to show a visible placeholder if a figure is missing.

Important: compile from `paper/` so relative `\input{...}` and `\bibliography{Bib/...}` paths resolve correctly.

## Label and citation conventions

- Track labels in `paper/labels.md`. For each new equation/figure/table/algorithm:
  1) Add `\label{...}` in LaTeX.
  2) Add the key and short description to `paper/labels.md`.
- Recommended prefixes:
  - Sections: `sec:<name>`
  - Equations: `eq:<name>`
  - Figures: `fig:<name>`
  - Tables: `tab:<name>`
  - Algorithms: `alg:<name>`
- Prefer `\cref{...}` for cross-references.
- If a new citation is added, verify the key exists in one of:
  - `paper/Bib/references.bib`
  - `paper/Bib/sarshar.bib`
  - `paper/Bib/sandu.bib`
  - `paper/Bib/ode_multirate.bib`
  - `paper/Bib/ode_imex.bib`
  - `paper/Bib/ode_general.bib`

## Editing and compile workflow

1) Edit one section file at a time (for example `paper/method.tex`).
2) Update `paper/labels.md` immediately when adding labels.
3) Verify citation keys exist before compile.
4) Compile and fix the first LaTeX error before proceeding.

Commands:
- Compile: `cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- Clean auxiliaries: `cd paper && latexmk -C`

Avoid editing generated artifacts:
- `paper/main.aux`, `paper/main.bbl`, `paper/main.blg`, `paper/main.fls`, `paper/main.fdb_latexmk`, `paper/main.log`, `paper/main.pdf`, and similar auxiliary files.

## Next actions

- Replace the remaining 2D placeholder main figure with finalized content and caption.
- Finalize publication polish in `paper/experiments.tex` (consistency, tone, and cross-references).
- Run bibliography quality pass (complete venue metadata/pages where missing for key citations).
- Run a final manuscript coherence pass across abstract/introduction/method/experiments/conclusions.
