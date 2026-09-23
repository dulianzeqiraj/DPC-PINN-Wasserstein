# DPC-PINN-Wasserstein

Physics-informed deep learning for **joint hydraulic conductivity–porosity mapping** with a
**Wasserstein-2 optimal experimental design** criterion, applied to the Fushë-Kuqe alluvial
aquifer (northwestern Albania).

MATLAB implementation and curated dataset accompanying the manuscript:

> D. Zeqiraj, *Physics-Informed Deep Learning for Joint Hydraulic Conductivity–Porosity Mapping
> with Wasserstein Optimal Experimental Design in Alluvial Aquifers*.

Release **v2.8.0**. Every number in the manuscript comes from this release.

### What changed in v2.8.0

- **The anisotropic prior pointed the wrong way.** Up to v2.7 the ratio R of Eq. (6) weighted the
  across-channel gradient, which rewards a field elongated perpendicular to the channel fabric.
  A channel is a body along which conductivity barely changes, so R now weights the along-channel
  gradient (`src/model_loss.m`). Every result that trains a model with L5 on was recomputed.
- **No held-out well enters a prior.** The facies class of each collocation point and the
  calibration of the valley-geometry prior used all 37 wells, including the ones held out. In
  cross-validation, the factorial ablation and the bootstrap they are now rebuilt from the training
  wells of each fold or resample (`src/fold_priors.m`).
- The climatological floor of the accuracy table is the training-fold mean on the same folds, not
  the standard deviation of all 37 wells (`summarize_all.m`).
- `report_betaF.m` labelled the medium and fine prefactors the wrong way round; the smoke test's
  training check could not fail and now trains long enough to mean something.

## What the code does

A physics-informed neural network with a shared trunk and two property heads infers the coupled
fields *K*(**x**) and *n*(**x**) jointly with the hydraulic head *h*(**x**), under a six-term
loss plus a weak boundary penalty: steady Darcy residual, head misfit, conductivity misfit, a
paleo-channel anisotropic prior, a Kozeny–Carman coupling, and a valley-geometry prior.
Predictive uncertainty is obtained by Monte-Carlo dropout combined with a bootstrap ensemble and
is calibrated by distribution-free split-conformal prediction. On the calibrated posterior,
candidate well locations are scored by the expected squared 2-Wasserstein gap between posterior
and prior (entropic Sinkhorn solver) and selected by sequential greedy optimization.

### On the DRASTIC layer, and on a term that was removed

Earlier versions carried a seventh loss term, `L4`, tying predicted log-conductivity to a
standardized DRASTIC vulnerability index over 180 points. It was removed. Over the 43 index
points that also carry a pumping test, index and log-conductivity are all but uncorrelated
(r = +0.15, r² = 0.02), and in the factorial ablation adding the term to the physics-and-data
baseline moves held-out log-conductivity RMSE from 0.421 to 0.424, on identical folds
(`results/ablation_factorial_rows.csv`, rows `base` and `base+L4`). The term
survives in the code behind `cfg.w.L4 > 0` so that the ablation can still quantify it; the
canonical configuration sets that weight to zero, and the loss-term labels keep their original
numbering, so the six terms are L1, L2, L3, L5, L6 and L7.

The DRASTIC field is still used: as a covariate of the regression-kriging baseline, and as the
diagonal weighting of the Wasserstein-2 design criterion, which is a statement about where
information is worth having rather than a claim that the index predicts conductivity.

A companion study on DRASTIC-based vulnerability assessment (Zeqiraj et al., 2026,
*Journal of Hazardous Materials Advances* 23, 101261) sets out the physical background.

## Requirements

- **MATLAB R2026a**
- Deep Learning Toolbox, Statistics and Machine Learning Toolbox, Parallel Computing Toolbox
- A GPU is optional. All results here were produced on a laptop with an NVIDIA GTX 1050;
  `cfg.forceCPU = true` runs the same pipeline on the CPU.

## Reproducibility

Master seed **20260610** throughout. `main_cv.m` and `main_ablation_factorial.m` also
seed each fold and each ablation cell separately, so a result does not depend on what ran before
it and an interrupted run resumes from its checkpoints without changing any number.

Checkpoints are keyed only on the unit of work (configuration, fold, member, design), not on
the loss or the data. After any change to the loss, the data or the configuration, delete the
matching files in `results/synth`, `results/cv_ckpt`, `results/boot`, `results/ablation_fac`
and `results/closed_loop` before rerunning, or the old results are silently reused.

`smoke_test.m` checks the data, the boundary roles, the fold split and a short training pass
before any long run. `verify_precision.m` is the FP32 gate: it trains the same fold in single and
double precision and accepts single only if the held-out errors agree.

## How to run

- `RUN_ALL.m` is the canonical chain, from the environment report to the last figure. Every
  training stage checkpoints per unit of work and skips what is already on disk, so it can be
  interrupted and relaunched. Run it from `matlab/` with the working directory set there.

Stages, in the order they are meant to run:

| script | what it does |
| --- | --- |
| `env_report.m` | records MATLAB version, GPU, toolboxes, free memory |
| `main_fushekuqe.m` | joint inversion on all 37 wells, writes the *K* and *n* fields |
| `main_cv.m` | five-fold cross-validation on seeded random folds, one checkpoint per fold; facies prior and geo-prior refitted on each fold's training wells |
| `main_baselines.m` | ordinary kriging, regression kriging with DRASTIC, random forest |
| `summarize_all.m` | recomputes the accuracy table from the per-well files |
| `uq_bootstrap.m` | 30-member bootstrap plus MC dropout, out-of-bag predictions; priors refitted on each resample |
| `conformal_calibration.m` | split-conformal multipliers, both protocols, side by side |
| `main_oed_w2.m` | Wasserstein-2 design as Eqs (11) to (13) define it, plus the five-design comparison; loads the stored posterior, trains nothing |
| `main_synthetic.m` | recovery of known conductivity and porosity fields, with the loss ablation repeated on them |
| `main_closed_loop.m` | drills each design on the synthetic aquifer and refits, so the criterion is scored by what it recovers |
| `main_ablation_factorial.m` | all eight subsets of {L5, L6, L7}, plus a DRASTIC evidence row |
| `main_ablation.m` | the older two-mode ablation, kept for comparison |

Figures:

| script | figure |
| --- | --- |
| `make_fig4_framework.m` | Figure 4, the framework diagram |
| `make_fig7_reliability.m` | Figure 7, reliability of the predictive intervals |
| `make_fig8_rmse_bars.m` | Figure 8, held-out error by method |
| `make_fig9_oed.m` | Figure 9, the two design-utility panels |
| `make_fig11_synthetic.m` | Figure 11, redrawn from stored results without retraining |
| `make_figS1_variogram.m` | Figure S1, directional semivariograms |
| `make_figS3_boundary.m` | Figure S3, where each boundary condition acts |
| `tools/make_fig1_site.py` | Figure 1, right panel: the aquifer, wells and channel network |

Figure 10, the design comparison, is written by `main_oed_w2.m`; Figure 12, the closed-loop
validation, by `main_closed_loop.m`. `main_synthetic.m` writes a first version of Figure 11 at
the end of its run; `make_fig11_synthetic.m` redraws it from the stored results, so the layout
can be changed without repeating a two-hour fit.

Figures 5 and 6 are written by `main_fushekuqe.m`. Figures 2 and 3 are maps prepared outside
this repository; `tools/merge_morphology.py` documents how the channel network of Figure 3 was
extracted from the two legacy maps. Figure 1 is half and half: its right-hand panel is drawn
from `data/` by `tools/make_fig1_site.py`, and its left-hand locator is cartography prepared
outside the repository and carried over as an image.

## What this repository does not contain

Stated plainly, because the manuscript is judged on it.

- **No transport data, so no data-driven porosity.** Porosity here is identified through an
  assumed Kozeny-Carman coupling. Where concentration or temperature observations exist,
  porosity can be identified from the data instead (Li et al., 2012; Xu and Gomez-Hernandez,
  2016), which is the stronger position; Fushe-Kuqe has no such observations.
- **No transient calibration.** The model reads a single-epoch head snapshot as a quasi-stationary
  annual mean. There is no time dimension anywhere in the code.
- **No recharge and no pumping term.** `cfg.N = 0` in every driver and the PDE residual carries no
  source. Lateral underflow and mountain-front recharge are absorbed by the free land boundary
  rather than resolved.
- **The design study uses the MC-dropout posterior, not the bootstrap ensemble.** The bootstrap
  members store per-well predictions only, not parameters, so the field samples that Eq. (12)
  reweights come from dropout alone. Widening that would mean storing 30 sets of weights.
- Figures 2 and 3, and the locator panel of Figure 1, cannot be regenerated from this
  repository. The rest of Figure 1 can.

## Data

`data/` holds the curated dataset:

- `inversion_wells_37.csv`: 37 monitoring wells: coordinates, *K*, head, facies, porosity priors
- `drastic_grid_180.csv`: 180-point DRASTIC index grid
- `boundary_349km2.csv`: digitized aquifer boundary, Gauss–Krüger Zone 4
- `river_axes.csv`, `channel_points_merged.csv`, `orientation_field_grid.csv`: the fluvial fabric
  used by the anisotropic and valley-geometry priors
- `well_anisotropy_37.csv`, `Fushe_Kuqe_All_Data.xlsx`: source compilation

Conductivities in the source database are rounded to six levels between 50 and 200 m/day. That
quantization is a property of the data, not of the code, and it bounds what any of these results
can resolve.

## License

- **Code:** MIT (see `LICENSE`).
- **Data:** CC-BY-4.0.

## Citation

Cite the software through `CITATION.cff` together with the manuscript. Each release is archived
on Zenodo with its own DOI.
