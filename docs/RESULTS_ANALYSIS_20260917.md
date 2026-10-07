# Full 5-seed result review and targeted refinement

The 100k deterministic-evaluation summary contained 4 environments, 4 methods, and 5 seeds (80 runs), all marked complete.

Mean deterministic return across five seeds:

| Environment | MPC | Action residual | Planning residual | ZPRL-style | Planning vs MPC |
|---|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | 2464.60 | 1789.30 | 2347.28 | 176.07 | -4.76% |
| Hopper-v5 | 371.85 | 360.96 | 314.85 | 171.91 | -15.33% |
| Reacher-v5 | -7.374 | -7.107 | -7.143 | -5.690 | +3.13% |
| Walker2d-v5 | 214.43 | 221.03 | 225.33 | 404.78 | +5.08% |

The current proposed planning residual is therefore not yet a consistent improvement over MPC: it improves Reacher and Walker2d, is close on HalfCheetah, and is materially below MPC on Hopper.

## Why the refinement is limited to two tasks

HalfCheetah has strong seed dependence in the uncertainty gate. The two strongest seeds use mean gates around 0.41-0.47, while weak seeds can fall to roughly 0.09-0.19. The refinement config therefore constrains the gate to a moderate trust region rather than allowing almost complete residual shutdown.

Hopper's planning residual uses a mean gate around 0.45-0.52 but its effective correction is smaller than the action-residual comparator. The refinement increases raw residual scale slightly while bounding gate authority, so its maximum effective correction remains below full action-residual control.

Reacher and Walker2d are frozen because planning residual already improves on MPC there. Baseline configurations and the completed 80-run result set are preserved for reproducibility; refinements live under `configs/rtx5090/refine/` rather than silently replacing the original experiment configuration.

## Recommended next computation

Do **not** rerun all baselines. Run only planning residual for HalfCheetah and Hopper with five seeds:

```bash
bash scripts/run_planning_refine.sh
```

Afterwards regenerate summaries and publication figures:

```bash
bash scripts/make_paper_figures.sh
```

The plotting utility outputs PNG and vector PDF figures, including deterministic return mean±std, relative return vs MPC, prediction error, action smoothness, effective residual/gate, MPC time, episode length, and learning curves when `runs/**/episodes.csv` is available.
