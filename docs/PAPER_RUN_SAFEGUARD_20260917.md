# 2026-09-17 full-paper rerun: model-safeguarded MPC residual

## 1. Result status before this revision

The completed 5-seed/100k unified MuJoCo result set is **not** a consistent win over MPC.

| Environment | MPC | Planning residual | ZPRL-style |
|---|---:|---:|---:|
| HalfCheetah-v5 | 2464.60 | 2347.28 | 176.07 |
| Hopper-v5 | 371.85 | 314.85 | 171.91 |
| Reacher-v5 | -7.374 | -7.143 | -5.690 |
| Walker2d-v5 | 214.43 | 225.33 | 404.78 |

The main planning-residual method is close on HalfCheetah, below MPC on Hopper, and above MPC on Reacher/Walker2d. ZPRL-style is a proxy comparator and should not be used as the paper's proposed method.

## 2. Revision

A lightweight world-model safeguard is inserted **after** the SAC residual is proposed and **before** the environment action is executed.

For a residual correction \(\Delta \mathbf a\), the controller evaluates the same correction at scales

\[
\lambda\in\{1, 0.5, 0.25, 0\}.
\]

Scale 0 is the original MPC plan. The frozen/current world model scores all four plans using the same rollout objective as CEM. A nonzero residual is accepted only when its predicted return is not below the rescored MPC plan (or exceeds it by `safeguard_min_improvement` when a positive margin is configured).

This does **not** change the MPC baseline search, training budget, seeds, environment, or evaluation seeds. It only constrains the proposed RL residual controller.

Logged diagnostics:

- `safeguard_scale`: accepted fraction of the proposed residual;
- `safeguard_gain`: predicted return gain over the unmodified MPC plan;
- existing gate/residual/prediction/action-smoothness metrics remain unchanged.

The same idea is implemented for the direct paired `LeWM-MPC` vs `LeWM-MPC+RL` protocol using the frozen LeWM + reward probe to rescore candidate action sequences.

## 3. Recommended paper experiment

The primary paper experiment should use the paired LeWM protocol because it controls the world model within every `(task, seed)` pair:

```bash
bash scripts/run_lewm_mpc_rl_paper_full.sh
```

It runs 4 tasks × 5 seeds. Each job trains one shared LeWM, evaluates `LeWM-MPC`, freezes the LeWM, trains only the residual controller, and evaluates `LeWM-MPC+RL` on identical evaluation seeds. On completion it automatically creates strict coverage tables, paired summaries, 300-dpi PNG figures, and vector PDF figures.

For the broader 4-method engineering benchmark:

```bash
bash scripts/run_paper_suite_full.sh
```

This runs the existing `MPC / Action Residual / Planning Residual / ZPRL-style` suite, aggregates deterministic final evaluation, and now also calls `experiments.paper_figures` automatically.

## 4. Paper figures and quantitative evidence

The generic suite produces:

- deterministic return mean ± std with all seed points;
- relative return vs MPC;
- same-seed `Planning Residual - MPC` differences with 95% CI;
- learning curves (mean ± std);
- learning-curve AUC under the same step budget as a sample-efficiency measure;
- prediction MSE, action smoothness, effective residual, gate, safeguard scale/gain, and MPC time diagnostics;
- 300-dpi PNG and vector PDF outputs.

The direct LeWM paired suite additionally produces per-seed paired improvement, success rate when defined, shared latent effective-rank diagnostics, RL learning curves, and safeguard-scale curves.

## 5. Acceptance criteria

Do not claim that the proposed method exceeds baseline from a single seed. Recommended minimum acceptance criteria are:

1. all 4 tasks × 5 seeds are complete under identical budgets;
2. deterministic final evaluation uses the same evaluation seeds inside each pair;
3. report mean ± std and paired 95% CI;
4. report the number of positive paired seeds, not only the grand mean;
5. inspect sample-efficiency AUC and prediction/action diagnostics for regressions;
6. keep the old completed result set unchanged as the pre-safeguard ablation/control;
7. report the new safeguard as an explicit method component/ablation rather than silently replacing the old controller.

The safeguard is designed to reduce controller regressions according to the learned model. It is **not** a guarantee of higher real-environment return because model/reward-probe error can still rank candidate corrections incorrectly. Full reruns are required before making a performance claim.
