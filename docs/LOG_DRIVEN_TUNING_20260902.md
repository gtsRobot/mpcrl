# Log-driven tuning — 2026-09-02

Basis: three 10k quick diagnostics on RTX 5090, seeds 0/1/2, for HalfCheetah-v5, Hopper-v5 and Walker2d-v5.

## Observations

Using the mean of the last 5 completed episodes per seed:

- HalfCheetah: action residual improves early learning, while planning residual is under-gated at 10k and has higher seed sensitivity.
- Hopper: planning residual is already slightly better than MPC in the final window; its configuration is therefore intentionally left unchanged.
- Walker2d: MPC remains strongest and the task is sensitive to residual perturbations, so planning residual should be more conservative.
- MPC cache hit rates are high for residual methods, so the cache optimization is working.
- The diagnostic plotting warnings are font warnings, not training failures.

## Structural fix

Previously the SAC critic observed `(observation, MPC chunk)` but the transition actually depended on
`effective_gate * residual_scale * raw_residual`. Because `effective_gate` changes during residual ramp and adaptive uncertainty gating, the critic could observe the same context/residual pair with different executed actions.

The new context is:

`(observation, MPC chunk, effective_gate)`

The next-state context also includes the predicted next effective gate. This makes the gate/ramp state observable to the critic and removes a source of non-stationary credit assignment.

## Parameter changes

### HalfCheetah

- gate z threshold: 0.50 -> 0.25
- gate minimum: 0.05 -> 0.08
- gate maximum: 0.95 -> 0.90
- residual ramp: 5000 -> 4000 steps

Rationale: the 10k planning-residual effective gate was substantially smaller than action residual, while the earlier 100k experiments showed that planning residual can improve long-horizon performance.

### Hopper

No additional parameter tuning after the structural fix. The existing planning-residual setup already led the final 10k window.

### Walker2d

- gate_power: 1.5

The effective planning gate becomes `adaptive_gate ** 1.5 * ramp`, reducing medium-confidence interventions while retaining corrections for genuinely high uncertainty.

## Diagnostic improvements

- Adds `comparison.csv` with final-window mean/std and relative improvement versus MPC.
- Separates `requested_steps` from `last_completed_episode_step`; `episodes.csv` only records completed episodes, so the latter can be below the actual completed training step.
- Uses English diagnostic plot labels on headless Ubuntu to avoid missing-CJK-glyph warning spam.

## Validation protocol

After pulling this revision:

```bash
python -m unittest discover -s tests -v
python -m experiments.quick_diagnostic --config configs/rtx5090/halfcheetah.yaml --steps 10000 --seeds 0 1 2
python -m experiments.quick_diagnostic --config configs/rtx5090/hopper.yaml --steps 10000 --seeds 0 1 2
python -m experiments.quick_diagnostic --config configs/rtx5090/walker2d.yaml --steps 10000 --seeds 0 1 2
```

Do not compare checkpoints across the previous and current residual implementations: the SAC context dimension has changed by one feature (`effective_gate`), so old residual-agent checkpoints are intentionally incompatible.
