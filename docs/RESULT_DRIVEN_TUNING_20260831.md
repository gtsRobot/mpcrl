# Result-driven tuning after the first 5-seed RTX 5090 suite

The first completed/near-completed 100k-step runs were summarized with the mean of the final 20 episodes for each seed.

| Environment | MPC | Action residual | Planning residual | ZPRL-style |
|---|---:|---:|---:|---:|
| HalfCheetah-v5 | 2293.05 ± 1054.42 | 2421.97 ± 1267.68 | **2594.27 ± 578.76** | -64.99 ± 5.61 |
| Hopper-v5 | 359.37 ± 110.01 | **380.25 ± 22.27** | 331.25 ± 23.93 | 71.41 ± 66.31 |
| Walker2d-v5 | **220.40 ± 37.94** | 208.33 ± 16.63 | 207.89 ± 30.96 | 152.78 ± 23.45 |

These values are diagnostic statistics from the supplied run summary, not final paper claims.

## Changes made

1. **Planning residual parameterization**
   - Previous implementation produced `chunk_len × action_dim` SAC outputs but receding-horizon MPC executed only the first action. The unused future residual dimensions increased the critic/actor action dimension without directly affecting the environment reward.
   - The optimized implementation learns one `action_dim` residual conditioned on the complete MPC chunk and expands it across the plan with geometric decay. This preserves a coherent plan-level correction while ensuring every learned residual dimension affects the executed action.

2. **Environment-specific conservative tuning**
   - HalfCheetah keeps the previous gate strength because planning residual was already the best and substantially more stable.
   - Hopper uses a less restrictive uncertainty gate because action residual clearly outperformed the previous gated planning residual.
   - Walker2d reduces residual scale and raises the uncertainty threshold because MPC-only remained the strongest method; residual correction is now used only more selectively.

3. **ZPRL-style proxy repair**
   - The previous base policy was behavior-cloned from only ~2048 random-warmup states and each nominal BC epoch performed a single random mini-batch update.
   - The new implementation collects a dedicated MPC rollout dataset, updates the world model during collection, performs true full-dataset BC epochs, freezes the base policy, and ramps latent residual strength from 0 to its configured maximum.

4. **Experiment observability**
   - Added `tqdm` progress bars with percentage, elapsed time, ETA, episode return, episode length, world-model loss, gate and cache hit rate.
   - The paper suite now displays `job / total jobs` and supports `--start-job N` after an interruption.

5. **Aggregation correctness**
   - Repeated/restarted run directories are no longer double-counted. For each `(env, method, seed)`, aggregation keeps the run with the largest `global_step`; ties choose the newest run.
   - Default paper aggregation uses the final 20 episodes per selected run and exports both per-seed and mean±std CSV files.

## Validation

The optimized source passes Python compilation and 11 unit tests, including the new temporal planning-residual tests.
