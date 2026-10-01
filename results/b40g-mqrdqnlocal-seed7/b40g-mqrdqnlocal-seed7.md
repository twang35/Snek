# b40g-mqrdqnlocal-seed7

step **3,000,000** · 3000 evals · trailing **93.85** · peak **94.44** @1,471,000 · sef **64.2** · best30 **95.9** @1,807,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | qrdqn |
| batch_size | 128 |
| beta_anneal_steps | 300000 |
| collect_envs | 1 |
| discount | 0.99 |
| dist_atoms | 51 |
| dist_embedding | 64 |
| dist_fraction_entropy | 0.001 |
| dist_fraction_lr | 2.5e-09 |
| dist_kappa | 1.0 |
| dist_policy_samples | 32 |
| dist_quantiles | 32 |
| dist_risk_alpha | 1.0 |
| dist_risk_train | False |
| dist_tau_prime_samples | 64 |
| dist_tau_samples | 64 |
| dist_v_max | 110.0 |
| dist_v_min | -10.0 |
| epsilon_anneal_steps | 250000 |
| epsilon_schedule | eval |
| eval_interval | 1000 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| fork_branches | 4 |
| fork_max_steps | 60 |
| fork_min_length | 85 |
| fork_prob | 0.5 |
| gradient_clipping | 0.0 |
| graph_eval_episodes | 100 |
| guided_fraction | 0.8 |
| init_from | None |
| initial_collect_steps | 2000 |
| initial_epsilon | 0.4 |
| is_beta | 0.4 |
| is_beta_final | 1.0 |
| is_weights | True |
| learning_rate | 1e-05 |
| max_steps | 3000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.002 |
| munchausen_alpha | 0.9 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 1 |
| priority_exponent | 0.6 |
| replay_buffer_max_length | 100000 |
| replay_ratio | 1.0 |
| seed | 7 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b40g-mqrdqnlocal-seed7](b40g-mqrdqnlocal-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.44 | 0.44 | 0.0 | 2.0 | -0.113 | 0.0 | 0.4 |
| 2000 | 0.53 | 0.48 | 0.0 | 4.0 | -0.024 | 0.0 | 0.4 |
| 3000 | 0.67 | 0.55 | 0.0 | 5.0 | 0.116 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 92.2 | 93.8 | 32.0 | 95.0 | 177.532 | 87.0 | 0.002 |
| 2990000 | 94.58 | 93.82 | 84.0 | 95.0 | 185.986 | 93.0 | 0.002 |
| 2991000 | 94.06 | 93.86 | 45.0 | 95.0 | 184.429 | 92.0 | 0.002 |
| 2992000 | 93.98 | 93.89 | 66.0 | 95.0 | 183.332 | 91.0 | 0.002 |
| 2993000 | 93.97 | 93.87 | 79.0 | 95.0 | 179.285 | 87.0 | 0.002 |
| 2994000 | 94.2 | 93.88 | 72.0 | 95.0 | 181.467 | 89.0 | 0.002 |
| 2995000 | 94.22 | 93.89 | 70.0 | 95.0 | 182.553 | 90.0 | 0.002 |
| 2996000 | 93.73 | 93.89 | 36.0 | 95.0 | 181.023 | 89.0 | 0.002 |
| 2997000 | 93.96 | 93.9 | 46.0 | 95.0 | 184.334 | 92.0 | 0.002 |
| 2998000 | 93.77 | 93.9 | 53.0 | 95.0 | 179.931 | 88.0 | 0.002 |
| 2999000 | 93.08 | 93.86 | 27.0 | 95.0 | 181.455 | 90.0 | 0.002 |
| 3000000 | 93.61 | 93.85 | 12.0 | 95.0 | 183.967 | 92.0 | 0.002 |
