# b43b-reset600k-seed2

step **3,000,000** · 3000 evals · trailing **93.88** · peak **94.0** @2,988,000 · sef **0.0** · best30 **62.4** @2,887,000

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
| reset_alpha | 0.5 |
| reset_interval | 600000 |
| reset_stop_after | 10500000 |
| seed | 2 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b43b-reset600k-seed2](b43b-reset600k-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.64 | 0.64 | 0.0 | 3.0 | 0.087 | 0.0 | 0.4 |
| 2000 | 0.61 | 0.62 | 0.0 | 3.0 | 0.057 | 0.0 | 0.4 |
| 3000 | 0.6 | 0.62 | 0.0 | 5.0 | 0.047 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 93.84 | 94.0 | 82.0 | 95.0 | 151.277 | 60.0 | 0.00334 |
| 2990000 | 94.03 | 94.0 | 83.0 | 95.0 | 149.282 | 58.0 | 0.00332 |
| 2991000 | 93.65 | 94.0 | 46.0 | 95.0 | 149.043 | 58.0 | 0.00331 |
| 2992000 | 93.55 | 93.99 | 76.0 | 95.0 | 140.391 | 50.0 | 0.00331 |
| 2993000 | 93.62 | 93.97 | 83.0 | 95.0 | 146.982 | 56.0 | 0.0033 |
| 2994000 | 93.47 | 93.95 | 85.0 | 95.0 | 145.926 | 55.0 | 0.00331 |
| 2995000 | 93.46 | 93.93 | 80.0 | 95.0 | 139.744 | 49.0 | 0.00331 |
| 2996000 | 93.86 | 93.92 | 87.0 | 95.0 | 142.003 | 51.0 | 0.00332 |
| 2997000 | 93.98 | 93.92 | 84.0 | 95.0 | 155.578 | 64.0 | 0.00333 |
| 2998000 | 93.62 | 93.91 | 78.0 | 95.0 | 145.735 | 55.0 | 0.00335 |
| 2999000 | 93.75 | 93.9 | 82.0 | 95.0 | 141.648 | 51.0 | 0.00337 |
| 3000000 | 93.72 | 93.88 | 77.0 | 95.0 | 143.688 | 53.0 | 0.00335 |
