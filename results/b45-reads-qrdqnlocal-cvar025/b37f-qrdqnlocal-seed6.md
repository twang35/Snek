# b37f-qrdqnlocal-seed6

step **3,000,000** · 3000 evals · trailing **93.67** · peak **94.69** @2,193,000 · sef **44.3** · best30 **96.9** @2,081,000

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
| munchausen_alpha | 0.0 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 1 |
| priority_exponent | 0.6 |
| replay_buffer_max_length | 100000 |
| replay_ratio | 1.0 |
| seed | 6 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b37f-qrdqnlocal-seed6](b37f-qrdqnlocal-seed6.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.82 | 0.82 | 0.0 | 7.0 | 0.266 | 0.0 | 0.4 |
| 2000 | 1.6 | 1.21 | 0.0 | 9.0 | 1.044 | 0.0 | 0.4 |
| 3000 | 49.89 | 17.44 | 1.0 | 95.0 | 60.743 | 12.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 93.54 | 93.93 | 68.0 | 95.0 | 175.622 | 84.0 | 0.002 |
| 2990000 | 93.76 | 93.95 | 70.0 | 95.0 | 181.08 | 89.0 | 0.002 |
| 2991000 | 93.36 | 93.92 | 54.0 | 95.0 | 178.674 | 87.0 | 0.002 |
| 2992000 | 93.42 | 93.89 | 68.0 | 95.0 | 179.728 | 88.0 | 0.002 |
| 2993000 | 93.52 | 93.85 | 68.0 | 95.0 | 177.95 | 86.0 | 0.002 |
| 2994000 | 93.21 | 93.81 | 70.0 | 95.0 | 173.384 | 82.0 | 0.002 |
| 2995000 | 93.81 | 93.81 | 76.0 | 95.0 | 182.146 | 90.0 | 0.002 |
| 2996000 | 94.08 | 93.8 | 76.0 | 95.0 | 180.434 | 88.0 | 0.002 |
| 2997000 | 93.0 | 93.76 | 55.0 | 95.0 | 177.301 | 86.0 | 0.002 |
| 2998000 | 93.13 | 93.71 | 64.0 | 95.0 | 176.316 | 85.0 | 0.002 |
| 2999000 | 93.45 | 93.67 | 56.0 | 95.0 | 176.577 | 85.0 | 0.002 |
| 3000000 | 94.35 | 93.67 | 78.0 | 95.0 | 181.6 | 89.0 | 0.002 |
