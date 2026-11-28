# b50a-resetanneal-seed1

step **3,000,000** · 3000 evals · trailing **93.58** · peak **93.75** @2,953,000 · sef **0.0** · best30 **67.5** @2,924,000

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
| reset_anneal_gamma | 0.97,0.997 |
| reset_anneal_n_step | 10,3 |
| reset_anneal_steps | 10000 |
| reset_interval | 600000 |
| reset_stop_after | 10500000 |
| seed | 1 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b50a-resetanneal-seed1](b50a-resetanneal-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.84 | 0.84 | 0.0 | 5.0 | 0.287 | 0.0 | 0.4 |
| 2000 | 13.81 | 7.33 | 2.0 | 27.0 | 8.847 | 0.0 | 0.4 |
| 3000 | 11.15 | 8.6 | 2.0 | 32.0 | 6.145 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 93.85 | 93.55 | 84.0 | 95.0 | 158.954 | 67.0 | 0.00317 |
| 2990000 | 93.61 | 93.55 | 54.0 | 95.0 | 157.392 | 66.0 | 0.00311 |
| 2991000 | 92.98 | 93.53 | 54.0 | 95.0 | 156.907 | 66.0 | 0.00309 |
| 2992000 | 92.8 | 93.5 | 2.0 | 95.0 | 152.716 | 62.0 | 0.00308 |
| 2993000 | 94.17 | 93.52 | 86.0 | 95.0 | 153.55 | 62.0 | 0.00308 |
| 2994000 | 93.44 | 93.51 | 68.0 | 95.0 | 157.222 | 66.0 | 0.00308 |
| 2995000 | 93.58 | 93.5 | 68.0 | 95.0 | 149.132 | 58.0 | 0.00307 |
| 2996000 | 93.88 | 93.53 | 84.0 | 95.0 | 150.412 | 59.0 | 0.00306 |
| 2997000 | 93.79 | 93.55 | 62.0 | 95.0 | 158.512 | 67.0 | 0.00308 |
| 2998000 | 93.11 | 93.55 | 72.0 | 95.0 | 143.633 | 53.0 | 0.00308 |
| 2999000 | 93.91 | 93.57 | 80.0 | 95.0 | 154.418 | 63.0 | 0.00306 |
| 3000000 | 93.91 | 93.58 | 82.0 | 95.0 | 163.758 | 72.0 | 0.00303 |
