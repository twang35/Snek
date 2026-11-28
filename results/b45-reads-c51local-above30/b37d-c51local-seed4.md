# b37d-c51local-seed4

step **3,000,000** · 3000 evals · trailing **94.21** · peak **94.53** @2,585,000 · sef **86.0** · best30 **94.9** @2,633,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | c51 |
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
| seed | 4 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b37d-c51local-seed4](b37d-c51local-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.49 | 0.49 | 0.0 | 5.0 | -0.108 | 0.0 | 0.4 |
| 2000 | 0.61 | 0.55 | 0.0 | 4.0 | 0.055 | 0.0 | 0.4 |
| 3000 | 0.66 | 0.59 | 0.0 | 7.0 | 0.104 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 94.81 | 94.15 | 92.0 | 95.0 | 183.166 | 90.0 | 0.002 |
| 2990000 | 94.84 | 94.17 | 93.0 | 95.0 | 185.366 | 92.0 | 0.002 |
| 2991000 | 92.89 | 94.15 | 13.0 | 95.0 | 185.411 | 94.0 | 0.002 |
| 2992000 | 93.93 | 94.13 | 15.0 | 95.0 | 178.419 | 86.0 | 0.002 |
| 2993000 | 94.52 | 94.15 | 57.0 | 95.0 | 187.994 | 95.0 | 0.002 |
| 2994000 | 92.78 | 94.09 | 21.0 | 95.0 | 181.199 | 90.0 | 0.002 |
| 2995000 | 94.08 | 94.13 | 45.0 | 95.0 | 181.585 | 89.0 | 0.002 |
| 2996000 | 92.99 | 94.13 | 12.0 | 95.0 | 184.418 | 93.0 | 0.002 |
| 2997000 | 94.7 | 94.16 | 89.0 | 95.0 | 182.205 | 89.0 | 0.002 |
| 2998000 | 94.21 | 94.17 | 26.0 | 95.0 | 186.716 | 94.0 | 0.002 |
| 2999000 | 94.07 | 94.21 | 27.0 | 95.0 | 181.593 | 89.0 | 0.002 |
| 3000000 | 93.95 | 94.21 | 39.0 | 95.0 | 184.427 | 92.0 | 0.002 |
