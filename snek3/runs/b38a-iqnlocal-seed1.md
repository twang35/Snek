# b38a-iqnlocal-seed1

step **929,000** · 929 evals · trailing **93.76** · peak **93.78** @915,000 · sef **0.1** · best30 **64.7** @725,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | iqn |
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
| dist_tau_prime_samples | 8 |
| dist_tau_samples | 8 |
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
| max_steps | 2000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.002 |
| munchausen_alpha | 0.0 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 1 |
| priority_exponent | 0.6 |
| replay_buffer_max_length | 100000 |
| replay_ratio | 1.0 |
| seed | 1 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b38a-iqnlocal-seed1](b38a-iqnlocal-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.63 | 0.63 | 0.0 | 5.0 | 0.076 | 0.0 | 0.4 |
| 2000 | 0.54 | 0.58 | 0.0 | 3.0 | -0.013 | 0.0 | 0.4 |
| 3000 | 0.76 | 0.64 | 0.0 | 4.0 | 0.206 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 918000 | 93.1 | 93.74 | 40.0 | 95.0 | 148.556 | 57.0 | 0.00293 |
| 919000 | 93.85 | 93.74 | 86.0 | 95.0 | 157.352 | 65.0 | 0.00291 |
| 920000 | 93.71 | 93.74 | 80.0 | 95.0 | 158.173 | 66.0 | 0.00293 |
| 921000 | 93.56 | 93.72 | 84.0 | 95.0 | 146.079 | 54.0 | 0.00292 |
| 922000 | 93.72 | 93.72 | 82.0 | 95.0 | 159.221 | 67.0 | 0.00291 |
| 923000 | 92.98 | 93.73 | 76.0 | 95.0 | 145.508 | 54.0 | 0.00294 |
| 924000 | 94.07 | 93.73 | 88.0 | 95.0 | 158.532 | 66.0 | 0.00295 |
| 925000 | 93.84 | 93.73 | 86.0 | 95.0 | 152.278 | 60.0 | 0.00295 |
| 926000 | 93.86 | 93.75 | 68.0 | 95.0 | 162.355 | 70.0 | 0.00294 |
| 927000 | 93.72 | 93.75 | 86.0 | 95.0 | 154.24 | 62.0 | 0.00296 |
| 928000 | 93.85 | 93.74 | 80.0 | 95.0 | 157.263 | 65.0 | 0.00293 |
| 929000 | 93.99 | 93.76 | 84.0 | 95.0 | 161.462 | 69.0 | 0.00293 |
