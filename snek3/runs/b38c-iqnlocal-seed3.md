# b38c-iqnlocal-seed3

step **929,000** · 929 evals · trailing **93.26** · peak **93.6** @337,000 · sef **0.0** · best30 **66.3** @337,000

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
| seed | 3 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b38c-iqnlocal-seed3](b38c-iqnlocal-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.68 | 0.68 | 0.0 | 5.0 | 0.126 | 0.0 | 0.4 |
| 2000 | 0.59 | 0.64 | 0.0 | 3.0 | 0.036 | 0.0 | 0.4 |
| 3000 | 0.71 | 0.66 | 0.0 | 4.0 | 0.157 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 918000 | 92.94 | 93.36 | 64.0 | 95.0 | 153.31 | 62.0 | 0.00317 |
| 919000 | 93.13 | 93.35 | 76.0 | 95.0 | 146.206 | 55.0 | 0.0032 |
| 920000 | 93.55 | 93.36 | 78.0 | 95.0 | 158.782 | 67.0 | 0.0032 |
| 921000 | 92.91 | 93.35 | 50.0 | 95.0 | 152.151 | 61.0 | 0.00323 |
| 922000 | 92.86 | 93.32 | 74.0 | 95.0 | 142.953 | 52.0 | 0.00322 |
| 923000 | 93.37 | 93.32 | 82.0 | 95.0 | 144.738 | 53.0 | 0.0032 |
| 924000 | 93.53 | 93.31 | 84.0 | 95.0 | 143.721 | 52.0 | 0.00323 |
| 925000 | 93.62 | 93.31 | 78.0 | 95.0 | 152.846 | 61.0 | 0.00325 |
| 926000 | 93.41 | 93.31 | 80.0 | 95.0 | 153.484 | 62.0 | 0.00328 |
| 927000 | 92.99 | 93.29 | 82.0 | 95.0 | 137.186 | 46.0 | 0.00327 |
| 928000 | 93.04 | 93.26 | 82.0 | 95.0 | 145.187 | 54.0 | 0.00327 |
| 929000 | 93.32 | 93.26 | 80.0 | 95.0 | 151.325 | 60.0 | 0.0033 |
