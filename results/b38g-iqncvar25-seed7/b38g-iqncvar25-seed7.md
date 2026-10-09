# b38g-iqncvar25-seed7

step **2,000,000** · 2000 evals · trailing **92.37** · peak **93.71** @1,392,000 · sef **0.0** · best30 **55.4** @1,395,000

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
| dist_risk_alpha | 0.25 |
| dist_risk_train | True |
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
| seed | 7 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b38g-iqncvar25-seed7](b38g-iqncvar25-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.56 | 0.56 | 0.0 | 3.0 | 0.006 | 0.0 | 0.4 |
| 2000 | 0.57 | 0.56 | 0.0 | 4.0 | 0.016 | 0.0 | 0.4 |
| 3000 | 0.65 | 0.59 | 0.0 | 4.0 | 0.097 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1989000 | 92.78 | 91.81 | 29.0 | 95.0 | 130.334 | 41.0 | 0.00521 |
| 1990000 | 92.65 | 91.86 | 54.0 | 95.0 | 120.69 | 32.0 | 0.00515 |
| 1991000 | 92.56 | 91.88 | 50.0 | 95.0 | 122.823 | 34.0 | 0.00512 |
| 1992000 | 92.83 | 91.94 | 48.0 | 95.0 | 138.708 | 49.0 | 0.00517 |
| 1993000 | 91.71 | 91.95 | 35.0 | 95.0 | 120.913 | 33.0 | 0.00521 |
| 1994000 | 92.96 | 92.06 | 31.0 | 95.0 | 132.545 | 43.0 | 0.00514 |
| 1995000 | 93.4 | 92.18 | 88.0 | 95.0 | 130.803 | 41.0 | 0.00514 |
| 1996000 | 92.81 | 92.23 | 60.0 | 95.0 | 126.107 | 37.0 | 0.00511 |
| 1997000 | 92.43 | 92.29 | 25.0 | 95.0 | 120.471 | 32.0 | 0.00509 |
| 1998000 | 93.01 | 92.33 | 81.0 | 95.0 | 133.602 | 44.0 | 0.00507 |
| 1999000 | 92.18 | 92.38 | 31.0 | 95.0 | 118.161 | 30.0 | 0.00508 |
| 2000000 | 92.74 | 92.37 | 77.0 | 95.0 | 122.768 | 34.0 | 0.00507 |
