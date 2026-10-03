# b38b-iqnlocal-seed2

step **2,000,000** · 2000 evals · trailing **93.94** · peak **94.15** @1,755,000 · sef **0.1** · best30 **67.3** @1,759,000

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
| seed | 2 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b38b-iqnlocal-seed2](b38b-iqnlocal-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.67 | 0.67 | 0.0 | 3.0 | 0.118 | 0.0 | 0.4 |
| 2000 | 0.51 | 0.59 | 0.0 | 3.0 | -0.043 | 0.0 | 0.4 |
| 3000 | 1.0 | 0.73 | 0.0 | 5.0 | 0.445 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1989000 | 93.58 | 93.96 | 60.0 | 95.0 | 147.921 | 56.0 | 0.00295 |
| 1990000 | 93.78 | 93.95 | 84.0 | 95.0 | 148.182 | 56.0 | 0.00293 |
| 1991000 | 93.87 | 93.94 | 82.0 | 95.0 | 149.314 | 57.0 | 0.00295 |
| 1992000 | 94.02 | 93.95 | 86.0 | 95.0 | 153.277 | 61.0 | 0.00298 |
| 1993000 | 93.88 | 93.94 | 83.0 | 95.0 | 149.004 | 57.0 | 0.00302 |
| 1994000 | 94.08 | 93.94 | 89.0 | 95.0 | 157.285 | 65.0 | 0.00301 |
| 1995000 | 94.18 | 93.95 | 90.0 | 95.0 | 155.521 | 63.0 | 0.00303 |
| 1996000 | 94.02 | 93.95 | 80.0 | 95.0 | 150.327 | 58.0 | 0.00303 |
| 1997000 | 94.11 | 93.95 | 91.0 | 95.0 | 153.336 | 61.0 | 0.00303 |
| 1998000 | 94.13 | 93.94 | 88.0 | 95.0 | 158.472 | 66.0 | 0.00304 |
| 1999000 | 94.0 | 93.95 | 84.0 | 95.0 | 154.293 | 62.0 | 0.00303 |
| 2000000 | 93.85 | 93.94 | 84.0 | 95.0 | 148.049 | 56.0 | 0.00304 |
