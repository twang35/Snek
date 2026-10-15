# b38f-iqncvar25-seed6

step **2,000,000** · 2000 evals · trailing **43.97** · peak **93.45** @620,000 · sef **0.0** · best30 **59.6** @621,000

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
| seed | 6 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b38f-iqncvar25-seed6](b38f-iqncvar25-seed6.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.22 | 1.22 | 0.0 | 7.0 | 0.657 | 0.0 | 0.4 |
| 2000 | 0.73 | 0.97 | 0.0 | 5.0 | 0.176 | 0.0 | 0.4 |
| 3000 | 2.26 | 1.4 | 0.0 | 12.0 | 1.696 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1989000 | 45.01 | 44.81 | 27.0 | 60.0 | 39.905 | 0.0 | 0.0125 |
| 1990000 | 43.56 | 44.7 | 21.0 | 58.0 | 38.458 | 0.0 | 0.0125 |
| 1991000 | 44.01 | 44.58 | 26.0 | 59.0 | 38.911 | 0.0 | 0.0125 |
| 1992000 | 44.04 | 44.48 | 26.0 | 61.0 | 38.936 | 0.0 | 0.0125 |
| 1993000 | 44.22 | 44.36 | 19.0 | 60.0 | 39.114 | 0.0 | 0.0125 |
| 1994000 | 44.62 | 44.32 | 28.0 | 57.0 | 39.514 | 0.0 | 0.0125 |
| 1995000 | 42.82 | 44.18 | 13.0 | 59.0 | 37.722 | 0.0 | 0.0125 |
| 1996000 | 44.56 | 44.12 | 21.0 | 59.0 | 39.457 | 0.0 | 0.0125 |
| 1997000 | 45.11 | 44.07 | 26.0 | 59.0 | 39.999 | 0.0 | 0.0125 |
| 1998000 | 44.34 | 44.0 | 22.0 | 62.0 | 39.234 | 0.0 | 0.0125 |
| 1999000 | 44.43 | 43.95 | 25.0 | 57.0 | 39.322 | 0.0 | 0.0125 |
| 2000000 | 46.02 | 43.97 | 23.0 | 61.0 | 40.908 | 0.0 | 0.0125 |
