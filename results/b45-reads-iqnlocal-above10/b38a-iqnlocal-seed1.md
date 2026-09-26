# b38a-iqnlocal-seed1

step **2,000,000** · 2000 evals · trailing **93.48** · peak **93.98** @1,032,000 · sef **0.1** · best30 **67.5** @1,222,000

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
| 1989000 | 93.22 | 93.56 | 64.0 | 95.0 | 146.569 | 55.0 | 0.00359 |
| 1990000 | 93.38 | 93.54 | 75.0 | 95.0 | 148.668 | 57.0 | 0.0036 |
| 1991000 | 93.57 | 93.53 | 76.0 | 95.0 | 145.852 | 54.0 | 0.00361 |
| 1992000 | 93.37 | 93.53 | 62.0 | 95.0 | 150.722 | 59.0 | 0.00361 |
| 1993000 | 92.94 | 93.51 | 18.0 | 95.0 | 148.316 | 57.0 | 0.00359 |
| 1994000 | 93.12 | 93.5 | 72.0 | 95.0 | 135.427 | 44.0 | 0.00359 |
| 1995000 | 93.95 | 93.51 | 82.0 | 95.0 | 158.412 | 66.0 | 0.00359 |
| 1996000 | 93.68 | 93.5 | 84.0 | 95.0 | 151.116 | 59.0 | 0.00363 |
| 1997000 | 93.65 | 93.5 | 82.0 | 95.0 | 146.949 | 55.0 | 0.0036 |
| 1998000 | 93.49 | 93.48 | 84.0 | 95.0 | 143.727 | 52.0 | 0.0036 |
| 1999000 | 93.88 | 93.48 | 88.0 | 95.0 | 152.233 | 60.0 | 0.00361 |
| 2000000 | 93.73 | 93.48 | 84.0 | 95.0 | 144.08 | 52.0 | 0.00363 |
