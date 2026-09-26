# b38d-iqnlocal-seed4

step **2,000,000** · 2000 evals · trailing **93.71** · peak **93.77** @1,982,000 · sef **0.0** · best30 **60.1** @1,193,000

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
| seed | 4 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b38d-iqnlocal-seed4](b38d-iqnlocal-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.69 | 0.69 | 0.0 | 5.0 | 0.136 | 0.0 | 0.4 |
| 2000 | 0.66 | 0.68 | 0.0 | 4.0 | 0.107 | 0.0 | 0.4 |
| 3000 | 0.59 | 0.65 | 0.0 | 5.0 | 0.037 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1989000 | 93.72 | 93.73 | 84.0 | 95.0 | 146.836 | 56.0 | 0.0032 |
| 1990000 | 93.52 | 93.72 | 82.0 | 95.0 | 141.567 | 51.0 | 0.00321 |
| 1991000 | 93.84 | 93.73 | 89.0 | 95.0 | 143.723 | 53.0 | 0.00321 |
| 1992000 | 93.84 | 93.73 | 88.0 | 95.0 | 142.692 | 52.0 | 0.00323 |
| 1993000 | 93.66 | 93.73 | 82.0 | 95.0 | 145.82 | 55.0 | 0.00324 |
| 1994000 | 93.87 | 93.72 | 86.0 | 95.0 | 150.187 | 59.0 | 0.00326 |
| 1995000 | 93.53 | 93.7 | 72.0 | 95.0 | 144.516 | 54.0 | 0.00326 |
| 1996000 | 92.98 | 93.67 | 30.0 | 95.0 | 140.931 | 51.0 | 0.00328 |
| 1997000 | 93.66 | 93.69 | 86.0 | 95.0 | 143.769 | 53.0 | 0.00331 |
| 1998000 | 93.91 | 93.71 | 86.0 | 95.0 | 144.814 | 54.0 | 0.00334 |
| 1999000 | 93.84 | 93.72 | 88.0 | 95.0 | 144.742 | 54.0 | 0.00336 |
| 2000000 | 93.7 | 93.71 | 84.0 | 95.0 | 140.429 | 50.0 | 0.00336 |
