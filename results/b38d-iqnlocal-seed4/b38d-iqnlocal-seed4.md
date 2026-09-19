# b38d-iqnlocal-seed4

step **1,749,000** · 1749 evals · trailing **93.58** · peak **93.72** @1,552,000 · sef **0.0** · best30 **60.1** @1,193,000

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
| 1738000 | 94.11 | 93.54 | 88.0 | 95.0 | 154.18 | 63.0 | 0.00361 |
| 1739000 | 93.59 | 93.56 | 82.0 | 95.0 | 145.686 | 55.0 | 0.0036 |
| 1740000 | 92.94 | 93.54 | 53.0 | 95.0 | 141.853 | 52.0 | 0.00359 |
| 1741000 | 93.57 | 93.55 | 72.0 | 95.0 | 155.11 | 64.0 | 0.00358 |
| 1742000 | 93.42 | 93.53 | 76.0 | 95.0 | 146.77 | 56.0 | 0.00358 |
| 1743000 | 93.65 | 93.53 | 80.0 | 95.0 | 144.729 | 54.0 | 0.00358 |
| 1744000 | 93.77 | 93.54 | 82.0 | 95.0 | 150.056 | 59.0 | 0.00357 |
| 1745000 | 93.41 | 93.55 | 84.0 | 95.0 | 142.665 | 52.0 | 0.00357 |
| 1746000 | 93.27 | 93.55 | 84.0 | 95.0 | 138.394 | 48.0 | 0.00355 |
| 1747000 | 93.62 | 93.54 | 78.0 | 95.0 | 153.15 | 62.0 | 0.00356 |
| 1748000 | 93.87 | 93.56 | 84.0 | 95.0 | 156.444 | 65.0 | 0.00357 |
| 1749000 | 93.78 | 93.58 | 86.0 | 95.0 | 156.423 | 65.0 | 0.00355 |
