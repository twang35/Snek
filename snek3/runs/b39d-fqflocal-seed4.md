# b39d-fqflocal-seed4

step **779,000** · 779 evals · trailing **93.69** · peak **93.86** @756,000 · sef **0.0** · best30 **67.6** @754,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | fqf |
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
| dist_quantiles | 8 |
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

![b39d-fqflocal-seed4](b39d-fqflocal-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.8 | 0.8 | 0.0 | 4.0 | 0.245 | 0.0 | 0.4 |
| 2000 | 0.63 | 0.72 | 0.0 | 4.0 | 0.076 | 0.0 | 0.4 |
| 3000 | 1.05 | 0.83 | 0.0 | 4.0 | 0.499 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 768000 | 93.75 | 93.75 | 84.0 | 95.0 | 156.944 | 65.0 | 0.00279 |
| 769000 | 93.21 | 93.72 | 34.0 | 95.0 | 153.728 | 62.0 | 0.00277 |
| 770000 | 93.94 | 93.71 | 86.0 | 95.0 | 155.093 | 63.0 | 0.00276 |
| 771000 | 93.35 | 93.68 | 83.0 | 95.0 | 148.333 | 57.0 | 0.00277 |
| 772000 | 93.62 | 93.68 | 83.0 | 95.0 | 155.772 | 64.0 | 0.0028 |
| 773000 | 93.33 | 93.65 | 74.0 | 95.0 | 150.633 | 59.0 | 0.00284 |
| 774000 | 93.29 | 93.66 | 50.0 | 95.0 | 160.394 | 69.0 | 0.00284 |
| 775000 | 93.84 | 93.68 | 84.0 | 95.0 | 156.955 | 65.0 | 0.00287 |
| 776000 | 93.8 | 93.67 | 88.0 | 95.0 | 153.792 | 62.0 | 0.00285 |
| 777000 | 93.78 | 93.68 | 86.0 | 95.0 | 156.975 | 65.0 | 0.00285 |
| 778000 | 94.09 | 93.69 | 88.0 | 95.0 | 159.12 | 67.0 | 0.00287 |
| 779000 | 94.01 | 93.69 | 88.0 | 95.0 | 159.332 | 67.0 | 0.00287 |
