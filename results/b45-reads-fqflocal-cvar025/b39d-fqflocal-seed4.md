# b39d-fqflocal-seed4

step **2,000,000** · 2000 evals · trailing **92.76** · peak **94.27** @1,242,000 · sef **3.2** · best30 **79.1** @1,606,000

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
| 1989000 | 92.8 | 92.94 | 73.0 | 95.0 | 164.373 | 74.0 | 0.00226 |
| 1990000 | 92.98 | 92.94 | 74.0 | 95.0 | 166.63 | 76.0 | 0.00226 |
| 1991000 | 91.76 | 92.92 | 66.0 | 95.0 | 158.08 | 69.0 | 0.00227 |
| 1992000 | 92.56 | 92.91 | 69.0 | 95.0 | 163.046 | 73.0 | 0.00227 |
| 1993000 | 93.38 | 92.92 | 76.0 | 95.0 | 170.181 | 79.0 | 0.00226 |
| 1994000 | 93.24 | 92.94 | 66.0 | 95.0 | 163.669 | 73.0 | 0.00226 |
| 1995000 | 92.38 | 92.92 | 64.0 | 95.0 | 162.828 | 73.0 | 0.00226 |
| 1996000 | 92.45 | 92.91 | 69.0 | 95.0 | 163.027 | 73.0 | 0.00226 |
| 1997000 | 91.56 | 92.88 | 64.0 | 95.0 | 150.635 | 62.0 | 0.00227 |
| 1998000 | 91.52 | 92.84 | 60.0 | 95.0 | 156.753 | 68.0 | 0.00227 |
| 1999000 | 91.49 | 92.81 | 69.0 | 95.0 | 154.697 | 66.0 | 0.00228 |
| 2000000 | 90.98 | 92.76 | 64.0 | 95.0 | 150.105 | 62.0 | 0.00229 |
