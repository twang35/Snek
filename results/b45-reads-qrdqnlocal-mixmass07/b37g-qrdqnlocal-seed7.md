# b37g-qrdqnlocal-seed7

step **3,000,000** · 3000 evals · trailing **94.45** · peak **94.59** @2,970,000 · sef **64.2** · best30 **95.3** @2,968,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | qrdqn |
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
| max_steps | 3000000 |
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

![b37g-qrdqnlocal-seed7](b37g-qrdqnlocal-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.55 | 0.55 | 0.0 | 4.0 | -0.003 | 0.0 | 0.4 |
| 2000 | 1.38 | 0.96 | 0.0 | 6.0 | 0.824 | 0.0 | 0.4 |
| 3000 | 18.82 | 6.92 | 1.0 | 95.0 | 21.103 | 3.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 93.29 | 94.42 | 14.0 | 95.0 | 179.516 | 88.0 | 0.002 |
| 2990000 | 94.79 | 94.41 | 88.0 | 95.0 | 189.249 | 96.0 | 0.002 |
| 2991000 | 94.58 | 94.41 | 79.0 | 95.0 | 190.202 | 97.0 | 0.002 |
| 2992000 | 94.54 | 94.42 | 72.0 | 95.0 | 185.922 | 93.0 | 0.002 |
| 2993000 | 94.52 | 94.42 | 82.0 | 95.0 | 184.921 | 92.0 | 0.002 |
| 2994000 | 94.31 | 94.44 | 69.0 | 95.0 | 184.746 | 92.0 | 0.002 |
| 2995000 | 94.67 | 94.44 | 84.0 | 95.0 | 188.17 | 95.0 | 0.002 |
| 2996000 | 94.52 | 94.43 | 73.0 | 95.0 | 188.015 | 95.0 | 0.002 |
| 2997000 | 94.59 | 94.42 | 84.0 | 95.0 | 187.08 | 94.0 | 0.002 |
| 2998000 | 94.58 | 94.45 | 78.0 | 95.0 | 184.945 | 92.0 | 0.002 |
| 2999000 | 94.64 | 94.44 | 78.0 | 95.0 | 188.132 | 95.0 | 0.002 |
| 3000000 | 94.67 | 94.45 | 73.0 | 95.0 | 191.295 | 98.0 | 0.002 |
