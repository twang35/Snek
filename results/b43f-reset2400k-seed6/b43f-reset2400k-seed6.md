# b43f-reset2400k-seed6

step **3,000,000** · 3000 evals · trailing **92.96** · peak **93.54** @1,178,000 · sef **1.1** · best30 **82.5** @2,999,000

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
| munchausen_alpha | 0.9 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 1 |
| priority_exponent | 0.6 |
| replay_buffer_max_length | 100000 |
| replay_ratio | 1.0 |
| reset_alpha | 0.5 |
| reset_interval | 2400000 |
| reset_stop_after | 10500000 |
| seed | 6 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b43f-reset2400k-seed6](b43f-reset2400k-seed6.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.54 | 0.54 | 0.0 | 3.0 | -0.014 | 0.0 | 0.4 |
| 2000 | 0.75 | 0.65 | 0.0 | 5.0 | 0.195 | 0.0 | 0.4 |
| 3000 | 0.61 | 0.63 | 0.0 | 3.0 | 0.057 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 92.72 | 92.56 | 53.0 | 95.0 | 175.983 | 85.0 | 0.00207 |
| 2990000 | 93.54 | 92.65 | 50.0 | 95.0 | 178.811 | 87.0 | 0.00204 |
| 2991000 | 93.77 | 92.71 | 69.0 | 95.0 | 177.01 | 85.0 | 0.00203 |
| 2992000 | 93.53 | 92.73 | 51.0 | 95.0 | 180.864 | 89.0 | 0.00201 |
| 2993000 | 93.01 | 92.75 | 36.0 | 95.0 | 178.335 | 87.0 | 0.002 |
| 2994000 | 94.17 | 92.82 | 70.0 | 95.0 | 178.362 | 86.0 | 0.002 |
| 2995000 | 92.82 | 92.8 | 43.0 | 95.0 | 170.885 | 80.0 | 0.002 |
| 2996000 | 93.41 | 92.83 | 70.0 | 95.0 | 172.664 | 81.0 | 0.002 |
| 2997000 | 93.12 | 92.84 | 45.0 | 95.0 | 174.289 | 83.0 | 0.002 |
| 2998000 | 93.63 | 92.89 | 54.0 | 95.0 | 176.915 | 85.0 | 0.002 |
| 2999000 | 94.04 | 93.0 | 78.0 | 95.0 | 177.182 | 85.0 | 0.002 |
| 3000000 | 92.41 | 92.96 | 42.0 | 95.0 | 167.473 | 77.0 | 0.002 |
