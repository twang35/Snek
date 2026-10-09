# b37h-qrdqnlocal-seed8

step **3,000,000** · 3000 evals · trailing **93.57** · peak **94.37** @2,487,000 · sef **62.0** · best30 **92.4** @2,487,000

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
| seed | 8 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b37h-qrdqnlocal-seed8](b37h-qrdqnlocal-seed8.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.66 | 0.66 | 0.0 | 4.0 | 0.107 | 0.0 | 0.4 |
| 2000 | 0.98 | 0.82 | 0.0 | 4.0 | 0.426 | 0.0 | 0.4 |
| 3000 | 21.78 | 7.81 | 0.0 | 95.0 | 21.962 | 1.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 94.43 | 93.33 | 55.0 | 95.0 | 188.933 | 96.0 | 0.002 |
| 2990000 | 93.64 | 93.33 | 64.0 | 95.0 | 180.11 | 88.0 | 0.002 |
| 2991000 | 93.88 | 93.35 | 43.0 | 95.0 | 180.302 | 88.0 | 0.002 |
| 2992000 | 94.47 | 93.39 | 72.0 | 95.0 | 188.056 | 95.0 | 0.002 |
| 2993000 | 94.15 | 93.45 | 71.0 | 95.0 | 184.679 | 92.0 | 0.002 |
| 2994000 | 92.77 | 93.45 | 5.0 | 95.0 | 179.269 | 88.0 | 0.002 |
| 2995000 | 93.74 | 93.47 | 35.0 | 95.0 | 179.166 | 87.0 | 0.002 |
| 2996000 | 94.26 | 93.49 | 77.0 | 95.0 | 183.73 | 91.0 | 0.002 |
| 2997000 | 93.93 | 93.48 | 58.0 | 95.0 | 181.411 | 89.0 | 0.002 |
| 2998000 | 94.5 | 93.51 | 81.0 | 95.0 | 183.892 | 91.0 | 0.002 |
| 2999000 | 93.6 | 93.5 | 55.0 | 95.0 | 181.027 | 89.0 | 0.002 |
| 3000000 | 93.79 | 93.57 | 74.0 | 95.0 | 178.208 | 86.0 | 0.002 |
