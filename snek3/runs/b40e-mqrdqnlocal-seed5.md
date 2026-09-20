# b40e-mqrdqnlocal-seed5

step **3,000,000** · 3000 evals · trailing **93.88** · peak **94.51** @2,498,000 · sef **49.3** · best30 **93.6** @2,501,000

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
| seed | 5 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b40e-mqrdqnlocal-seed5](b40e-mqrdqnlocal-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.48 | 0.48 | 0.0 | 4.0 | -0.073 | 0.0 | 0.4 |
| 2000 | 0.57 | 0.52 | 0.0 | 5.0 | 0.016 | 0.0 | 0.4 |
| 3000 | 1.56 | 0.87 | 0.0 | 11.0 | 1.006 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 93.72 | 93.88 | 77.0 | 95.0 | 177.133 | 85.0 | 0.002 |
| 2990000 | 94.32 | 93.9 | 78.0 | 95.0 | 184.82 | 92.0 | 0.002 |
| 2991000 | 93.99 | 93.92 | 79.0 | 95.0 | 177.382 | 85.0 | 0.002 |
| 2992000 | 93.87 | 93.91 | 48.0 | 95.0 | 182.322 | 90.0 | 0.002 |
| 2993000 | 93.56 | 93.91 | 75.0 | 95.0 | 172.954 | 81.0 | 0.002 |
| 2994000 | 92.93 | 93.86 | 47.0 | 95.0 | 174.338 | 83.0 | 0.002 |
| 2995000 | 94.44 | 93.88 | 83.0 | 95.0 | 178.747 | 86.0 | 0.002 |
| 2996000 | 94.27 | 93.88 | 83.0 | 95.0 | 180.706 | 88.0 | 0.002 |
| 2997000 | 93.41 | 93.85 | 56.0 | 95.0 | 172.78 | 81.0 | 0.002 |
| 2998000 | 94.47 | 93.86 | 80.0 | 95.0 | 180.804 | 88.0 | 0.002 |
| 2999000 | 93.82 | 93.88 | 78.0 | 95.0 | 173.055 | 81.0 | 0.002 |
| 3000000 | 94.25 | 93.88 | 81.0 | 95.0 | 177.491 | 85.0 | 0.002 |
