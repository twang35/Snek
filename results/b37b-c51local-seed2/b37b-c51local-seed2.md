# b37b-c51local-seed2

step **3,000,000** · 3000 evals · trailing **94.35** · peak **94.58** @2,257,000 · sef **85.9** · best30 **93.6** @2,943,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | c51 |
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
| seed | 2 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b37b-c51local-seed2](b37b-c51local-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.79 | 0.79 | 0.0 | 4.0 | 0.235 | 0.0 | 0.4 |
| 2000 | 0.64 | 0.72 | 0.0 | 3.0 | 0.083 | 0.0 | 0.4 |
| 3000 | 2.66 | 1.36 | 0.0 | 62.0 | 2.0 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 94.87 | 94.44 | 91.0 | 95.0 | 189.462 | 96.0 | 0.002 |
| 2990000 | 94.12 | 94.43 | 29.0 | 95.0 | 184.733 | 92.0 | 0.002 |
| 2991000 | 94.33 | 94.42 | 62.0 | 95.0 | 181.885 | 89.0 | 0.002 |
| 2992000 | 94.37 | 94.4 | 80.0 | 95.0 | 176.959 | 84.0 | 0.002 |
| 2993000 | 93.97 | 94.39 | 25.0 | 95.0 | 183.559 | 91.0 | 0.002 |
| 2994000 | 93.76 | 94.36 | 6.0 | 95.0 | 182.31 | 90.0 | 0.002 |
| 2995000 | 94.52 | 94.35 | 87.0 | 95.0 | 181.008 | 88.0 | 0.002 |
| 2996000 | 93.26 | 94.3 | 20.0 | 95.0 | 180.827 | 89.0 | 0.002 |
| 2997000 | 94.7 | 94.3 | 84.0 | 95.0 | 187.301 | 94.0 | 0.002 |
| 2998000 | 94.46 | 94.33 | 85.0 | 95.0 | 174.989 | 82.0 | 0.002 |
| 2999000 | 94.68 | 94.33 | 88.0 | 95.0 | 184.2 | 91.0 | 0.002 |
| 3000000 | 94.75 | 94.35 | 86.0 | 95.0 | 187.327 | 94.0 | 0.002 |
