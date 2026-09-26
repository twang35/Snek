# b37a-c51local-seed1

step **3,000,000** · 3000 evals · trailing **94.28** · peak **94.46** @2,647,000 · sef **90.3** · best30 **94.1** @1,770,000

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
| seed | 1 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b37a-c51local-seed1](b37a-c51local-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.78 | 0.78 | 0.0 | 6.0 | 0.225 | 0.0 | 0.4 |
| 2000 | 0.6 | 0.69 | 0.0 | 4.0 | 0.045 | 0.0 | 0.4 |
| 3000 | 1.5 | 0.96 | 0.0 | 6.0 | 0.944 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 94.45 | 94.3 | 73.0 | 95.0 | 175.959 | 83.0 | 0.002 |
| 2990000 | 94.75 | 94.33 | 92.0 | 95.0 | 182.275 | 89.0 | 0.002 |
| 2991000 | 94.81 | 94.35 | 92.0 | 95.0 | 184.399 | 91.0 | 0.002 |
| 2992000 | 94.77 | 94.36 | 90.0 | 95.0 | 185.257 | 92.0 | 0.002 |
| 2993000 | 94.09 | 94.33 | 42.0 | 95.0 | 177.562 | 85.0 | 0.002 |
| 2994000 | 94.26 | 94.34 | 47.0 | 95.0 | 184.629 | 92.0 | 0.002 |
| 2995000 | 94.26 | 94.33 | 42.0 | 95.0 | 183.846 | 91.0 | 0.002 |
| 2996000 | 93.06 | 94.29 | 16.0 | 95.0 | 179.594 | 88.0 | 0.002 |
| 2997000 | 94.04 | 94.32 | 19.0 | 95.0 | 184.604 | 92.0 | 0.002 |
| 2998000 | 94.14 | 94.3 | 60.0 | 95.0 | 176.558 | 84.0 | 0.002 |
| 2999000 | 92.72 | 94.23 | 25.0 | 95.0 | 180.097 | 89.0 | 0.002 |
| 3000000 | 94.12 | 94.28 | 21.0 | 95.0 | 185.697 | 93.0 | 0.002 |
