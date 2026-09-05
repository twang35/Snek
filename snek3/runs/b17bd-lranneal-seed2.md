# b17bd-lranneal-seed2

step **2,015,232** · 116 evals · trailing **92.99** · peak **93.56** @1,458,176 · sef **0.9** · best30 **70.1** @1,605,632

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 50003968 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | 0.0 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b17bd-lranneal-seed2](b17bd-lranneal-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.64 | 1.64 | 0.0 | 4.0 | -0.784 | 0.0 |  |
| 32768 | 16.76 | 18.14 | 4.0 | 41.0 | 11.959 | 0.0 |  |
| 49152 | 26.87 | 14.26 | 6.0 | 49.0 | 21.821 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1720320 | 90.85 | 92.87 | 80.0 | 95.0 | 109.535 | 20.0 |  |
| 1736704 | 91.96 | 93.41 | 80.0 | 95.0 | 126.563 | 36.0 |  |
| 1753088 | 92.55 | 93.27 | 84.0 | 95.0 | 139.142 | 48.0 |  |
| 1769472 | 91.96 | 93.17 | 84.0 | 95.0 | 128.552 | 38.0 |  |
| 1785856 | 92.59 | 92.96 | 80.0 | 95.0 | 138.224 | 47.0 |  |
| 1802240 | 93.64 | 93.31 | 88.0 | 95.0 | 149.261 | 57.0 |  |
| 1818624 | 93.58 | 93.3 | 80.0 | 95.0 | 155.183 | 63.0 |  |
| 1835008 | 93.11 | 93.21 | 80.0 | 95.0 | 152.676 | 61.0 |  |
| 1851392 | 92.98 | 93.06 | 10.0 | 95.0 | 156.518 | 65.0 |  |
| 1867776 | 93.79 | 92.95 | 86.0 | 95.0 | 157.365 | 65.0 |  |
| 1916928 | 93.74 | 92.96 | 84.0 | 95.0 | 157.258 | 65.0 |  |
| 2015232 | 94.4 | 92.99 | 88.0 | 95.0 | 173.984 | 81.0 |  |
