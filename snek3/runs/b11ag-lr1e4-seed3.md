# b11ag-lr1e4-seed3

step **32,210,944** · 1960 evals · trailing **94.48** · peak **94.61** @31,916,032 · sef **80.9** · best30 **98.6** @22,036,480

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
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.0001 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b11ag-lr1e4-seed3](b11ag-lr1e4-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.28 | 0.28 | 0.0 | 1.0 | -0.22 | 0.0 |  |
| 32768 | 1.58 | 0.93 | 1.0 | 7.0 | -0.405 | 0.0 |  |
| 49152 | 7.68 | 3.18 | 0.0 | 23.0 | 3.175 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 31932416 | 94.85 | 94.57 | 80.0 | 95.0 | 192.855 | 99.0 |  |
| 31948800 | 93.33 | 94.58 | 12.0 | 95.0 | 185.365 | 93.0 |  |
| 31965184 | 94.2 | 94.6 | 48.0 | 95.0 | 191.21 | 98.0 |  |
| 31981568 | 94.23 | 94.52 | 55.0 | 95.0 | 191.24 | 98.0 |  |
| 31997952 | 94.49 | 94.51 | 55.0 | 95.0 | 191.5 | 98.0 |  |
| 32030720 | 94.98 | 94.55 | 93.0 | 95.0 | 192.985 | 99.0 |  |
| 32096256 | 93.73 | 94.51 | 16.0 | 95.0 | 188.75 | 96.0 |  |
| 32112640 | 94.37 | 94.53 | 61.0 | 95.0 | 190.385 | 97.0 |  |
| 32129024 | 94.62 | 94.58 | 63.0 | 95.0 | 190.635 | 97.0 |  |
| 32145408 | 92.88 | 94.53 | 8.0 | 95.0 | 186.905 | 95.0 |  |
| 32161792 | 94.92 | 94.56 | 90.0 | 95.0 | 191.93 | 98.0 |  |
| 32210944 | 93.46 | 94.48 | 34.0 | 95.0 | 187.44 | 95.0 |  |
