# b9by-lam99-seed3

step **50,003,968** · 3052 evals · trailing **94.36** · peak **94.7** @41,435,136 · sef **88.8** · best30 **98.4** @46,333,952

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
| ppo_learning_rate | 0.0003 |
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

![b9by-lam99-seed3](b9by-lam99-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.03 | 0.03 | 0.0 | 1.0 | -4.025 | 0.0 |  |
| 32768 | 3.58 | 1.8 | 1.0 | 19.0 | 2.495 | 0.0 |  |
| 49152 | 13.12 | 9.35 | 0.0 | 43.0 | 9.2 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.75 | 94.31 | 73.0 | 95.0 | 191.76 | 98.0 |  |
| 49840128 | 93.67 | 94.34 | 26.0 | 95.0 | 187.695 | 95.0 |  |
| 49856512 | 94.31 | 94.37 | 57.0 | 95.0 | 188.335 | 95.0 |  |
| 49872896 | 93.34 | 94.3 | 8.0 | 95.0 | 185.375 | 93.0 |  |
| 49889280 | 93.09 | 94.27 | 10.0 | 95.0 | 186.12 | 94.0 |  |
| 49905664 | 94.61 | 94.27 | 67.0 | 95.0 | 191.62 | 98.0 |  |
| 49922048 | 94.4 | 94.35 | 80.0 | 95.0 | 187.43 | 94.0 |  |
| 49938432 | 94.96 | 94.32 | 91.0 | 95.0 | 192.965 | 99.0 |  |
| 49954816 | 95.0 | 94.3 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49971200 | 94.23 | 94.36 | 18.0 | 95.0 | 192.235 | 99.0 |  |
| 49987584 | 94.32 | 94.36 | 71.0 | 95.0 | 188.345 | 95.0 |  |
| 50003968 | 94.8 | 94.36 | 77.0 | 95.0 | 191.81 | 98.0 |  |
