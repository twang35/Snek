# b7be-fc200x100x50-seed3

step **50,003,968** · 3052 evals · trailing **93.44** · peak **94.59** @31,195,136 · sef **94.4** · best30 **98.2** @31,178,752

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
| fc_layers | (200, 100, 50) |
| graph_eval_episodes | 100 |
| max_steps | 50003968 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b7be-fc200x100x50-seed3](b7be-fc200x100x50-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 16.42 | 16.42 | 1.0 | 31.0 | 11.51 | 0.0 |  |
| 32768 | 43.72 | 30.82 | 13.0 | 69.0 | 38.81 | 0.0 |  |
| 49152 | 32.63 | 31.27 | 15.0 | 63.0 | 27.63 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.24 | 93.71 | 56.0 | 95.0 | 191.25 | 98.0 |  |
| 49840128 | 93.24 | 93.63 | 44.0 | 95.0 | 186.18 | 94.0 |  |
| 49856512 | 93.6 | 93.67 | 37.0 | 95.0 | 189.57 | 97.0 |  |
| 49872896 | 94.65 | 93.57 | 70.0 | 95.0 | 190.62 | 97.0 |  |
| 49889280 | 93.52 | 93.63 | 50.0 | 95.0 | 186.505 | 94.0 |  |
| 49905664 | 93.2 | 93.59 | 3.0 | 95.0 | 187.225 | 95.0 |  |
| 49922048 | 92.31 | 93.48 | 10.0 | 95.0 | 182.355 | 91.0 |  |
| 49938432 | 93.97 | 93.57 | 24.0 | 95.0 | 188.99 | 96.0 |  |
| 49954816 | 93.31 | 93.48 | 1.0 | 95.0 | 186.34 | 94.0 |  |
| 49971200 | 92.94 | 93.45 | 20.0 | 95.0 | 183.98 | 92.0 |  |
| 49987584 | 93.95 | 93.4 | 63.0 | 95.0 | 184.945 | 92.0 |  |
| 50003968 | 93.89 | 93.44 | 60.0 | 95.0 | 186.92 | 94.0 |  |
