# b7bd-fc200x100x50-seed2

step **50,003,968** · 3052 evals · trailing **93.95** · peak **94.49** @14,942,208 · sef **95.3** · best30 **97.8** @48,627,712

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
| seed | 2 |
| torch_threads | 1 |

![b7bd-fc200x100x50-seed2](b7bd-fc200x100x50-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 19.41 | 19.41 | 4.0 | 37.0 | 14.41 | 0.0 |  |
| 32768 | 30.32 | 24.87 | 6.0 | 62.0 | 25.32 | 0.0 |  |
| 49152 | 37.11 | 32.91 | 13.0 | 79.0 | 32.155 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.65 | 93.78 | 82.0 | 95.0 | 190.665 | 97.0 |  |
| 49840128 | 94.6 | 93.89 | 70.0 | 95.0 | 189.575 | 96.0 |  |
| 49856512 | 93.7 | 93.89 | 26.0 | 95.0 | 189.625 | 97.0 |  |
| 49872896 | 94.3 | 93.91 | 79.0 | 95.0 | 186.335 | 93.0 |  |
| 49889280 | 94.45 | 93.89 | 64.0 | 95.0 | 190.42 | 97.0 |  |
| 49905664 | 94.79 | 93.88 | 82.0 | 95.0 | 190.805 | 97.0 |  |
| 49922048 | 94.84 | 93.85 | 82.0 | 95.0 | 191.805 | 98.0 |  |
| 49938432 | 93.34 | 93.86 | 3.0 | 95.0 | 187.365 | 95.0 |  |
| 49954816 | 93.35 | 93.95 | 7.0 | 95.0 | 186.38 | 94.0 |  |
| 49971200 | 93.81 | 93.95 | 21.0 | 95.0 | 187.835 | 95.0 |  |
| 49987584 | 94.34 | 93.98 | 75.0 | 95.0 | 187.325 | 94.0 |  |
| 50003968 | 94.25 | 93.95 | 35.0 | 95.0 | 190.175 | 97.0 |  |
