# b7ah-fc200x100-seed4

step **50,003,968** · 3052 evals · trailing **94.09** · peak **94.47** @21,250,048 · sef **95.4** · best30 **98.2** @21,381,120

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
| fc_layers | (200, 100) |
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
| seed | 4 |
| torch_threads | 1 |

![b7ah-fc200x100-seed4](b7ah-fc200x100-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 19.4 | 19.4 | 0.0 | 34.0 | 14.49 | 0.0 |  |
| 32768 | 28.73 | 25.76 | 10.0 | 56.0 | 23.73 | 0.0 |  |
| 49152 | 27.2 | 23.3 | 10.0 | 49.0 | 22.2 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.38 | 94.11 | 36.0 | 95.0 | 191.39 | 98.0 |  |
| 49840128 | 91.51 | 93.94 | 13.0 | 95.0 | 184.495 | 94.0 |  |
| 49856512 | 94.15 | 94.04 | 67.0 | 95.0 | 186.185 | 93.0 |  |
| 49872896 | 94.05 | 93.9 | 34.0 | 95.0 | 189.07 | 96.0 |  |
| 49889280 | 94.82 | 93.99 | 77.0 | 95.0 | 192.825 | 99.0 |  |
| 49905664 | 94.92 | 93.98 | 87.0 | 95.0 | 192.925 | 99.0 |  |
| 49922048 | 93.62 | 94.07 | 8.0 | 95.0 | 189.635 | 97.0 |  |
| 49938432 | 94.18 | 94.09 | 58.0 | 95.0 | 189.2 | 96.0 |  |
| 49954816 | 94.39 | 94.12 | 63.0 | 95.0 | 191.4 | 98.0 |  |
| 49971200 | 93.47 | 94.09 | 5.0 | 95.0 | 190.48 | 98.0 |  |
| 49987584 | 94.39 | 94.12 | 34.0 | 95.0 | 192.395 | 99.0 |  |
| 50003968 | 93.98 | 94.09 | 26.0 | 95.0 | 190.99 | 98.0 |  |
