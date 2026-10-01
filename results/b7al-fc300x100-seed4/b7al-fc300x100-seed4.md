# b7al-fc300x100-seed4

step **50,003,968** · 3052 evals · trailing **93.85** · peak **94.49** @27,901,952 · sef **95.0** · best30 **97.9** @28,147,712

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
| fc_layers | (300, 100) |
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

![b7al-fc300x100-seed4](b7al-fc300x100-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 30.18 | 30.18 | 6.0 | 56.0 | 26.575 | 0.0 |  |
| 32768 | 50.22 | 42.52 | 0.0 | 84.0 | 45.85 | 0.0 |  |
| 49152 | 44.66 | 38.96 | 13.0 | 78.0 | 39.885 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.67 | 93.84 | 16.0 | 95.0 | 176.655 | 85.0 |  |
| 49840128 | 94.41 | 93.86 | 81.0 | 95.0 | 188.435 | 95.0 |  |
| 49856512 | 93.64 | 93.85 | 71.0 | 95.0 | 179.66 | 87.0 |  |
| 49872896 | 94.89 | 93.87 | 90.0 | 95.0 | 190.86 | 97.0 |  |
| 49889280 | 94.48 | 93.86 | 71.0 | 95.0 | 189.41 | 96.0 |  |
| 49905664 | 93.57 | 93.83 | 63.0 | 95.0 | 182.44 | 90.0 |  |
| 49922048 | 93.74 | 93.84 | 61.0 | 95.0 | 187.585 | 95.0 |  |
| 49938432 | 94.37 | 93.85 | 72.0 | 95.0 | 185.365 | 92.0 |  |
| 49954816 | 93.5 | 93.83 | 4.0 | 95.0 | 186.44 | 94.0 |  |
| 49971200 | 94.47 | 93.91 | 74.0 | 95.0 | 189.445 | 96.0 |  |
| 49987584 | 94.57 | 93.82 | 59.0 | 95.0 | 190.54 | 97.0 |  |
| 50003968 | 93.21 | 93.85 | 14.0 | 95.0 | 184.16 | 92.0 |  |
