# b11ar-lr5e4-seed2

step **50,003,968** · 3052 evals · trailing **94.21** · peak **94.64** @47,792,128 · sef **91.5** · best30 **98.1** @22,380,544

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
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.0005 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b11ar-lr5e4-seed2](b11ar-lr5e4-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.57 | 1.57 | 0.0 | 6.0 | -1.54 | 0.0 |  |
| 32768 | 9.48 | 11.12 | 0.0 | 23.0 | 4.975 | 0.0 |  |
| 49152 | 22.31 | 11.94 | 8.0 | 50.0 | 17.625 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.78 | 94.01 | 73.0 | 95.0 | 192.785 | 99.0 |  |
| 49840128 | 94.21 | 94.08 | 63.0 | 95.0 | 189.23 | 96.0 |  |
| 49856512 | 94.18 | 94.04 | 42.0 | 95.0 | 190.195 | 97.0 |  |
| 49872896 | 94.25 | 94.06 | 65.0 | 95.0 | 190.265 | 97.0 |  |
| 49889280 | 94.95 | 94.22 | 90.0 | 95.0 | 192.955 | 99.0 |  |
| 49905664 | 93.94 | 94.09 | 61.0 | 95.0 | 188.96 | 96.0 |  |
| 49922048 | 94.49 | 94.05 | 66.0 | 95.0 | 190.46 | 97.0 |  |
| 49938432 | 94.68 | 94.21 | 86.0 | 95.0 | 188.66 | 95.0 |  |
| 49954816 | 94.77 | 94.15 | 86.0 | 95.0 | 189.745 | 96.0 |  |
| 49971200 | 95.0 | 94.07 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49987584 | 94.56 | 94.24 | 72.0 | 95.0 | 188.585 | 95.0 |  |
| 50003968 | 93.9 | 94.21 | 63.0 | 95.0 | 186.885 | 94.0 |  |
