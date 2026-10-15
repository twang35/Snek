# b9be-lam93-seed3

step **50,003,968** · 3052 evals · trailing **94.15** · peak **94.58** @48,103,424 · sef **91.2** · best30 **96.9** @15,220,736

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
| ppo_gae_lambda | 0.93 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 12.6 |
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

![b9be-lam93-seed3](b9be-lam93-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.08 | 0.08 | 0.0 | 1.0 | -0.42 | 0.0 |  |
| 32768 | 0.22 | 0.15 | 0.0 | 2.0 | -0.28 | 0.0 |  |
| 49152 | 13.18 | 14.51 | 2.0 | 41.0 | 10.7 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.18 | 94.16 | 24.0 | 95.0 | 182.23 | 90.0 |  |
| 49840128 | 94.1 | 94.24 | 80.0 | 95.0 | 182.155 | 89.0 |  |
| 49856512 | 94.27 | 94.22 | 78.0 | 95.0 | 186.305 | 93.0 |  |
| 49872896 | 93.04 | 94.15 | 18.0 | 95.0 | 185.075 | 93.0 |  |
| 49889280 | 95.0 | 94.19 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49905664 | 94.62 | 94.2 | 80.0 | 95.0 | 189.64 | 96.0 |  |
| 49922048 | 94.37 | 94.19 | 68.0 | 95.0 | 188.395 | 95.0 |  |
| 49938432 | 93.63 | 94.15 | 30.0 | 95.0 | 184.67 | 92.0 |  |
| 49954816 | 94.49 | 94.15 | 44.0 | 95.0 | 192.45 | 99.0 |  |
| 49971200 | 94.09 | 94.14 | 74.0 | 95.0 | 186.125 | 93.0 |  |
| 49987584 | 94.82 | 94.17 | 86.0 | 95.0 | 189.84 | 96.0 |  |
| 50003968 | 93.31 | 94.15 | 68.0 | 95.0 | 180.37 | 88.0 |  |
