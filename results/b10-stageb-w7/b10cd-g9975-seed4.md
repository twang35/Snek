# b10cd-g9975-seed4

step **50,003,968** · 3052 evals · trailing **94.56** · peak **94.67** @36,683,776 · sef **92.6** · best30 **98.7** @36,864,000

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.9975 |
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
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 44.5 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b10cd-g9975-seed4](b10cd-g9975-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.27 | 0.27 | 0.0 | 2.0 | -0.635 | 0.0 |  |
| 32768 | 18.41 | 9.34 | 2.0 | 33.0 | 13.455 | 0.0 |  |
| 49152 | 23.3 | 13.99 | 3.0 | 45.0 | 18.3 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.5 | 94.52 | 81.0 | 95.0 | 188.525 | 95.0 |  |
| 49840128 | 94.37 | 94.49 | 73.0 | 95.0 | 186.405 | 93.0 |  |
| 49856512 | 94.71 | 94.51 | 85.0 | 95.0 | 189.73 | 96.0 |  |
| 49872896 | 93.94 | 94.51 | 80.0 | 95.0 | 181.995 | 89.0 |  |
| 49889280 | 94.41 | 94.5 | 79.0 | 95.0 | 187.395 | 94.0 |  |
| 49905664 | 94.23 | 94.52 | 51.0 | 95.0 | 188.21 | 95.0 |  |
| 49922048 | 94.21 | 94.49 | 58.0 | 95.0 | 190.225 | 97.0 |  |
| 49938432 | 94.96 | 94.5 | 91.0 | 95.0 | 192.965 | 99.0 |  |
| 49954816 | 94.54 | 94.52 | 71.0 | 95.0 | 190.555 | 97.0 |  |
| 49971200 | 94.95 | 94.51 | 90.0 | 95.0 | 192.955 | 99.0 |  |
| 49987584 | 94.7 | 94.53 | 70.0 | 95.0 | 191.71 | 98.0 |  |
| 50003968 | 94.56 | 94.56 | 83.0 | 95.0 | 188.585 | 95.0 |  |
