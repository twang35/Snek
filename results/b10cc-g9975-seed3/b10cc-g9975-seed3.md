# b10cc-g9975-seed3

step **50,003,968** · 3052 evals · trailing **94.36** · peak **94.62** @31,358,976 · sef **93.7** · best30 **98.4** @40,402,944

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
| seed | 3 |
| torch_threads | 1 |

![b10cc-g9975-seed3](b10cc-g9975-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.05 | 0.05 | 0.0 | 1.0 | -3.915 | 0.0 |  |
| 32768 | 2.35 | 1.2 | 0.0 | 10.0 | 1.805 | 0.0 |  |
| 49152 | 19.37 | 7.26 | 0.0 | 40.0 | 14.685 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.57 | 94.38 | 69.0 | 95.0 | 190.585 | 97.0 |  |
| 49840128 | 93.87 | 94.38 | 59.0 | 95.0 | 188.89 | 96.0 |  |
| 49856512 | 94.13 | 94.37 | 72.0 | 95.0 | 188.155 | 95.0 |  |
| 49872896 | 92.91 | 94.35 | 8.0 | 95.0 | 186.935 | 95.0 |  |
| 49889280 | 94.48 | 94.38 | 83.0 | 95.0 | 187.51 | 94.0 |  |
| 49905664 | 94.64 | 94.41 | 70.0 | 95.0 | 191.65 | 98.0 |  |
| 49922048 | 94.62 | 94.36 | 72.0 | 95.0 | 190.635 | 97.0 |  |
| 49938432 | 93.85 | 94.35 | 6.0 | 95.0 | 188.87 | 96.0 |  |
| 49954816 | 94.73 | 94.36 | 68.0 | 95.0 | 192.735 | 99.0 |  |
| 49971200 | 94.93 | 94.36 | 88.0 | 95.0 | 192.935 | 99.0 |  |
| 49987584 | 94.37 | 94.35 | 63.0 | 95.0 | 189.39 | 96.0 |  |
| 50003968 | 94.56 | 94.36 | 71.0 | 95.0 | 190.575 | 97.0 |  |
