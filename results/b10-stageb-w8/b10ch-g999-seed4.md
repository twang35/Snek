# b10ch-g999-seed4

step **50,003,968** · 3052 evals · trailing **94.6** · peak **94.71** @39,714,816 · sef **93.3** · best30 **98.7** @35,389,440

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.999 |
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
| ppo_horizon | 47.7 |
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

![b10ch-g999-seed4](b10ch-g999-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.15 | 0.15 | 0.0 | 2.0 | -0.62 | 0.0 |  |
| 32768 | 13.41 | 6.78 | 1.0 | 26.0 | 8.545 | 0.0 |  |
| 49152 | 19.92 | 11.16 | 5.0 | 38.0 | 14.92 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.8 | 94.57 | 86.0 | 95.0 | 190.815 | 97.0 |  |
| 49840128 | 94.63 | 94.59 | 65.0 | 95.0 | 190.645 | 97.0 |  |
| 49856512 | 94.66 | 94.6 | 84.0 | 95.0 | 188.685 | 95.0 |  |
| 49872896 | 94.21 | 94.56 | 59.0 | 95.0 | 188.235 | 95.0 |  |
| 49889280 | 94.47 | 94.58 | 66.0 | 95.0 | 190.485 | 97.0 |  |
| 49905664 | 94.54 | 94.55 | 74.0 | 95.0 | 188.52 | 95.0 |  |
| 49922048 | 94.18 | 94.64 | 16.0 | 95.0 | 191.19 | 98.0 |  |
| 49938432 | 93.84 | 94.63 | 24.0 | 95.0 | 188.86 | 96.0 |  |
| 49954816 | 93.67 | 94.61 | 14.0 | 95.0 | 187.695 | 95.0 |  |
| 49971200 | 94.39 | 94.62 | 60.0 | 95.0 | 190.36 | 97.0 |  |
| 49987584 | 94.46 | 94.62 | 63.0 | 95.0 | 189.48 | 96.0 |  |
| 50003968 | 94.7 | 94.6 | 71.0 | 95.0 | 191.71 | 98.0 |  |
