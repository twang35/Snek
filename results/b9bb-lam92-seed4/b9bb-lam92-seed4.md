# b9bb-lam92-seed4

step **50,003,968** · 3052 evals · trailing **93.95** · peak **94.65** @48,463,872 · sef **88.0** · best30 **97.6** @48,513,024

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
| ppo_gae_lambda | 0.92 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 11.2 |
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

![b9bb-lam92-seed4](b9bb-lam92-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.91 | 2.91 | 0.0 | 11.0 | 0.745 | 0.0 |  |
| 32768 | 7.74 | 24.9 | 0.0 | 34.0 | 6.025 | 0.0 |  |
| 49152 | 27.72 | 22.88 | 8.0 | 45.0 | 22.765 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.67 | 94.04 | 80.0 | 95.0 | 188.65 | 95.0 |  |
| 49840128 | 93.97 | 94.02 | 71.0 | 95.0 | 187.0 | 94.0 |  |
| 49856512 | 94.27 | 94.01 | 58.0 | 95.0 | 187.21 | 94.0 |  |
| 49872896 | 94.63 | 93.96 | 78.0 | 95.0 | 188.655 | 95.0 |  |
| 49889280 | 94.59 | 93.98 | 84.0 | 95.0 | 187.62 | 94.0 |  |
| 49905664 | 94.75 | 94.01 | 86.0 | 95.0 | 188.685 | 95.0 |  |
| 49922048 | 93.27 | 93.97 | 14.0 | 95.0 | 186.3 | 94.0 |  |
| 49938432 | 93.87 | 93.98 | 26.0 | 95.0 | 188.89 | 96.0 |  |
| 49954816 | 94.56 | 94.0 | 79.0 | 95.0 | 186.505 | 93.0 |  |
| 49971200 | 94.73 | 94.01 | 86.0 | 95.0 | 188.71 | 95.0 |  |
| 49987584 | 94.68 | 94.0 | 86.0 | 95.0 | 188.705 | 95.0 |  |
| 50003968 | 93.46 | 93.95 | 32.0 | 95.0 | 184.5 | 92.0 |  |
