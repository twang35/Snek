# b10ay-g93-seed1

step **50,003,968** · 3052 evals · trailing **92.91** · peak **93.99** @16,826,368 · sef **22.5** · best30 **87.7** @17,088,512

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.93 |
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
| ppo_horizon | 11.3 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

![b10ay-g93-seed1](b10ay-g93-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 3.93 | 3.93 | 0.0 | 15.0 | 3.43 | 0.0 |  |
| 32768 | 21.41 | 25.08 | 1.0 | 78.0 | 19.65 | 0.0 |  |
| 49152 | 51.66 | 43.46 | 11.0 | 84.0 | 48.055 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.48 | 92.74 | 70.0 | 95.0 | 160.64 | 68.0 |  |
| 49840128 | 93.66 | 92.84 | 69.0 | 95.0 | 169.775 | 77.0 |  |
| 49856512 | 93.78 | 92.8 | 76.0 | 95.0 | 175.865 | 83.0 |  |
| 49872896 | 92.36 | 92.78 | 22.0 | 95.0 | 165.445 | 74.0 |  |
| 49889280 | 92.79 | 92.9 | 73.0 | 95.0 | 164.925 | 73.0 |  |
| 49905664 | 93.71 | 92.86 | 64.0 | 95.0 | 172.81 | 80.0 |  |
| 49922048 | 91.19 | 92.73 | 9.0 | 95.0 | 157.265 | 67.0 |  |
| 49938432 | 91.66 | 92.66 | 16.0 | 95.0 | 163.705 | 73.0 |  |
| 49954816 | 93.11 | 92.68 | 68.0 | 95.0 | 162.26 | 70.0 |  |
| 49971200 | 92.35 | 92.83 | 27.0 | 95.0 | 162.45 | 71.0 |  |
| 49987584 | 92.96 | 92.92 | 23.0 | 95.0 | 169.03 | 77.0 |  |
| 50003968 | 92.78 | 92.91 | 11.0 | 95.0 | 173.825 | 82.0 |  |
