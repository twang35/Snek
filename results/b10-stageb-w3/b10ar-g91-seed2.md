# b10ar-g91-seed2

step **50,003,968** · 3052 evals · trailing **92.12** · peak **93.46** @18,022,400 · sef **2.5** · best30 **79.9** @49,889,280

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.91 |
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
| ppo_horizon | 9.2 |
| ppo_learning_rate | 0.0003 |
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

![b10ar-g91-seed2](b10ar-g91-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.99 | 2.99 | 0.0 | 7.0 | -0.795 | 0.0 |  |
| 32768 | 12.15 | 7.57 | 0.0 | 26.0 | 7.555 | 0.0 |  |
| 49152 | 17.16 | 10.77 | 1.0 | 41.0 | 13.78 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.18 | 92.4 | 25.0 | 95.0 | 169.25 | 77.0 |  |
| 49840128 | 93.01 | 92.39 | 20.0 | 95.0 | 174.055 | 82.0 |  |
| 49856512 | 93.29 | 92.13 | 25.0 | 95.0 | 166.375 | 74.0 |  |
| 49872896 | 91.19 | 92.12 | 16.0 | 95.0 | 168.165 | 78.0 |  |
| 49889280 | 93.06 | 92.11 | 16.0 | 95.0 | 179.08 | 87.0 |  |
| 49905664 | 91.42 | 92.06 | 24.0 | 95.0 | 171.38 | 81.0 |  |
| 49922048 | 91.42 | 91.97 | 10.0 | 95.0 | 165.5 | 75.0 |  |
| 49938432 | 93.44 | 91.96 | 58.0 | 95.0 | 164.58 | 72.0 |  |
| 49954816 | 93.66 | 92.21 | 78.0 | 95.0 | 173.755 | 81.0 |  |
| 49971200 | 94.3 | 92.13 | 84.0 | 95.0 | 175.39 | 82.0 |  |
| 49987584 | 93.56 | 92.14 | 51.0 | 95.0 | 160.72 | 68.0 |  |
| 50003968 | 94.24 | 92.12 | 84.0 | 95.0 | 176.325 | 83.0 |  |
