# b10ax-g92-seed4

step **50,003,968** · 3052 evals · trailing **92.62** · peak **93.88** @17,383,424 · sef **12.9** · best30 **83.1** @34,553,856

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.92 |
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
| ppo_horizon | 10.2 |
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

![b10ax-g92-seed4](b10ax-g92-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.96 | 2.96 | 0.0 | 10.0 | 1.155 | 0.0 |  |
| 32768 | 1.61 | 2.29 | 0.0 | 14.0 | 1.11 | 0.0 |  |
| 49152 | 11.73 | 5.43 | 1.0 | 35.0 | 9.61 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.33 | 92.57 | 13.0 | 95.0 | 160.485 | 69.0 |  |
| 49840128 | 91.59 | 92.46 | 33.0 | 95.0 | 152.735 | 62.0 |  |
| 49856512 | 93.09 | 92.43 | 51.0 | 95.0 | 154.28 | 62.0 |  |
| 49872896 | 92.46 | 92.41 | 16.0 | 95.0 | 155.64 | 64.0 |  |
| 49889280 | 92.85 | 92.48 | 65.0 | 95.0 | 158.02 | 66.0 |  |
| 49905664 | 92.75 | 92.45 | 49.0 | 95.0 | 164.885 | 73.0 |  |
| 49922048 | 91.64 | 92.46 | 8.0 | 95.0 | 162.78 | 72.0 |  |
| 49938432 | 93.3 | 92.44 | 18.0 | 95.0 | 176.38 | 84.0 |  |
| 49954816 | 94.26 | 92.44 | 81.0 | 95.0 | 176.345 | 83.0 |  |
| 49971200 | 93.93 | 92.49 | 83.0 | 95.0 | 175.02 | 82.0 |  |
| 49987584 | 93.37 | 92.56 | 55.0 | 95.0 | 169.485 | 77.0 |  |
| 50003968 | 93.65 | 92.62 | 49.0 | 95.0 | 169.765 | 77.0 |  |
