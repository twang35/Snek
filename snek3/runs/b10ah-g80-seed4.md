# b10ah-g80-seed4

step **35,815,424** · 2186 evals · trailing **83.15** · peak **90.72** @8,962,048 · sef **0.0** · best30 **44.4** @29,605,888

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.8 |
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
| ppo_horizon | 4.6 |
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

![b10ah-g80-seed4](b10ah-g80-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.6 | 0.6 | 0.0 | 4.0 | 0.01 | 0.0 |  |
| 32768 | 5.13 | 2.86 | 0.0 | 17.0 | 4.45 | 0.0 |  |
| 49152 | 25.5 | 28.1 | 0.0 | 75.0 | 23.38 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 35635200 | 84.14 | 83.16 | 22.0 | 95.0 | 128.78 | 46.0 |  |
| 35651584 | 82.85 | 83.19 | 18.0 | 95.0 | 129.48 | 48.0 |  |
| 35667968 | 87.44 | 80.68 | 27.0 | 95.0 | 132.215 | 46.0 |  |
| 35684352 | 85.52 | 80.94 | 12.0 | 95.0 | 122.47 | 38.0 |  |
| 35700736 | 88.04 | 81.07 | 22.0 | 95.0 | 139.87 | 53.0 |  |
| 35717120 | 89.16 | 81.91 | 30.0 | 95.0 | 137.145 | 49.0 |  |
| 35733504 | 87.74 | 82.18 | 15.0 | 95.0 | 136.585 | 50.0 |  |
| 35749888 | 86.7 | 82.84 | 9.0 | 95.0 | 137.49 | 52.0 |  |
| 35766272 | 84.77 | 81.26 | 14.0 | 95.0 | 135.29 | 52.0 |  |
| 35782656 | 82.7 | 81.15 | 13.0 | 95.0 | 116.26 | 35.0 |  |
| 35799040 | 82.87 | 83.21 | 14.0 | 95.0 | 124.255 | 43.0 |  |
| 35815424 | 82.11 | 83.15 | 12.0 | 95.0 | 127.565 | 47.0 |  |
