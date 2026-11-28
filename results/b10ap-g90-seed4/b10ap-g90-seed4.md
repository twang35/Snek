# b10ap-g90-seed4

step **50,003,968** · 3052 evals · trailing **92.02** · peak **93.79** @22,069,248 · sef **6.1** · best30 **81.4** @41,205,760

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.9 |
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
| ppo_horizon | 8.5 |
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

![b10ap-g90-seed4](b10ap-g90-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.94 | 2.94 | 0.0 | 11.0 | 1.315 | 0.0 |  |
| 32768 | 7.56 | 6.59 | 0.0 | 38.0 | 6.745 | 0.0 |  |
| 49152 | 9.28 | 6.11 | 0.0 | 48.0 | 8.285 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.99 | 92.69 | 50.0 | 95.0 | 171.095 | 79.0 |  |
| 49840128 | 91.11 | 92.69 | 49.0 | 95.0 | 150.22 | 60.0 |  |
| 49856512 | 89.36 | 92.55 | 18.0 | 95.0 | 153.445 | 65.0 |  |
| 49872896 | 91.19 | 92.76 | 14.0 | 95.0 | 148.22 | 58.0 |  |
| 49889280 | 92.12 | 92.66 | 57.0 | 95.0 | 156.07 | 65.0 |  |
| 49905664 | 92.29 | 92.52 | 61.0 | 95.0 | 161.305 | 70.0 |  |
| 49922048 | 91.16 | 92.3 | 44.0 | 95.0 | 151.31 | 61.0 |  |
| 49938432 | 89.32 | 92.39 | 22.0 | 95.0 | 150.51 | 62.0 |  |
| 49954816 | 90.47 | 91.92 | 53.0 | 95.0 | 151.66 | 62.0 |  |
| 49971200 | 90.59 | 92.13 | 49.0 | 95.0 | 149.79 | 60.0 |  |
| 49987584 | 90.67 | 92.21 | 26.0 | 95.0 | 153.85 | 64.0 |  |
| 50003968 | 90.83 | 92.02 | 22.0 | 95.0 | 155.005 | 65.0 |  |
