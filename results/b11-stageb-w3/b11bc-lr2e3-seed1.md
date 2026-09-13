# b11bc-lr2e3-seed1

step **50,003,968** · 3052 evals · trailing **93.93** · peak **94.23** @48,349,184 · sef **85.7** · best30 **97.1** @28,491,776

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
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.002 |
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

![b11bc-lr2e3-seed1](b11bc-lr2e3-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 11.46 | 11.46 | 0.0 | 31.0 | 8.08 | 0.0 |  |
| 32768 | 42.14 | 29.55 | 8.0 | 71.0 | 37.32 | 0.0 |  |
| 49152 | 39.56 | 31.55 | 10.0 | 76.0 | 35.01 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.84 | 93.69 | 74.0 | 95.0 | 183.84 | 91.0 |  |
| 49840128 | 94.37 | 93.7 | 80.0 | 95.0 | 188.395 | 95.0 |  |
| 49856512 | 94.74 | 93.75 | 85.0 | 95.0 | 189.715 | 96.0 |  |
| 49872896 | 94.57 | 93.81 | 81.0 | 95.0 | 188.595 | 95.0 |  |
| 49889280 | 93.52 | 93.81 | 20.0 | 95.0 | 182.48 | 90.0 |  |
| 49905664 | 94.25 | 93.83 | 56.0 | 95.0 | 191.26 | 98.0 |  |
| 49922048 | 94.32 | 93.91 | 71.0 | 95.0 | 188.345 | 95.0 |  |
| 49938432 | 94.62 | 93.94 | 64.0 | 95.0 | 190.635 | 97.0 |  |
| 49954816 | 94.2 | 93.85 | 67.0 | 95.0 | 187.23 | 94.0 |  |
| 49971200 | 93.9 | 93.95 | 61.0 | 95.0 | 187.925 | 95.0 |  |
| 49987584 | 92.71 | 93.88 | 24.0 | 95.0 | 181.76 | 90.0 |  |
| 50003968 | 93.72 | 93.93 | 22.0 | 95.0 | 187.745 | 95.0 |  |
