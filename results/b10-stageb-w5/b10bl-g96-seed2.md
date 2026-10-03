# b10bl-g96-seed2

step **50,003,968** · 3052 evals · trailing **92.33** · peak **94.45** @35,815,424 · sef **68.8** · best30 **95.6** @35,815,424

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.96 |
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
| ppo_horizon | 16.9 |
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

![b10bl-g96-seed2](b10bl-g96-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.57 | 2.57 | 0.0 | 9.0 | -1.17 | 0.0 |  |
| 32768 | 10.68 | 6.62 | 3.0 | 24.0 | 5.905 | 0.0 |  |
| 49152 | 22.22 | 15.36 | 2.0 | 62.0 | 17.355 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.45 | 92.45 | 74.0 | 95.0 | 173.545 | 81.0 |  |
| 49840128 | 91.99 | 92.22 | 1.0 | 95.0 | 172.085 | 81.0 |  |
| 49856512 | 93.71 | 92.17 | 76.0 | 95.0 | 178.78 | 86.0 |  |
| 49872896 | 93.72 | 92.42 | 60.0 | 95.0 | 175.805 | 83.0 |  |
| 49889280 | 93.41 | 92.45 | 73.0 | 95.0 | 171.515 | 79.0 |  |
| 49905664 | 94.36 | 92.48 | 79.0 | 95.0 | 185.4 | 92.0 |  |
| 49922048 | 91.86 | 92.47 | 5.0 | 95.0 | 170.96 | 80.0 |  |
| 49938432 | 93.62 | 92.51 | 30.0 | 95.0 | 176.655 | 84.0 |  |
| 49954816 | 92.96 | 92.35 | 22.0 | 95.0 | 168.08 | 76.0 |  |
| 49971200 | 93.24 | 92.37 | 22.0 | 95.0 | 179.26 | 87.0 |  |
| 49987584 | 94.02 | 92.36 | 79.0 | 95.0 | 173.12 | 80.0 |  |
| 50003968 | 94.69 | 92.33 | 90.0 | 95.0 | 183.74 | 90.0 |  |
