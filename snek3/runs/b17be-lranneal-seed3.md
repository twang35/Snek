# b17be-lranneal-seed3

step **7,323,648** · 444 evals · trailing **90.82** · peak **92.97** @1,753,088 · sef **39.0** · best30 **90.6** @7,274,496

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
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | 0.0 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b17be-lranneal-seed3](b17be-lranneal-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.03 | 0.03 | 0.0 | 1.0 | -3.548 | 0.0 |  |
| 32768 | 1.35 | 0.69 | 0.0 | 13.0 | 0.661 | 0.0 |  |
| 49152 | 19.79 | 15.39 | 0.0 | 43.0 | 14.953 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 7110656 | 93.11 | 86.72 | 60.0 | 95.0 | 184.752 | 93.0 |  |
| 7127040 | 94.29 | 87.55 | 62.0 | 95.0 | 188.923 | 96.0 |  |
| 7143424 | 94.48 | 87.22 | 74.0 | 95.0 | 188.126 | 95.0 |  |
| 7159808 | 92.62 | 88.27 | 37.0 | 95.0 | 183.236 | 92.0 |  |
| 7176192 | 92.29 | 89.03 | 50.0 | 95.0 | 180.918 | 90.0 |  |
| 7192576 | 94.72 | 89.85 | 67.0 | 95.0 | 192.321 | 99.0 |  |
| 7208960 | 95.0 | 89.52 | 95.0 | 95.0 | 193.586 | 100.0 |  |
| 7225344 | 94.76 | 91.38 | 76.0 | 95.0 | 190.362 | 97.0 |  |
| 7241728 | 94.23 | 91.0 | 28.0 | 95.0 | 190.862 | 98.0 |  |
| 7258112 | 90.84 | 89.89 | 41.0 | 95.0 | 180.542 | 91.0 |  |
| 7274496 | 93.69 | 91.82 | 51.0 | 95.0 | 187.354 | 95.0 |  |
| 7323648 | 89.26 | 90.82 | 35.0 | 95.0 | 171.985 | 84.0 |  |
