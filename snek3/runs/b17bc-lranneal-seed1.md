# b17bc-lranneal-seed1

step **7,438,336** · 453 evals · trailing **92.99** · peak **93.93** @1,720,320 · sef **53.0** · best30 **94.3** @7,356,416

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
| seed | 1 |
| torch_threads | 1 |

![b17bc-lranneal-seed1](b17bc-lranneal-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 14.59 | 14.59 | 1.0 | 33.0 | 12.828 | 0.0 |  |
| 32768 | 45.69 | 32.83 | 11.0 | 81.0 | 40.625 | 0.0 |  |
| 49152 | 37.99 | 34.12 | 1.0 | 82.0 | 32.976 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 7241728 | 94.13 | 92.22 | 61.0 | 95.0 | 189.801 | 97.0 |  |
| 7258112 | 93.01 | 92.27 | 49.0 | 95.0 | 185.703 | 94.0 |  |
| 7274496 | 93.04 | 92.84 | 20.0 | 95.0 | 186.721 | 95.0 |  |
| 7290880 | 94.21 | 92.17 | 55.0 | 95.0 | 190.887 | 98.0 |  |
| 7307264 | 92.1 | 92.87 | 44.0 | 95.0 | 183.806 | 93.0 |  |
| 7323648 | 91.51 | 93.0 | 26.0 | 95.0 | 180.209 | 90.0 |  |
| 7340032 | 92.56 | 93.0 | 44.0 | 95.0 | 183.263 | 92.0 |  |
| 7356416 | 92.51 | 92.42 | 46.0 | 95.0 | 184.213 | 93.0 |  |
| 7389184 | 92.46 | 93.06 | 45.0 | 95.0 | 182.143 | 91.0 |  |
| 7405568 | 91.43 | 93.03 | 43.0 | 95.0 | 178.127 | 88.0 |  |
| 7421952 | 92.28 | 92.68 | 55.0 | 95.0 | 180.972 | 90.0 |  |
| 7438336 | 89.92 | 92.99 | 16.0 | 95.0 | 172.642 | 84.0 |  |
