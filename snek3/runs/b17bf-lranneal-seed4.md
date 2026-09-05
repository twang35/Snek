# b17bf-lranneal-seed4

step **7,405,568** · 450 evals · trailing **87.77** · peak **93.74** @2,031,616 · sef **64.7** · best30 **92.5** @6,324,224

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
| seed | 4 |
| torch_threads | 1 |

![b17bf-lranneal-seed4](b17bf-lranneal-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.2 | 0.2 | 0.0 | 1.0 | -0.483 | 0.0 |  |
| 32768 | 16.47 | 8.33 | 1.0 | 32.0 | 12.098 | 0.0 |  |
| 49152 | 22.79 | 13.15 | 5.0 | 45.0 | 17.755 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 7208960 | 90.86 | 87.09 | 37.0 | 95.0 | 178.566 | 89.0 |  |
| 7225344 | 89.68 | 87.04 | 16.0 | 95.0 | 173.386 | 85.0 |  |
| 7241728 | 84.81 | 86.72 | 27.0 | 95.0 | 162.558 | 79.0 |  |
| 7258112 | 89.83 | 86.83 | 37.0 | 95.0 | 173.558 | 85.0 |  |
| 7274496 | 89.04 | 87.22 | 46.0 | 95.0 | 166.774 | 79.0 |  |
| 7290880 | 92.42 | 88.06 | 56.0 | 95.0 | 182.111 | 91.0 |  |
| 7307264 | 91.91 | 88.78 | 43.0 | 95.0 | 182.579 | 92.0 |  |
| 7323648 | 90.6 | 88.45 | 29.0 | 95.0 | 178.31 | 89.0 |  |
| 7340032 | 91.23 | 88.96 | 44.0 | 95.0 | 175.941 | 86.0 |  |
| 7372800 | 89.35 | 87.46 | 28.0 | 95.0 | 172.069 | 84.0 |  |
| 7389184 | 86.92 | 87.27 | 28.0 | 95.0 | 165.642 | 80.0 |  |
| 7405568 | 91.03 | 87.77 | 43.0 | 95.0 | 179.748 | 90.0 |  |
