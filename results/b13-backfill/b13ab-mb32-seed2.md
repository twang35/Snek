# b13ab-mb32-seed2

step **50,003,968** · 3052 evals · trailing **92.23** · peak **94.03** @38,748,160 · sef **81.2** · best30 **96.9** @17,481,728

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
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 32 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b13ab-mb32-seed2](b13ab-mb32-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.68 | 0.68 | 0.0 | 4.0 | -0.54 | 0.0 |  |
| 32768 | 13.94 | 7.31 | 5.0 | 24.0 | 8.985 | 0.0 |  |
| 49152 | 24.41 | 13.01 | 7.0 | 52.0 | 19.41 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 91.1 | 92.44 | 12.0 | 95.0 | 171.875 | 82.0 |  |
| 49840128 | 92.98 | 92.24 | 53.0 | 95.0 | 177.645 | 86.0 |  |
| 49856512 | 93.94 | 92.21 | 79.0 | 95.0 | 180.64 | 88.0 |  |
| 49872896 | 93.52 | 93.58 | 79.0 | 95.0 | 179.36 | 87.0 |  |
| 49889280 | 84.07 | 92.59 | 4.0 | 95.0 | 151.19 | 69.0 |  |
| 49905664 | 80.1 | 92.94 | 6.0 | 95.0 | 140.12 | 62.0 |  |
| 49922048 | 89.54 | 93.39 | 15.0 | 95.0 | 163.08 | 75.0 |  |
| 49938432 | 93.71 | 92.58 | 63.0 | 95.0 | 183.395 | 91.0 |  |
| 49954816 | 92.89 | 92.52 | 11.0 | 95.0 | 184.7 | 93.0 |  |
| 49971200 | 92.96 | 92.52 | 21.0 | 95.0 | 181.785 | 90.0 |  |
| 49987584 | 90.35 | 92.33 | 27.0 | 95.0 | 178.09 | 89.0 |  |
| 50003968 | 91.1 | 92.23 | 17.0 | 95.0 | 176.625 | 87.0 |  |
