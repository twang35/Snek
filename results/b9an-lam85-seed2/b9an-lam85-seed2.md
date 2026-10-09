# b9an-lam85-seed2

step **50,003,968** · 3052 evals · trailing **92.96** · peak **94.33** @17,907,712 · sef **88.3** · best30 **96.2** @7,110,656

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
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.85 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 6.3 |
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

![b9an-lam85-seed2](b9an-lam85-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.21 | 2.21 | 1.0 | 5.0 | -1.26 | 0.0 |  |
| 32768 | 11.62 | 6.91 | 0.0 | 27.0 | 7.115 | 0.0 |  |
| 49152 | 27.36 | 13.73 | 10.0 | 47.0 | 22.36 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.09 | 92.85 | 6.0 | 95.0 | 177.115 | 86.0 |  |
| 49840128 | 91.48 | 92.86 | 5.0 | 95.0 | 173.52 | 83.0 |  |
| 49856512 | 92.17 | 93.02 | 22.0 | 95.0 | 179.23 | 88.0 |  |
| 49872896 | 92.98 | 92.87 | 44.0 | 95.0 | 179.0 | 87.0 |  |
| 49889280 | 92.46 | 92.84 | 30.0 | 95.0 | 175.405 | 84.0 |  |
| 49905664 | 92.09 | 92.81 | 3.0 | 95.0 | 181.05 | 90.0 |  |
| 49922048 | 93.15 | 92.83 | 6.0 | 95.0 | 187.13 | 95.0 |  |
| 49938432 | 93.68 | 92.89 | 38.0 | 95.0 | 186.62 | 94.0 |  |
| 49954816 | 94.32 | 92.85 | 73.0 | 95.0 | 188.345 | 95.0 |  |
| 49971200 | 94.07 | 92.84 | 61.0 | 95.0 | 190.085 | 97.0 |  |
| 49987584 | 94.71 | 92.85 | 74.0 | 95.0 | 188.645 | 95.0 |  |
| 50003968 | 94.39 | 92.96 | 65.0 | 95.0 | 189.41 | 96.0 |  |
