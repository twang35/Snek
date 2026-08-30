# p3g-fc200x100-seed7

step **58,114,048** · 3536 evals · trailing **94.14** · peak **94.38** @15,826,944 · sef **96.2** · best30 **97.5** @14,467,072

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 6 |
| fc_layers | (200, 100) |
| graph_eval_episodes | 100 |
| max_steps | 400000000 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 7 |
| torch_threads | 1 |

![p3g-fc200x100-seed7](p3g-fc200x100-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 4.78 | 4.78 | 0.0 | 19.0 | 2.435 | 0.0 |  |
| 32768 | 27.1 | 15.94 | 4.0 | 49.0 | 22.1 | 0.0 |  |
| 49152 | 31.29 | 21.06 | 4.0 | 65.0 | 26.29 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 57753600 | 94.83 | 94.27 | 78.0 | 95.0 | 192.835 | 99.0 |  |
| 57769984 | 94.05 | 94.34 | 17.0 | 95.0 | 191.06 | 98.0 |  |
| 57786368 | 94.06 | 94.31 | 1.0 | 95.0 | 192.065 | 99.0 |  |
| 57802752 | 93.98 | 94.31 | 7.0 | 95.0 | 189.995 | 97.0 |  |
| 57819136 | 93.56 | 94.28 | 3.0 | 95.0 | 186.59 | 94.0 |  |
| 57835520 | 94.18 | 94.14 | 69.0 | 95.0 | 187.21 | 94.0 |  |
| 57851904 | 94.81 | 94.2 | 87.0 | 95.0 | 190.78 | 97.0 |  |
| 57917440 | 94.88 | 94.33 | 83.0 | 95.0 | 192.885 | 99.0 |  |
| 57933824 | 94.07 | 94.28 | 72.0 | 95.0 | 184.07 | 91.0 |  |
| 57950208 | 93.79 | 94.16 | 10.0 | 95.0 | 188.765 | 96.0 |  |
| 57999360 | 94.9 | 94.18 | 90.0 | 95.0 | 191.91 | 98.0 |  |
| 58114048 | 93.21 | 94.14 | 39.0 | 95.0 | 183.21 | 91.0 |  |
