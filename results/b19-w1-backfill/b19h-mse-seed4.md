# b19h-mse-seed4

step **50,003,968** · 3052 evals · trailing **94.55** · peak **94.62** @48,529,408 · sef **95.2** · best30 **98.2** @34,455,552

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
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | mse |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b19h-mse-seed4](b19h-mse-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.32 | 0.32 | 0.0 | 2.0 | -0.459 | 0.0 |  |
| 32768 | 13.6 | 18.04 | 1.0 | 28.0 | 10.145 | 0.0 |  |
| 49152 | 23.35 | 11.84 | 2.0 | 42.0 | 18.618 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.24 | 94.45 | 26.0 | 95.0 | 190.948 | 98.0 |  |
| 49840128 | 94.67 | 94.49 | 62.0 | 95.0 | 192.367 | 99.0 |  |
| 49856512 | 94.68 | 94.47 | 63.0 | 95.0 | 192.387 | 99.0 |  |
| 49872896 | 93.78 | 94.46 | 8.0 | 95.0 | 189.496 | 97.0 |  |
| 49889280 | 94.7 | 94.48 | 66.0 | 95.0 | 191.352 | 98.0 |  |
| 49905664 | 94.93 | 94.5 | 88.0 | 95.0 | 192.629 | 99.0 |  |
| 49922048 | 94.66 | 94.5 | 75.0 | 95.0 | 191.361 | 98.0 |  |
| 49938432 | 94.03 | 94.51 | 65.0 | 95.0 | 187.732 | 95.0 |  |
| 49954816 | 94.81 | 94.53 | 76.0 | 95.0 | 192.516 | 99.0 |  |
| 49971200 | 94.45 | 94.53 | 74.0 | 95.0 | 190.159 | 97.0 |  |
| 49987584 | 94.95 | 94.54 | 90.0 | 95.0 | 192.655 | 99.0 |  |
| 50003968 | 95.0 | 94.55 | 95.0 | 95.0 | 193.709 | 100.0 |  |
