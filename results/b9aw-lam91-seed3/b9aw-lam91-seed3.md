# b9aw-lam91-seed3

step **50,003,968** · 3052 evals · trailing **94.26** · peak **94.45** @39,911,424 · sef **89.8** · best30 **96.7** @10,534,912

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
| ppo_gae_lambda | 0.91 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 10.1 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b9aw-lam91-seed3](b9aw-lam91-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.03 | 0.03 | 0.0 | 1.0 | -0.56 | 0.0 |  |
| 32768 | 1.14 | 0.58 | 0.0 | 4.0 | 0.64 | 0.0 |  |
| 49152 | 14.86 | 14.09 | 0.0 | 42.0 | 12.425 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.94 | 94.29 | 78.0 | 95.0 | 182.9 | 90.0 |  |
| 49840128 | 94.51 | 94.27 | 74.0 | 95.0 | 188.535 | 95.0 |  |
| 49856512 | 94.58 | 94.24 | 83.0 | 95.0 | 186.57 | 93.0 |  |
| 49872896 | 92.98 | 94.21 | 18.0 | 95.0 | 183.025 | 91.0 |  |
| 49889280 | 93.54 | 94.24 | 65.0 | 95.0 | 181.595 | 89.0 |  |
| 49905664 | 94.86 | 94.22 | 84.0 | 95.0 | 191.825 | 98.0 |  |
| 49922048 | 94.26 | 94.26 | 40.0 | 95.0 | 188.195 | 95.0 |  |
| 49938432 | 93.63 | 94.22 | 28.0 | 95.0 | 186.66 | 94.0 |  |
| 49954816 | 94.41 | 94.27 | 77.0 | 95.0 | 188.39 | 95.0 |  |
| 49971200 | 94.17 | 94.26 | 14.0 | 95.0 | 190.095 | 97.0 |  |
| 49987584 | 94.74 | 94.22 | 82.0 | 95.0 | 190.71 | 97.0 |  |
| 50003968 | 94.64 | 94.26 | 62.0 | 95.0 | 191.65 | 98.0 |  |
