# b10bw-g995-seed1

step **50,003,968** · 3052 evals · trailing **94.55** · peak **94.69** @49,217,536 · sef **91.7** · best30 **97.7** @49,446,912

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.995 |
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
| ppo_horizon | 40.2 |
| ppo_learning_rate | 0.0003 |
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

![b10bw-g995-seed1](b10bw-g995-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 17.37 | 17.37 | 1.0 | 39.0 | 14.485 | 0.0 |  |
| 32768 | 47.26 | 32.43 | 7.0 | 80.0 | 42.44 | 0.0 |  |
| 49152 | 27.9 | 22.63 | 7.0 | 53.0 | 22.99 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.85 | 94.55 | 12.0 | 95.0 | 189.865 | 97.0 |  |
| 49840128 | 94.83 | 94.56 | 81.0 | 95.0 | 191.84 | 98.0 |  |
| 49856512 | 94.56 | 94.56 | 68.0 | 95.0 | 189.58 | 96.0 |  |
| 49872896 | 94.03 | 94.53 | 18.0 | 95.0 | 189.05 | 96.0 |  |
| 49889280 | 94.57 | 94.51 | 80.0 | 95.0 | 188.595 | 95.0 |  |
| 49905664 | 93.89 | 94.59 | 38.0 | 95.0 | 184.93 | 92.0 |  |
| 49922048 | 94.78 | 94.54 | 80.0 | 95.0 | 191.79 | 98.0 |  |
| 49938432 | 94.36 | 94.55 | 69.0 | 95.0 | 188.385 | 95.0 |  |
| 49954816 | 94.87 | 94.56 | 82.0 | 95.0 | 192.875 | 99.0 |  |
| 49971200 | 94.88 | 94.55 | 83.0 | 95.0 | 192.885 | 99.0 |  |
| 49987584 | 94.75 | 94.55 | 87.0 | 95.0 | 189.77 | 96.0 |  |
| 50003968 | 94.15 | 94.55 | 67.0 | 95.0 | 188.175 | 95.0 |  |
