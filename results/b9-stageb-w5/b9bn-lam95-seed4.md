# b9bn-lam95-seed4

step **50,003,968** · 3052 evals · trailing **94.01** · peak **94.43** @22,691,840 · sef **93.8** · best30 **96.9** @11,190,272

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
| ppo_gae_lambda | 0.95 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 16.8 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b9bn-lam95-seed4](b9bn-lam95-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.76 | 2.76 | 0.0 | 7.0 | 0.505 | 0.0 |  |
| 32768 | 15.8 | 24.15 | 0.0 | 45.0 | 12.015 | 0.0 |  |
| 49152 | 28.24 | 15.5 | 14.0 | 53.0 | 23.24 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.91 | 93.8 | 66.0 | 95.0 | 186.94 | 94.0 |  |
| 49840128 | 94.17 | 93.74 | 65.0 | 95.0 | 189.145 | 96.0 |  |
| 49856512 | 93.59 | 93.91 | 69.0 | 95.0 | 183.635 | 91.0 |  |
| 49872896 | 94.16 | 93.76 | 30.0 | 95.0 | 191.125 | 98.0 |  |
| 49889280 | 94.54 | 94.04 | 70.0 | 95.0 | 191.55 | 98.0 |  |
| 49905664 | 93.25 | 94.03 | 22.0 | 95.0 | 185.285 | 93.0 |  |
| 49922048 | 93.12 | 93.74 | 22.0 | 95.0 | 185.155 | 93.0 |  |
| 49938432 | 94.52 | 93.84 | 79.0 | 95.0 | 187.55 | 94.0 |  |
| 49954816 | 94.24 | 93.78 | 65.0 | 95.0 | 188.265 | 95.0 |  |
| 49971200 | 94.68 | 93.91 | 85.0 | 95.0 | 189.7 | 96.0 |  |
| 49987584 | 94.83 | 93.98 | 78.0 | 95.0 | 192.835 | 99.0 |  |
| 50003968 | 94.36 | 94.01 | 61.0 | 95.0 | 188.385 | 95.0 |  |
