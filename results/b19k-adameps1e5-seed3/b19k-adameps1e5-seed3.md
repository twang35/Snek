# b19k-adameps1e5-seed3

step **50,003,968** · 3052 evals · trailing **94.41** · peak **94.63** @36,028,416 · sef **92.3** · best30 **98.1** @47,185,920

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
| ppo_adam_epsilon | 1e-05 |
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
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b19k-adameps1e5-seed3](b19k-adameps1e5-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.1 | 0.1 | 0.0 | 1.0 | -3.522 | 0.0 |  |
| 32768 | 6.35 | 3.22 | 0.0 | 21.0 | 4.831 | 0.0 |  |
| 49152 | 18.25 | 8.23 | 0.0 | 35.0 | 13.675 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.71 | 94.5 | 71.0 | 95.0 | 191.392 | 98.0 |  |
| 49840128 | 94.81 | 94.5 | 76.0 | 95.0 | 192.501 | 99.0 |  |
| 49856512 | 94.53 | 94.53 | 72.0 | 95.0 | 190.24 | 97.0 |  |
| 49872896 | 94.94 | 94.5 | 91.0 | 95.0 | 191.603 | 98.0 |  |
| 49889280 | 94.9 | 94.56 | 85.0 | 95.0 | 192.595 | 99.0 |  |
| 49905664 | 94.61 | 94.55 | 71.0 | 95.0 | 190.315 | 97.0 |  |
| 49922048 | 94.89 | 94.53 | 86.0 | 95.0 | 191.585 | 98.0 |  |
| 49938432 | 94.38 | 94.44 | 65.0 | 95.0 | 190.09 | 97.0 |  |
| 49954816 | 94.48 | 94.43 | 61.0 | 95.0 | 188.143 | 95.0 |  |
| 49971200 | 94.75 | 94.43 | 70.0 | 95.0 | 192.441 | 99.0 |  |
| 49987584 | 93.94 | 94.41 | 14.0 | 95.0 | 190.643 | 98.0 |  |
| 50003968 | 94.97 | 94.41 | 92.0 | 95.0 | 192.61 | 99.0 |  |
