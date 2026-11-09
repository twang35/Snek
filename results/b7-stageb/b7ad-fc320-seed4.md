# b7ad-fc320-seed4

step **50,003,968** · 3052 evals · trailing **94.0** · peak **94.62** @49,070,080 · sef **91.7** · best30 **97.7** @32,669,696

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
| seed | 4 |
| torch_threads | 1 |

![b7ad-fc320-seed4](b7ad-fc320-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.21 | 0.21 | 0.0 | 1.0 | -0.605 | 0.0 |  |
| 32768 | 19.0 | 13.79 | 0.0 | 37.0 | 14.36 | 0.0 |  |
| 49152 | 22.16 | 11.19 | 4.0 | 39.0 | 17.16 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 95.0 | 94.0 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49840128 | 93.22 | 93.93 | 6.0 | 95.0 | 186.25 | 94.0 |  |
| 49856512 | 94.95 | 93.89 | 92.0 | 95.0 | 191.96 | 98.0 |  |
| 49872896 | 94.88 | 94.02 | 89.0 | 95.0 | 190.895 | 97.0 |  |
| 49889280 | 94.38 | 93.99 | 79.0 | 95.0 | 187.41 | 94.0 |  |
| 49905664 | 94.68 | 94.07 | 78.0 | 95.0 | 188.705 | 95.0 |  |
| 49922048 | 94.61 | 93.95 | 69.0 | 95.0 | 191.62 | 98.0 |  |
| 49938432 | 94.44 | 93.95 | 81.0 | 95.0 | 186.475 | 93.0 |  |
| 49954816 | 94.25 | 93.94 | 52.0 | 95.0 | 188.23 | 95.0 |  |
| 49971200 | 94.61 | 93.98 | 70.0 | 95.0 | 191.62 | 98.0 |  |
| 49987584 | 93.82 | 94.02 | 28.0 | 95.0 | 186.85 | 94.0 |  |
| 50003968 | 94.71 | 94.0 | 76.0 | 95.0 | 191.675 | 98.0 |  |
