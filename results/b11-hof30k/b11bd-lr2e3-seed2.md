# b11bd-lr2e3-seed2

step **50,003,968** · 3052 evals · trailing **88.71** · peak **93.87** @11,763,712 · sef **87.5** · best30 **95.9** @9,519,104

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
| ppo_learning_rate | 0.002 |
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

![b11bd-lr2e3-seed2](b11bd-lr2e3-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.27 | 0.27 | 0.0 | 4.0 | -0.32 | 0.0 |  |
| 32768 | 12.27 | 6.27 | 5.0 | 30.0 | 7.27 | 0.0 |  |
| 49152 | 31.95 | 19.79 | 0.0 | 61.0 | 27.58 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 88.41 | 85.79 | 19.0 | 95.0 | 170.95 | 84.0 |  |
| 49840128 | 88.51 | 84.98 | 28.0 | 95.0 | 172.045 | 85.0 |  |
| 49856512 | 89.32 | 84.74 | 21.0 | 95.0 | 172.9 | 85.0 |  |
| 49872896 | 89.58 | 84.89 | 21.0 | 95.0 | 171.125 | 83.0 |  |
| 49889280 | 90.81 | 85.57 | 3.0 | 95.0 | 179.59 | 90.0 |  |
| 49905664 | 89.93 | 87.27 | 14.0 | 95.0 | 171.745 | 83.0 |  |
| 49922048 | 90.91 | 85.44 | 21.0 | 95.0 | 178.74 | 89.0 |  |
| 49938432 | 94.75 | 84.81 | 70.0 | 95.0 | 192.755 | 99.0 |  |
| 49954816 | 94.07 | 84.76 | 56.0 | 95.0 | 188.095 | 95.0 |  |
| 49971200 | 94.69 | 84.79 | 80.0 | 95.0 | 190.66 | 97.0 |  |
| 49987584 | 92.05 | 88.26 | 18.0 | 95.0 | 179.02 | 88.0 |  |
| 50003968 | 94.12 | 88.71 | 56.0 | 95.0 | 188.145 | 95.0 |  |
