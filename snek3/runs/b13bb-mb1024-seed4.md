# b13bb-mb1024-seed4

step **50,003,968** · 3052 evals · trailing **94.15** · peak **94.47** @44,531,712 · sef **84.0** · best30 **97.9** @28,688,384

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
| ppo_minibatch | 1024 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b13bb-mb1024-seed4](b13bb-mb1024-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.01 | 0.01 | 0.0 | 1.0 | -0.49 | 0.0 |  |
| 32768 | 7.34 | 3.67 | 2.0 | 15.0 | 2.34 | 0.0 |  |
| 49152 | 13.61 | 6.99 | 2.0 | 27.0 | 8.61 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.37 | 94.28 | 58.0 | 95.0 | 191.38 | 98.0 |  |
| 49840128 | 92.93 | 94.25 | 10.0 | 95.0 | 184.92 | 93.0 |  |
| 49856512 | 94.24 | 94.28 | 62.0 | 95.0 | 189.26 | 96.0 |  |
| 49872896 | 94.31 | 94.23 | 56.0 | 95.0 | 190.325 | 97.0 |  |
| 49889280 | 94.64 | 94.29 | 59.0 | 95.0 | 192.645 | 99.0 |  |
| 49905664 | 93.18 | 94.26 | 8.0 | 95.0 | 188.2 | 96.0 |  |
| 49922048 | 93.2 | 94.22 | 2.0 | 95.0 | 186.23 | 94.0 |  |
| 49938432 | 93.98 | 94.21 | 61.0 | 95.0 | 187.01 | 94.0 |  |
| 49954816 | 94.23 | 94.2 | 73.0 | 95.0 | 188.255 | 95.0 |  |
| 49971200 | 93.63 | 94.25 | 48.0 | 95.0 | 187.655 | 95.0 |  |
| 49987584 | 94.72 | 94.27 | 73.0 | 95.0 | 191.73 | 98.0 |  |
| 50003968 | 93.01 | 94.15 | 59.0 | 95.0 | 181.065 | 89.0 |  |
