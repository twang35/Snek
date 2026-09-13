# b12ah-ep2-seed4

step **50,003,968** · 3052 evals · trailing **94.05** · peak **94.39** @30,834,688 · sef **86.7** · best30 **98.1** @31,064,064

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
| ppo_epochs | 2 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
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

![b12ah-ep2-seed4](b12ah-ep2-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.25 | 0.25 | 0.0 | 2.0 | -0.34 | 0.0 |  |
| 32768 | 3.32 | 1.78 | 2.0 | 9.0 | -1.68 | 0.0 |  |
| 49152 | 13.33 | 5.63 | 2.0 | 28.0 | 8.33 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.36 | 94.14 | 77.0 | 95.0 | 187.39 | 94.0 |  |
| 49840128 | 94.0 | 94.17 | 75.0 | 95.0 | 184.045 | 91.0 |  |
| 49856512 | 92.76 | 94.0 | 8.0 | 95.0 | 180.815 | 89.0 |  |
| 49872896 | 93.95 | 94.16 | 64.0 | 95.0 | 186.98 | 94.0 |  |
| 49889280 | 94.2 | 94.17 | 76.0 | 95.0 | 188.225 | 95.0 |  |
| 49905664 | 93.61 | 93.98 | 62.0 | 95.0 | 183.655 | 91.0 |  |
| 49922048 | 93.79 | 94.09 | 22.0 | 95.0 | 188.81 | 96.0 |  |
| 49938432 | 93.87 | 93.99 | 12.0 | 95.0 | 188.89 | 96.0 |  |
| 49954816 | 94.16 | 94.0 | 70.0 | 95.0 | 187.19 | 94.0 |  |
| 49971200 | 92.93 | 94.1 | 12.0 | 95.0 | 185.96 | 94.0 |  |
| 49987584 | 94.46 | 94.02 | 67.0 | 95.0 | 191.47 | 98.0 |  |
| 50003968 | 93.92 | 94.05 | 32.0 | 95.0 | 188.895 | 96.0 |  |
