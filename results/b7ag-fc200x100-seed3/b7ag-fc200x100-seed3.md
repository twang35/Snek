# b7ag-fc200x100-seed3

step **50,003,968** · 3052 evals · trailing **93.31** · peak **94.46** @40,747,008 · sef **95.1** · best30 **97.7** @39,813,120

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
| fc_layers | (200, 100) |
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
| seed | 3 |
| torch_threads | 1 |

![b7ag-fc200x100-seed3](b7ag-fc200x100-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 19.2 | 19.2 | 2.0 | 36.0 | 14.245 | 0.0 |  |
| 32768 | 34.85 | 28.2 | 6.0 | 57.0 | 29.94 | 0.0 |  |
| 49152 | 29.77 | 24.48 | 13.0 | 50.0 | 24.86 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.63 | 93.18 | 12.0 | 95.0 | 181.635 | 90.0 |  |
| 49840128 | 92.41 | 93.34 | 52.0 | 95.0 | 177.48 | 86.0 |  |
| 49856512 | 94.32 | 93.29 | 73.0 | 95.0 | 188.345 | 95.0 |  |
| 49872896 | 94.74 | 93.27 | 74.0 | 95.0 | 191.75 | 98.0 |  |
| 49889280 | 92.36 | 93.2 | 15.0 | 95.0 | 182.405 | 91.0 |  |
| 49905664 | 92.91 | 93.47 | 10.0 | 95.0 | 170.385 | 79.0 |  |
| 49922048 | 94.84 | 93.44 | 79.0 | 95.0 | 192.845 | 99.0 |  |
| 49938432 | 93.43 | 93.33 | 34.0 | 95.0 | 184.47 | 92.0 |  |
| 49954816 | 94.03 | 93.51 | 70.0 | 95.0 | 186.065 | 93.0 |  |
| 49971200 | 94.57 | 93.5 | 71.0 | 95.0 | 190.585 | 97.0 |  |
| 49987584 | 94.26 | 93.42 | 74.0 | 95.0 | 189.28 | 96.0 |  |
| 50003968 | 92.82 | 93.31 | 10.0 | 95.0 | 181.87 | 90.0 |  |
