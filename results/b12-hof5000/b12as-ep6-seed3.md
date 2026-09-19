# b12as-ep6-seed3

step **50,003,968** · 3052 evals · trailing **93.9** · peak **94.6** @32,833,536 · sef **90.1** · best30 **98.4** @31,293,440

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
| ppo_epochs | 6 |
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
| seed | 3 |
| torch_threads | 1 |

![b12as-ep6-seed3](b12as-ep6-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.03 | 0.03 | 0.0 | 1.0 | -1.01 | 0.0 |  |
| 32768 | 1.7 | 0.86 | 1.0 | 8.0 | 1.11 | 0.0 |  |
| 49152 | 8.97 | 8.72 | 2.0 | 22.0 | 5.545 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.83 | 93.52 | 76.0 | 95.0 | 184.87 | 92.0 |  |
| 49840128 | 93.82 | 93.61 | 60.0 | 95.0 | 184.77 | 92.0 |  |
| 49856512 | 95.0 | 93.62 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49872896 | 95.0 | 93.67 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49889280 | 94.46 | 93.84 | 72.0 | 95.0 | 190.385 | 97.0 |  |
| 49905664 | 93.98 | 93.86 | 12.0 | 95.0 | 189.995 | 97.0 |  |
| 49922048 | 94.94 | 94.19 | 89.0 | 95.0 | 192.9 | 99.0 |  |
| 49938432 | 94.69 | 93.79 | 66.0 | 95.0 | 191.655 | 98.0 |  |
| 49954816 | 94.97 | 93.82 | 92.0 | 95.0 | 192.93 | 99.0 |  |
| 49971200 | 94.45 | 93.8 | 70.0 | 95.0 | 189.47 | 96.0 |  |
| 49987584 | 93.78 | 94.06 | 10.0 | 95.0 | 186.765 | 94.0 |  |
| 50003968 | 93.9 | 93.9 | 6.0 | 95.0 | 188.92 | 96.0 |  |
