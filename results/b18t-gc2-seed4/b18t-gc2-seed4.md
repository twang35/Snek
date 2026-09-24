# b18t-gc2-seed4

step **50,003,968** · 3052 evals · trailing **94.24** · peak **94.71** @46,563,328 · sef **94.1** · best30 **98.2** @46,825,472

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
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 2.0 |
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
| seed | 4 |
| torch_threads | 1 |

![b18t-gc2-seed4](b18t-gc2-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.18 | 0.18 | 0.0 | 2.0 | -0.682 | 0.0 |  |
| 32768 | 8.13 | 9.39 | 2.0 | 20.0 | 3.38 | 0.0 |  |
| 49152 | 19.87 | 10.03 | 5.0 | 32.0 | 14.847 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.97 | 94.23 | 64.0 | 95.0 | 185.666 | 93.0 |  |
| 49840128 | 94.26 | 94.19 | 62.0 | 95.0 | 188.947 | 96.0 |  |
| 49856512 | 94.61 | 94.21 | 61.0 | 95.0 | 191.324 | 98.0 |  |
| 49872896 | 94.09 | 94.38 | 8.0 | 95.0 | 190.8 | 98.0 |  |
| 49889280 | 94.1 | 94.36 | 8.0 | 95.0 | 190.801 | 98.0 |  |
| 49905664 | 94.82 | 94.24 | 77.0 | 95.0 | 192.529 | 99.0 |  |
| 49922048 | 94.66 | 94.25 | 77.0 | 95.0 | 191.375 | 98.0 |  |
| 49938432 | 94.91 | 94.28 | 86.0 | 95.0 | 192.606 | 99.0 |  |
| 49954816 | 95.0 | 94.24 | 95.0 | 95.0 | 193.7 | 100.0 |  |
| 49971200 | 94.98 | 94.23 | 93.0 | 95.0 | 192.687 | 99.0 |  |
| 49987584 | 94.3 | 94.26 | 66.0 | 95.0 | 187.006 | 94.0 |  |
| 50003968 | 93.62 | 94.24 | 6.0 | 95.0 | 188.283 | 96.0 |  |
