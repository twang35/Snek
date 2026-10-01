# b13ao-mb192-seed3

step **50,003,968** · 3052 evals · trailing **94.05** · peak **94.65** @44,662,784 · sef **91.8** · best30 **98.2** @16,171,008

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
| ppo_minibatch | 192 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b13ao-mb192-seed3](b13ao-mb192-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.16 | 0.16 | 0.0 | 2.0 | -3.4 | 0.0 |  |
| 32768 | 5.07 | 7.56 | 0.0 | 20.0 | 3.31 | 0.0 |  |
| 49152 | 17.45 | 8.8 | 0.0 | 32.0 | 12.72 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.54 | 94.06 | 59.0 | 95.0 | 191.55 | 98.0 |  |
| 49840128 | 94.88 | 94.06 | 83.0 | 95.0 | 192.885 | 99.0 |  |
| 49856512 | 94.5 | 94.04 | 59.0 | 95.0 | 191.51 | 98.0 |  |
| 49872896 | 94.96 | 94.04 | 91.0 | 95.0 | 192.965 | 99.0 |  |
| 49889280 | 94.33 | 94.03 | 59.0 | 95.0 | 190.345 | 97.0 |  |
| 49905664 | 92.85 | 94.01 | 16.0 | 95.0 | 187.825 | 96.0 |  |
| 49922048 | 94.22 | 94.01 | 61.0 | 95.0 | 189.24 | 96.0 |  |
| 49938432 | 94.88 | 94.16 | 83.0 | 95.0 | 192.885 | 99.0 |  |
| 49954816 | 94.49 | 94.04 | 69.0 | 95.0 | 191.455 | 98.0 |  |
| 49971200 | 95.0 | 94.12 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49987584 | 94.18 | 94.13 | 16.0 | 95.0 | 191.19 | 98.0 |  |
| 50003968 | 92.03 | 94.05 | 6.0 | 95.0 | 183.93 | 93.0 |  |
