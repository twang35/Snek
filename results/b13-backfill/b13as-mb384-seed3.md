# b13as-mb384-seed3

step **50,003,968** · 3052 evals · trailing **94.26** · peak **94.48** @25,772,032 · sef **91.7** · best30 **98.1** @48,332,800

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
| ppo_minibatch | 384 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b13as-mb384-seed3](b13as-mb384-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.2 | 0.2 | 0.0 | 2.0 | -0.3 | 0.0 |  |
| 32768 | 4.79 | 6.95 | 0.0 | 15.0 | 2.715 | 0.0 |  |
| 49152 | 15.87 | 8.04 | 0.0 | 32.0 | 11.05 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.37 | 94.32 | 56.0 | 95.0 | 191.38 | 98.0 |  |
| 49840128 | 94.48 | 94.37 | 66.0 | 95.0 | 190.495 | 97.0 |  |
| 49856512 | 94.16 | 94.29 | 49.0 | 95.0 | 191.125 | 98.0 |  |
| 49872896 | 94.36 | 94.22 | 41.0 | 95.0 | 191.28 | 98.0 |  |
| 49889280 | 94.44 | 94.23 | 71.0 | 95.0 | 190.41 | 97.0 |  |
| 49905664 | 94.13 | 94.23 | 14.0 | 95.0 | 191.14 | 98.0 |  |
| 49922048 | 95.0 | 94.24 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49938432 | 93.89 | 94.25 | 8.0 | 95.0 | 190.9 | 98.0 |  |
| 49954816 | 94.05 | 94.28 | 21.0 | 95.0 | 189.025 | 96.0 |  |
| 49971200 | 94.83 | 94.3 | 78.0 | 95.0 | 192.835 | 99.0 |  |
| 49987584 | 94.57 | 94.27 | 78.0 | 95.0 | 189.59 | 96.0 |  |
| 50003968 | 94.91 | 94.26 | 88.0 | 95.0 | 191.92 | 98.0 |  |
