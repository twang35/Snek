# b9cb-lam995-seed2

step **50,003,968** · 3052 evals · trailing **94.36** · peak **94.56** @42,237,952 · sef **87.5** · best30 **98.5** @18,595,840

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
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.995 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 66.9 |
| ppo_learning_rate | 0.0003 |
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

![b9cb-lam995-seed2](b9cb-lam995-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.72 | 1.72 | 0.0 | 4.0 | -1.075 | 0.0 |  |
| 32768 | 6.12 | 3.92 | 2.0 | 16.0 | 1.12 | 0.0 |  |
| 49152 | 11.73 | 6.52 | 3.0 | 29.0 | 6.73 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.67 | 94.25 | 8.0 | 95.0 | 187.65 | 95.0 |  |
| 49840128 | 94.55 | 94.28 | 78.0 | 95.0 | 188.575 | 95.0 |  |
| 49856512 | 93.24 | 94.27 | 10.0 | 95.0 | 189.21 | 97.0 |  |
| 49872896 | 94.55 | 94.27 | 69.0 | 95.0 | 191.56 | 98.0 |  |
| 49889280 | 94.31 | 94.28 | 64.0 | 95.0 | 189.33 | 96.0 |  |
| 49905664 | 94.07 | 94.25 | 65.0 | 95.0 | 187.1 | 94.0 |  |
| 49922048 | 94.91 | 94.28 | 90.0 | 95.0 | 191.92 | 98.0 |  |
| 49938432 | 94.72 | 94.37 | 83.0 | 95.0 | 190.735 | 97.0 |  |
| 49954816 | 94.31 | 94.35 | 69.0 | 95.0 | 188.335 | 95.0 |  |
| 49971200 | 94.51 | 94.27 | 64.0 | 95.0 | 189.53 | 96.0 |  |
| 49987584 | 94.81 | 94.37 | 76.0 | 95.0 | 192.815 | 99.0 |  |
| 50003968 | 94.11 | 94.36 | 20.0 | 95.0 | 190.125 | 97.0 |  |
