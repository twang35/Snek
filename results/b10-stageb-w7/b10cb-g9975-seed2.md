# b10cb-g9975-seed2

step **50,003,968** · 3052 evals · trailing **94.65** · peak **94.71** @40,796,160 · sef **92.5** · best30 **98.7** @40,763,392

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.9975 |
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
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 44.5 |
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

![b10cb-g9975-seed2](b10cb-g9975-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.91 | 1.91 | 0.0 | 6.0 | -1.11 | 0.0 |  |
| 32768 | 13.2 | 7.55 | 4.0 | 30.0 | 8.245 | 0.0 |  |
| 49152 | 17.36 | 10.82 | 3.0 | 40.0 | 12.36 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.93 | 94.62 | 88.0 | 95.0 | 192.935 | 99.0 |  |
| 49840128 | 94.73 | 94.69 | 87.0 | 95.0 | 189.75 | 96.0 |  |
| 49856512 | 94.8 | 94.62 | 75.0 | 95.0 | 192.805 | 99.0 |  |
| 49872896 | 94.57 | 94.71 | 63.0 | 95.0 | 191.58 | 98.0 |  |
| 49889280 | 94.85 | 94.71 | 80.0 | 95.0 | 192.855 | 99.0 |  |
| 49905664 | 94.72 | 94.7 | 76.0 | 95.0 | 191.73 | 98.0 |  |
| 49922048 | 94.64 | 94.65 | 74.0 | 95.0 | 189.66 | 96.0 |  |
| 49938432 | 94.91 | 94.71 | 86.0 | 95.0 | 192.915 | 99.0 |  |
| 49954816 | 93.63 | 94.66 | 6.0 | 95.0 | 188.65 | 96.0 |  |
| 49971200 | 95.0 | 94.68 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49987584 | 94.38 | 94.66 | 63.0 | 95.0 | 190.395 | 97.0 |  |
| 50003968 | 94.3 | 94.65 | 66.0 | 95.0 | 190.315 | 97.0 |  |
