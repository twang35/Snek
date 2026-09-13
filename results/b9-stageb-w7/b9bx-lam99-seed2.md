# b9bx-lam99-seed2

step **50,003,968** · 3052 evals · trailing **94.14** · peak **94.53** @31,604,736 · sef **91.9** · best30 **98.3** @43,450,368

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
| seed | 2 |
| torch_threads | 1 |

![b9bx-lam99-seed2](b9bx-lam99-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.53 | 1.53 | 0.0 | 6.0 | -1.175 | 0.0 |  |
| 32768 | 6.94 | 4.24 | 3.0 | 14.0 | 2.03 | 0.0 |  |
| 49152 | 15.72 | 8.06 | 4.0 | 37.0 | 10.72 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.65 | 94.13 | 71.0 | 95.0 | 191.66 | 98.0 |  |
| 49840128 | 93.1 | 93.91 | 20.0 | 95.0 | 187.125 | 95.0 |  |
| 49856512 | 94.34 | 94.02 | 67.0 | 95.0 | 190.355 | 97.0 |  |
| 49872896 | 94.08 | 94.03 | 57.0 | 95.0 | 188.105 | 95.0 |  |
| 49889280 | 94.63 | 94.06 | 65.0 | 95.0 | 191.64 | 98.0 |  |
| 49905664 | 94.53 | 94.13 | 51.0 | 95.0 | 191.54 | 98.0 |  |
| 49922048 | 94.25 | 94.03 | 28.0 | 95.0 | 191.26 | 98.0 |  |
| 49938432 | 94.34 | 94.12 | 59.0 | 95.0 | 190.31 | 97.0 |  |
| 49954816 | 93.99 | 94.09 | 48.0 | 95.0 | 188.965 | 96.0 |  |
| 49971200 | 94.72 | 94.13 | 72.0 | 95.0 | 191.685 | 98.0 |  |
| 49987584 | 94.93 | 94.08 | 88.0 | 95.0 | 192.935 | 99.0 |  |
| 50003968 | 94.2 | 94.14 | 70.0 | 95.0 | 186.235 | 93.0 |  |
