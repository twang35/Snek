# b9bw-lam99-seed1

step **50,003,968** · 3052 evals · trailing **94.08** · peak **94.61** @46,743,552 · sef **91.4** · best30 **98.3** @44,761,088

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
| seed | 1 |
| torch_threads | 1 |

![b9bw-lam99-seed1](b9bw-lam99-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 20.07 | 20.07 | 3.0 | 45.0 | 16.51 | 0.0 |  |
| 32768 | 38.37 | 31.98 | 8.0 | 67.0 | 33.415 | 0.0 |  |
| 49152 | 37.49 | 28.78 | 9.0 | 68.0 | 32.49 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.67 | 94.19 | 62.0 | 95.0 | 192.675 | 99.0 |  |
| 49840128 | 95.0 | 94.15 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49856512 | 94.12 | 94.13 | 62.0 | 95.0 | 187.15 | 94.0 |  |
| 49872896 | 94.32 | 94.11 | 67.0 | 95.0 | 189.34 | 96.0 |  |
| 49889280 | 94.62 | 94.13 | 81.0 | 95.0 | 189.64 | 96.0 |  |
| 49905664 | 94.81 | 94.1 | 76.0 | 95.0 | 192.815 | 99.0 |  |
| 49922048 | 94.86 | 94.12 | 86.0 | 95.0 | 191.87 | 98.0 |  |
| 49938432 | 95.0 | 94.08 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49954816 | 93.78 | 94.11 | 59.0 | 95.0 | 185.815 | 93.0 |  |
| 49971200 | 93.66 | 94.08 | 27.0 | 95.0 | 183.66 | 91.0 |  |
| 49987584 | 92.34 | 94.03 | 59.0 | 95.0 | 173.43 | 82.0 |  |
| 50003968 | 93.1 | 94.08 | 20.0 | 95.0 | 186.085 | 94.0 |  |
