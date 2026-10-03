# b9cf-lam999-seed2

step **50,003,968** · 3052 evals · trailing **94.13** · peak **94.57** @35,143,680 · sef **88.4** · best30 **98.2** @41,615,360

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
| ppo_gae_lambda | 0.999 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 91.0 |
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

![b9cf-lam999-seed2](b9cf-lam999-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.92 | 1.92 | 0.0 | 6.0 | -1.685 | 0.0 |  |
| 32768 | 3.0 | 2.46 | 2.0 | 9.0 | -2.0 | 0.0 |  |
| 49152 | 7.78 | 4.23 | 2.0 | 19.0 | 2.78 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.7 | 94.23 | 24.0 | 95.0 | 185.73 | 94.0 |  |
| 49840128 | 94.42 | 94.17 | 68.0 | 95.0 | 190.435 | 97.0 |  |
| 49856512 | 93.71 | 94.15 | 27.0 | 95.0 | 188.685 | 96.0 |  |
| 49872896 | 93.06 | 94.11 | 20.0 | 95.0 | 186.995 | 95.0 |  |
| 49889280 | 93.74 | 94.08 | 4.0 | 95.0 | 190.75 | 98.0 |  |
| 49905664 | 94.73 | 94.12 | 77.0 | 95.0 | 191.74 | 98.0 |  |
| 49922048 | 93.76 | 94.12 | 56.0 | 95.0 | 187.785 | 95.0 |  |
| 49938432 | 94.93 | 94.12 | 88.0 | 95.0 | 192.935 | 99.0 |  |
| 49954816 | 93.36 | 94.08 | 39.0 | 95.0 | 188.245 | 96.0 |  |
| 49971200 | 93.25 | 94.04 | 6.0 | 95.0 | 189.265 | 97.0 |  |
| 49987584 | 93.33 | 94.01 | 26.0 | 95.0 | 188.305 | 96.0 |  |
| 50003968 | 94.28 | 94.13 | 42.0 | 95.0 | 191.29 | 98.0 |  |
