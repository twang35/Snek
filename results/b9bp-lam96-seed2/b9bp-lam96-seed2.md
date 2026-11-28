# b9bp-lam96-seed2

step **50,003,968** · 3052 evals · trailing **93.97** · peak **94.51** @12,435,456 · sef **91.7** · best30 **97.8** @12,730,368

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
| ppo_gae_lambda | 0.96 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 20.2 |
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

![b9bp-lam96-seed2](b9bp-lam96-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.21 | 2.21 | 0.0 | 7.0 | -1.125 | 0.0 |  |
| 32768 | 17.29 | 9.75 | 5.0 | 35.0 | 13.145 | 0.0 |  |
| 49152 | 29.12 | 16.21 | 11.0 | 67.0 | 24.12 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.97 | 94.28 | 75.0 | 95.0 | 183.02 | 90.0 |  |
| 49840128 | 94.1 | 94.22 | 8.0 | 95.0 | 191.11 | 98.0 |  |
| 49856512 | 94.38 | 94.21 | 78.0 | 95.0 | 186.415 | 93.0 |  |
| 49872896 | 94.2 | 94.21 | 60.0 | 95.0 | 189.22 | 96.0 |  |
| 49889280 | 94.14 | 94.2 | 66.0 | 95.0 | 186.175 | 93.0 |  |
| 49905664 | 93.25 | 94.23 | 16.0 | 95.0 | 184.29 | 92.0 |  |
| 49922048 | 93.93 | 94.28 | 64.0 | 95.0 | 187.955 | 95.0 |  |
| 49938432 | 92.3 | 94.06 | 18.0 | 95.0 | 175.38 | 84.0 |  |
| 49954816 | 93.09 | 94.14 | 65.0 | 95.0 | 180.15 | 88.0 |  |
| 49971200 | 93.8 | 94.03 | 74.0 | 95.0 | 181.855 | 89.0 |  |
| 49987584 | 94.36 | 94.01 | 78.0 | 95.0 | 187.345 | 94.0 |  |
| 50003968 | 93.8 | 93.97 | 75.0 | 95.0 | 183.845 | 91.0 |  |
