# b9aa-lam0-seed1

step **50,003,968** · 3052 evals · trailing **92.04** · peak **94.09** @35,717,120 · sef **27.1** · best30 **88.7** @35,733,504

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
| ppo_gae_lambda | 0.0 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 1.0 |
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

![b9aa-lam0-seed1](b9aa-lam0-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 14.76 | 20.13 | 0.0 | 61.0 | 13.18 | 0.0 |  |
| 32768 | 1.74 | 1.74 | 0.0 | 17.0 | 1.24 | 0.0 |  |
| 49152 | 42.17 | 30.62 | 0.0 | 85.0 | 37.845 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 90.62 | 92.15 | 41.0 | 95.0 | 148.02 | 60.0 |  |
| 49840128 | 91.11 | 92.05 | 24.0 | 95.0 | 156.83 | 68.0 |  |
| 49856512 | 91.61 | 91.74 | 18.0 | 95.0 | 157.33 | 68.0 |  |
| 49872896 | 92.51 | 91.78 | 54.0 | 95.0 | 162.39 | 72.0 |  |
| 49889280 | 91.77 | 91.7 | 58.0 | 95.0 | 151.25 | 62.0 |  |
| 49905664 | 90.29 | 91.82 | 51.0 | 95.0 | 143.53 | 56.0 |  |
| 49922048 | 92.52 | 91.82 | 56.0 | 95.0 | 163.44 | 73.0 |  |
| 49938432 | 91.84 | 91.8 | 26.0 | 95.0 | 162.76 | 73.0 |  |
| 49954816 | 92.75 | 91.87 | 49.0 | 95.0 | 167.83 | 77.0 |  |
| 49971200 | 92.38 | 91.94 | 10.0 | 95.0 | 170.58 | 80.0 |  |
| 49987584 | 93.69 | 91.93 | 55.0 | 95.0 | 179.17 | 87.0 |  |
| 50003968 | 93.32 | 92.04 | 7.0 | 95.0 | 175.68 | 84.0 |  |
