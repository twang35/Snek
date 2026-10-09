# b9bh-lam94-seed2

step **50,003,968** · 3052 evals · trailing **94.0** · peak **94.42** @19,644,416 · sef **89.1** · best30 **96.6** @19,808,256

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
| ppo_gae_lambda | 0.94 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 14.4 |
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

![b9bh-lam94-seed2](b9bh-lam94-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.25 | 2.25 | 0.0 | 7.0 | -1.4 | 0.0 |  |
| 32768 | 15.46 | 8.86 | 4.0 | 31.0 | 10.505 | 0.0 |  |
| 49152 | 23.73 | 13.81 | 6.0 | 44.0 | 18.775 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.72 | 93.77 | 67.0 | 95.0 | 192.725 | 99.0 |  |
| 49840128 | 92.98 | 93.77 | 45.0 | 95.0 | 181.985 | 90.0 |  |
| 49856512 | 94.2 | 93.79 | 61.0 | 95.0 | 188.225 | 95.0 |  |
| 49872896 | 93.49 | 93.81 | 69.0 | 95.0 | 184.53 | 92.0 |  |
| 49889280 | 93.97 | 93.87 | 68.0 | 95.0 | 187.0 | 94.0 |  |
| 49905664 | 93.82 | 94.0 | 14.0 | 95.0 | 186.85 | 94.0 |  |
| 49922048 | 94.74 | 94.02 | 87.0 | 95.0 | 188.765 | 95.0 |  |
| 49938432 | 94.49 | 94.04 | 81.0 | 95.0 | 187.475 | 94.0 |  |
| 49954816 | 94.63 | 93.86 | 80.0 | 95.0 | 188.655 | 95.0 |  |
| 49971200 | 93.86 | 94.02 | 75.0 | 95.0 | 184.9 | 92.0 |  |
| 49987584 | 94.87 | 93.92 | 82.0 | 95.0 | 192.875 | 99.0 |  |
| 50003968 | 94.12 | 94.0 | 61.0 | 95.0 | 188.145 | 95.0 |  |
