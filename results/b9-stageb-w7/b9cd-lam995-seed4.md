# b9cd-lam995-seed4

step **50,003,968** · 3052 evals · trailing **94.08** · peak **94.49** @45,596,672 · sef **92.4** · best30 **98.1** @41,811,968

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
| seed | 4 |
| torch_threads | 1 |

![b9cd-lam995-seed4](b9cd-lam995-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.35 | 0.35 | 0.0 | 2.0 | -0.96 | 0.0 |  |
| 32768 | 19.32 | 9.84 | 2.0 | 36.0 | 14.32 | 0.0 |  |
| 49152 | 24.66 | 14.78 | 7.0 | 48.0 | 19.66 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.65 | 93.91 | 80.0 | 95.0 | 190.62 | 97.0 |  |
| 49840128 | 94.91 | 94.02 | 90.0 | 95.0 | 191.92 | 98.0 |  |
| 49856512 | 94.3 | 93.92 | 75.0 | 95.0 | 188.325 | 95.0 |  |
| 49872896 | 93.96 | 93.85 | 65.0 | 95.0 | 185.995 | 93.0 |  |
| 49889280 | 94.54 | 94.01 | 85.0 | 95.0 | 186.575 | 93.0 |  |
| 49905664 | 93.26 | 94.08 | 8.0 | 95.0 | 190.27 | 98.0 |  |
| 49922048 | 94.37 | 94.11 | 70.0 | 95.0 | 189.39 | 96.0 |  |
| 49938432 | 94.94 | 94.04 | 89.0 | 95.0 | 192.945 | 99.0 |  |
| 49954816 | 93.97 | 94.01 | 6.0 | 95.0 | 189.985 | 97.0 |  |
| 49971200 | 94.01 | 94.01 | 9.0 | 95.0 | 189.98 | 97.0 |  |
| 49987584 | 94.84 | 94.14 | 84.0 | 95.0 | 191.85 | 98.0 |  |
| 50003968 | 93.69 | 94.08 | 8.0 | 95.0 | 188.665 | 96.0 |  |
