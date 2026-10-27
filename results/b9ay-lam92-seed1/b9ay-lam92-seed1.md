# b9ay-lam92-seed1

step **50,003,968** · 3052 evals · trailing **93.95** · peak **94.4** @45,727,744 · sef **90.0** · best30 **96.5** @45,842,432

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
| ppo_gae_lambda | 0.92 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 11.2 |
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

![b9ay-lam92-seed1](b9ay-lam92-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 4.0 | 4.0 | 0.0 | 14.0 | 3.5 | 0.0 |  |
| 32768 | 57.93 | 37.82 | 0.0 | 80.0 | 53.83 | 0.0 |  |
| 49152 | 53.8 | 35.3 | 10.0 | 86.0 | 49.205 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.12 | 93.99 | 66.0 | 95.0 | 186.11 | 93.0 |  |
| 49840128 | 93.57 | 93.89 | 26.0 | 95.0 | 186.555 | 94.0 |  |
| 49856512 | 93.3 | 93.91 | 31.0 | 95.0 | 185.245 | 93.0 |  |
| 49872896 | 93.45 | 94.01 | 35.0 | 95.0 | 184.4 | 92.0 |  |
| 49889280 | 93.85 | 93.94 | 28.0 | 95.0 | 186.835 | 94.0 |  |
| 49905664 | 93.44 | 93.91 | 25.0 | 95.0 | 189.41 | 97.0 |  |
| 49922048 | 94.8 | 94.05 | 85.0 | 95.0 | 190.77 | 97.0 |  |
| 49938432 | 94.36 | 94.01 | 67.0 | 95.0 | 190.375 | 97.0 |  |
| 49954816 | 93.91 | 93.94 | 10.0 | 95.0 | 189.88 | 97.0 |  |
| 49971200 | 94.05 | 93.97 | 24.0 | 95.0 | 188.03 | 95.0 |  |
| 49987584 | 93.82 | 93.9 | 16.0 | 95.0 | 188.84 | 96.0 |  |
| 50003968 | 94.08 | 93.95 | 40.0 | 95.0 | 188.105 | 95.0 |  |
