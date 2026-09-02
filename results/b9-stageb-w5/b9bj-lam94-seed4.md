# b9bj-lam94-seed4

step **50,003,968** · 3052 evals · trailing **93.93** · peak **94.36** @25,034,752 · sef **89.8** · best30 **96.9** @21,168,128

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
| seed | 4 |
| torch_threads | 1 |

![b9bj-lam94-seed4](b9bj-lam94-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 3.25 | 3.25 | 0.0 | 10.0 | 0.905 | 0.0 |  |
| 32768 | 19.59 | 27.63 | 1.0 | 44.0 | 15.895 | 0.0 |  |
| 49152 | 29.26 | 27.91 | 8.0 | 66.0 | 24.26 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.86 | 94.14 | 65.0 | 95.0 | 171.915 | 80.0 |  |
| 49840128 | 92.54 | 94.05 | 38.0 | 95.0 | 180.55 | 89.0 |  |
| 49856512 | 92.35 | 93.97 | 4.0 | 95.0 | 176.425 | 85.0 |  |
| 49872896 | 92.99 | 94.1 | 14.0 | 95.0 | 184.98 | 93.0 |  |
| 49889280 | 94.17 | 93.99 | 63.0 | 95.0 | 189.19 | 96.0 |  |
| 49905664 | 94.32 | 93.99 | 76.0 | 95.0 | 187.35 | 94.0 |  |
| 49922048 | 93.7 | 93.88 | 53.0 | 95.0 | 185.69 | 93.0 |  |
| 49938432 | 94.63 | 93.92 | 79.0 | 95.0 | 188.655 | 95.0 |  |
| 49954816 | 93.43 | 93.93 | 42.0 | 95.0 | 185.42 | 93.0 |  |
| 49971200 | 94.85 | 93.99 | 80.0 | 95.0 | 192.855 | 99.0 |  |
| 49987584 | 94.28 | 93.93 | 77.0 | 95.0 | 186.315 | 93.0 |  |
| 50003968 | 94.42 | 93.93 | 77.0 | 95.0 | 186.41 | 93.0 |  |
