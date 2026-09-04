# b11ac-lr4e5-seed3

step **31,834,112** · 1941 evals · trailing **91.52** · peak **93.85** @27,000,832 · sef **59.6** · best30 **96.1** @27,066,368

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
| ppo_learning_rate | 4e-05 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b11ac-lr4e5-seed3](b11ac-lr4e5-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.0 | 0.0 | 0.0 | 0.0 | -4.325 | 0.0 |  |
| 32768 | 3.67 | 1.83 | 0.0 | 11.0 | 0.92 | 0.0 |  |
| 49152 | 9.1 | 4.26 | 1.0 | 26.0 | 4.19 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 31621120 | 87.7 | 91.45 | 28.0 | 95.0 | 170.78 | 84.0 |  |
| 31637504 | 92.21 | 91.67 | 43.0 | 95.0 | 181.26 | 90.0 |  |
| 31653888 | 90.23 | 91.65 | 44.0 | 95.0 | 176.295 | 87.0 |  |
| 31670272 | 93.62 | 91.75 | 43.0 | 95.0 | 188.64 | 96.0 |  |
| 31686656 | 93.97 | 91.71 | 54.0 | 95.0 | 188.99 | 96.0 |  |
| 31703040 | 92.79 | 91.73 | 41.0 | 95.0 | 185.82 | 94.0 |  |
| 31752192 | 92.26 | 91.62 | 43.0 | 95.0 | 181.31 | 90.0 |  |
| 31768576 | 90.23 | 91.61 | 18.0 | 95.0 | 177.29 | 88.0 |  |
| 31784960 | 92.41 | 91.69 | 42.0 | 95.0 | 185.44 | 94.0 |  |
| 31801344 | 90.44 | 91.52 | 28.0 | 95.0 | 180.485 | 91.0 |  |
| 31817728 | 88.33 | 91.62 | 28.0 | 95.0 | 175.39 | 88.0 |  |
| 31834112 | 92.07 | 91.52 | 22.0 | 95.0 | 184.105 | 93.0 |  |
