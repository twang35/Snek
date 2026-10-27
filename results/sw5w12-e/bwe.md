# bwe

step **261,193,728** · 365 evals · trailing **94.01** · peak **94.09** @259,604,480 · sef **98.4** · best30 **96.3** @259,506,176

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 12 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 261193728 |
| min_checkpoint_score | 0.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 8 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 8 |
| torch_threads | 1 |

## Resumes

Resumed at 255,213,568, 256,409,600, 257,605,632, 258,801,664, 259,997,696

![bwe](bwe.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 255229952 | 93.15 | 92.41 | 60.0 | 95.0 | 179.985 | 88.0 |  |
| 255246336 | 92.0 | 91.99 | 22.0 | 95.0 | 180.735 | 90.0 |  |
| 255262720 | 91.44 | 92.03 | 24.0 | 95.0 | 176.285 | 86.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 261013504 | 92.32 | 93.92 | 31.0 | 95.0 | 181.01 | 90.0 |  |
| 261029888 | 94.24 | 94.06 | 44.0 | 95.0 | 189.125 | 96.0 |  |
| 261046272 | 93.19 | 94.05 | 6.0 | 95.0 | 187.08 | 95.0 |  |
| 261062656 | 92.51 | 93.96 | 36.0 | 95.0 | 178.035 | 87.0 |  |
| 261079040 | 92.6 | 93.97 | 7.0 | 95.0 | 182.42 | 91.0 |  |
| 261095424 | 93.46 | 94.01 | 7.0 | 95.0 | 187.35 | 95.0 |  |
| 261111808 | 93.97 | 93.97 | 28.0 | 95.0 | 189.85 | 97.0 |  |
| 261128192 | 93.31 | 93.88 | 14.0 | 95.0 | 187.155 | 95.0 |  |
| 261144576 | 92.8 | 93.82 | 4.0 | 95.0 | 184.61 | 93.0 |  |
| 261160960 | 94.49 | 93.85 | 50.0 | 95.0 | 190.37 | 97.0 |  |
| 261177344 | 94.05 | 93.85 | 34.0 | 95.0 | 188.89 | 96.0 |  |
| 261193728 | 94.9 | 94.01 | 85.0 | 95.0 | 192.86 | 99.0 |  |
