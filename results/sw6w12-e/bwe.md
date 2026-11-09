# bwe

step **262,389,760** · 438 evals · trailing **93.62** · peak **94.09** @259,604,480 · sef **98.2** · best30 **96.3** @259,506,176

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
| max_steps | 262389760 |
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

Resumed at 255,213,568, 256,409,600, 257,605,632, 258,801,664, 259,997,696, 261,193,728

![bwe](bwe.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 255229952 | 93.15 | 92.41 | 60.0 | 95.0 | 179.985 | 88.0 |  |
| 255246336 | 92.0 | 91.99 | 22.0 | 95.0 | 180.735 | 90.0 |  |
| 255262720 | 91.44 | 92.03 | 24.0 | 95.0 | 176.285 | 86.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 262209536 | 94.77 | 93.12 | 74.0 | 95.0 | 191.735 | 98.0 |  |
| 262225920 | 94.64 | 92.93 | 59.0 | 95.0 | 192.6 | 99.0 |  |
| 262242304 | 94.47 | 93.67 | 46.0 | 95.0 | 191.435 | 98.0 |  |
| 262258688 | 94.11 | 93.75 | 6.0 | 95.0 | 192.115 | 99.0 |  |
| 262275072 | 94.05 | 93.8 | 5.0 | 95.0 | 191.06 | 98.0 |  |
| 262291456 | 94.82 | 93.3 | 84.0 | 95.0 | 191.83 | 98.0 |  |
| 262307840 | 95.0 | 93.46 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 262324224 | 94.83 | 93.71 | 86.0 | 95.0 | 191.75 | 98.0 |  |
| 262340608 | 94.08 | 93.73 | 33.0 | 95.0 | 190.05 | 97.0 |  |
| 262356992 | 92.54 | 93.69 | 18.0 | 95.0 | 184.395 | 93.0 |  |
| 262373376 | 91.53 | 93.69 | 21.0 | 95.0 | 183.34 | 93.0 |  |
| 262389760 | 92.05 | 93.62 | 26.0 | 95.0 | 183.905 | 93.0 |  |
