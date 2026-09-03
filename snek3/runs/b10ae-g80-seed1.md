# b10ae-g80-seed1

step **36,044,800** · 2200 evals · trailing **57.12** · peak **91.34** @14,319,616 · sef **0.0** · best30 **49.7** @14,336,000

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.8 |
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
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 4.6 |
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

![b10ae-g80-seed1](b10ae-g80-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 7.09 | 7.09 | 0.0 | 18.0 | 6.59 | 0.0 |  |
| 32768 | 46.45 | 40.16 | 0.0 | 86.0 | 43.34 | 0.0 |  |
| 49152 | 72.72 | 44.63 | 30.0 | 93.0 | 70.78 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 35864576 | 51.12 | 56.77 | 7.0 | 95.0 | 61.22 | 14.0 |  |
| 35880960 | 56.03 | 58.18 | 15.0 | 95.0 | 71.24 | 19.0 |  |
| 35897344 | 60.53 | 58.83 | 16.0 | 95.0 | 72.89 | 16.0 |  |
| 35913728 | 58.84 | 57.61 | 13.0 | 95.0 | 74.41 | 19.0 |  |
| 35930112 | 60.54 | 57.59 | 14.0 | 95.0 | 74.075 | 17.0 |  |
| 35946496 | 56.66 | 57.3 | 14.0 | 95.0 | 70.92 | 18.0 |  |
| 35962880 | 53.32 | 58.47 | 6.0 | 95.0 | 66.315 | 17.0 |  |
| 35979264 | 54.73 | 57.11 | 12.0 | 95.0 | 67.905 | 17.0 |  |
| 35995648 | 53.51 | 57.14 | 12.0 | 95.0 | 69.715 | 20.0 |  |
| 36012032 | 57.34 | 57.47 | 19.0 | 95.0 | 73.635 | 20.0 |  |
| 36028416 | 55.24 | 57.51 | 15.0 | 95.0 | 65.43 | 14.0 |  |
| 36044800 | 49.5 | 57.12 | 14.0 | 95.0 | 51.28 | 6.0 |  |
