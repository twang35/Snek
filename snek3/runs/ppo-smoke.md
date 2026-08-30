# ppo-smoke

step **507,904** · 31 evals · trailing **62.47** · peak **64.55** @458,752 · sef **0.0** · best30 **0.4** @507,904

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 6 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 500000 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
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
| seed | 1 |
| torch_threads | 1 |

![ppo-smoke](ppo-smoke.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 17.03 | 17.03 | 1.0 | 39.0 | 13.92 | 0.0 |  |
| 32768 | 45.36 | 36.28 | 13.0 | 75.0 | 40.263 | 0.0 |  |
| 49152 | 39.11 | 28.07 | 17.0 | 72.0 | 34.016 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 327680 | 72.55 | 51.37 | 36.0 | 93.0 | 71.089 | 0.0 |  |
| 344064 | 75.17 | 55.35 | 38.0 | 91.0 | 73.683 | 0.0 |  |
| 360448 | 77.24 | 56.34 | 36.0 | 92.0 | 75.665 | 0.0 |  |
| 376832 | 77.67 | 57.27 | 22.0 | 95.0 | 77.022 | 1.0 |  |
| 393216 | 79.2 | 58.18 | 20.0 | 93.0 | 77.848 | 0.0 |  |
| 409600 | 80.55 | 61.28 | 54.0 | 95.0 | 80.116 | 1.0 |  |
| 425984 | 79.1 | 59.84 | 36.0 | 95.0 | 78.785 | 1.0 |  |
| 442368 | 80.41 | 59.07 | 8.0 | 95.0 | 81.947 | 3.0 |  |
| 458752 | 79.43 | 64.55 | 28.0 | 93.0 | 78.149 | 0.0 |  |
| 475136 | 79.36 | 60.57 | 28.0 | 95.0 | 79.059 | 1.0 |  |
| 491520 | 80.7 | 61.95 | 52.0 | 95.0 | 81.38 | 2.0 |  |
| 507904 | 77.62 | 62.47 | 6.0 | 95.0 | 77.307 | 1.0 |  |
