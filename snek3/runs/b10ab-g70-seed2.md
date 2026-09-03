# b10ab-g70-seed2

step **42,205,184** · 2571 evals · trailing **41.63** · peak **62.78** @802,816 · sef **0.0** · best30 **5.2** @3,538,944

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.7 |
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
| ppo_horizon | 3.2 |
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

![b10ab-g70-seed2](b10ab-g70-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 3.49 | 3.49 | 0.0 | 8.0 | -1.06 | 0.0 |  |
| 32768 | 16.45 | 9.97 | 0.0 | 38.0 | 12.845 | 0.0 |  |
| 49152 | 26.85 | 20.73 | 0.0 | 62.0 | 22.345 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 41943040 | 40.03 | 40.88 | 10.0 | 95.0 | 36.25 | 1.0 |  |
| 41959424 | 41.47 | 41.12 | 6.0 | 93.0 | 36.65 | 0.0 |  |
| 41975808 | 41.81 | 41.23 | 12.0 | 93.0 | 37.035 | 0.0 |  |
| 41992192 | 41.46 | 41.3 | 11.0 | 93.0 | 36.595 | 0.0 |  |
| 42008576 | 40.61 | 41.27 | 19.0 | 95.0 | 36.695 | 1.0 |  |
| 42024960 | 40.3 | 41.38 | 12.0 | 95.0 | 36.475 | 1.0 |  |
| 42041344 | 42.5 | 40.88 | 16.0 | 95.0 | 39.625 | 2.0 |  |
| 42057728 | 42.38 | 41.63 | 10.0 | 95.0 | 38.555 | 1.0 |  |
| 42074112 | 45.06 | 41.56 | 11.0 | 93.0 | 40.375 | 0.0 |  |
| 42090496 | 44.56 | 41.62 | 11.0 | 95.0 | 40.69 | 1.0 |  |
| 42106880 | 43.28 | 41.62 | 17.0 | 93.0 | 38.37 | 0.0 |  |
| 42205184 | 44.18 | 41.63 | 16.0 | 95.0 | 41.485 | 2.0 |  |
