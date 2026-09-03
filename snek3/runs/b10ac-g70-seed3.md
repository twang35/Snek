# b10ac-g70-seed3

step **42,106,880** · 2566 evals · trailing **38.8** · peak **62.92** @1,425,408 · sef **0.0** · best30 **5.1** @2,129,920

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
| seed | 3 |
| torch_threads | 1 |

![b10ac-g70-seed3](b10ac-g70-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.13 | 0.13 | 0.0 | 2.0 | -0.37 | 0.0 |  |
| 32768 | 1.42 | 0.77 | 0.0 | 7.0 | 0.92 | 0.0 |  |
| 49152 | 26.45 | 15.25 | 0.0 | 63.0 | 22.395 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 41861120 | 38.59 | 38.85 | 12.0 | 79.0 | 33.635 | 0.0 |  |
| 41877504 | 39.02 | 38.82 | 11.0 | 81.0 | 34.065 | 0.0 |  |
| 41893888 | 39.83 | 38.8 | 13.0 | 89.0 | 34.875 | 0.0 |  |
| 41910272 | 39.6 | 38.9 | 14.0 | 76.0 | 34.6 | 0.0 |  |
| 41926656 | 39.6 | 38.78 | 12.0 | 86.0 | 34.6 | 0.0 |  |
| 41943040 | 39.63 | 38.78 | 5.0 | 91.0 | 34.9 | 0.0 |  |
| 41959424 | 36.23 | 38.81 | 9.0 | 87.0 | 31.275 | 0.0 |  |
| 41992192 | 39.73 | 38.8 | 14.0 | 85.0 | 34.82 | 0.0 |  |
| 42041344 | 37.45 | 38.72 | 13.0 | 87.0 | 32.585 | 0.0 |  |
| 42074112 | 37.14 | 38.73 | 6.0 | 75.0 | 32.14 | 0.0 |  |
| 42090496 | 36.48 | 38.64 | 14.0 | 93.0 | 31.66 | 0.0 |  |
| 42106880 | 41.52 | 38.8 | 16.0 | 85.0 | 36.565 | 0.0 |  |
