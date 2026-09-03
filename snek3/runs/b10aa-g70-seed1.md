# b10aa-g70-seed1

step **43,417,600** · 2646 evals · trailing **39.23** · peak **63.35** @950,272 · sef **0.0** · best30 **2.7** @950,272

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
| seed | 1 |
| torch_threads | 1 |

![b10aa-g70-seed1](b10aa-g70-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 7.66 | 7.66 | 0.0 | 23.0 | 7.16 | 0.0 |  |
| 32768 | 54.52 | 46.03 | 0.0 | 84.0 | 50.33 | 0.0 |  |
| 49152 | 68.03 | 41.91 | 30.0 | 90.0 | 66.045 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 43171840 | 38.84 | 38.62 | 5.0 | 83.0 | 33.84 | 0.0 |  |
| 43188224 | 36.05 | 38.61 | 15.0 | 91.0 | 31.14 | 0.0 |  |
| 43204608 | 35.75 | 38.81 | 8.0 | 85.0 | 30.75 | 0.0 |  |
| 43220992 | 38.94 | 39.0 | 11.0 | 89.0 | 33.985 | 0.0 |  |
| 43237376 | 38.89 | 38.97 | 8.0 | 95.0 | 34.93 | 1.0 |  |
| 43253760 | 37.55 | 38.81 | 11.0 | 77.0 | 32.55 | 0.0 |  |
| 43270144 | 41.67 | 39.05 | 13.0 | 85.0 | 36.715 | 0.0 |  |
| 43286528 | 40.75 | 39.2 | 13.0 | 95.0 | 37.875 | 2.0 |  |
| 43368448 | 39.56 | 39.22 | 16.0 | 82.0 | 34.56 | 0.0 |  |
| 43384832 | 38.04 | 38.98 | 16.0 | 72.0 | 33.04 | 0.0 |  |
| 43401216 | 39.01 | 39.2 | 6.0 | 93.0 | 34.1 | 0.0 |  |
| 43417600 | 39.52 | 39.23 | 12.0 | 90.0 | 34.52 | 0.0 |  |
