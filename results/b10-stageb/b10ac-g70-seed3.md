# b10ac-g70-seed3

step **50,003,968** · 3052 evals · trailing **39.59** · peak **62.92** @1,425,408 · sef **0.0** · best30 **5.1** @2,129,920

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
| 49823744 | 41.19 | 39.69 | 15.0 | 85.0 | 36.28 | 0.0 |  |
| 49840128 | 39.84 | 39.64 | 17.0 | 93.0 | 34.885 | 0.0 |  |
| 49856512 | 42.65 | 39.62 | 12.0 | 87.0 | 37.83 | 0.0 |  |
| 49872896 | 40.25 | 39.67 | 17.0 | 89.0 | 35.385 | 0.0 |  |
| 49889280 | 39.92 | 39.46 | 11.0 | 80.0 | 34.965 | 0.0 |  |
| 49905664 | 37.93 | 39.66 | 14.0 | 95.0 | 34.06 | 1.0 |  |
| 49922048 | 38.55 | 39.51 | 15.0 | 86.0 | 33.595 | 0.0 |  |
| 49938432 | 37.31 | 39.52 | 17.0 | 79.0 | 32.355 | 0.0 |  |
| 49954816 | 36.92 | 39.57 | 12.0 | 70.0 | 31.92 | 0.0 |  |
| 49971200 | 42.6 | 39.54 | 11.0 | 85.0 | 37.6 | 0.0 |  |
| 49987584 | 38.92 | 39.5 | 14.0 | 89.0 | 34.01 | 0.0 |  |
| 50003968 | 38.51 | 39.59 | 7.0 | 89.0 | 33.69 | 0.0 |  |
