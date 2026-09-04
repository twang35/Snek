# b10ck-g100-seed3

step **50,003,968** · 3052 evals · trailing **43.46** · peak **94.63** @30,654,464 · sef **25.6** · best30 **98.4** @30,670,848

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 1.0 |
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
| ppo_horizon | 50.0 |
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

![b10ck-g100-seed3](b10ck-g100-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.03 | 0.03 | 0.0 | 1.0 | -4.16 | 0.0 |  |
| 32768 | 1.0 | 0.52 | 0.0 | 5.0 | 0.5 | 0.0 |  |
| 49152 | 14.16 | 5.06 | 0.0 | 28.0 | 9.61 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 41.13 | 42.11 | 12.0 | 55.0 | 36.22 | 0.0 |  |
| 49840128 | 42.44 | 41.56 | 18.0 | 58.0 | 37.8 | 0.0 |  |
| 49856512 | 41.25 | 41.78 | 19.0 | 57.0 | 36.43 | 0.0 |  |
| 49872896 | 41.11 | 41.77 | 16.0 | 60.0 | 36.695 | 0.0 |  |
| 49889280 | 44.15 | 41.89 | 6.0 | 63.0 | 39.78 | 0.0 |  |
| 49905664 | 44.54 | 42.02 | 6.0 | 64.0 | 40.305 | 0.0 |  |
| 49922048 | 41.43 | 41.95 | 4.0 | 64.0 | 37.33 | 0.0 |  |
| 49938432 | 43.28 | 42.0 | 2.0 | 63.0 | 38.955 | 0.0 |  |
| 49954816 | 43.35 | 42.17 | 6.0 | 62.0 | 39.34 | 0.0 |  |
| 49971200 | 50.17 | 42.46 | 22.0 | 67.0 | 45.575 | 0.0 |  |
| 49987584 | 50.03 | 43.05 | 28.0 | 67.0 | 45.21 | 0.0 |  |
| 50003968 | 48.18 | 43.46 | 18.0 | 77.0 | 43.405 | 0.0 |  |
