# b10ad-g70-seed4

step **41,779,200** · 2544 evals · trailing **38.79** · peak **60.16** @770,048 · sef **0.0** · best30 **2.8** @933,888

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
| seed | 4 |
| torch_threads | 1 |

![b10ad-g70-seed4](b10ad-g70-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.44 | 0.44 | 0.0 | 7.0 | -0.15 | 0.0 |  |
| 32768 | 17.74 | 33.55 | 0.0 | 65.0 | 15.62 | 0.0 |  |
| 49152 | 57.03 | 36.9 | 1.0 | 90.0 | 53.29 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 41500672 | 39.86 | 39.65 | 11.0 | 95.0 | 36.035 | 1.0 |  |
| 41517056 | 37.45 | 39.49 | 5.0 | 82.0 | 32.45 | 0.0 |  |
| 41533440 | 39.76 | 39.06 | 13.0 | 87.0 | 34.805 | 0.0 |  |
| 41549824 | 35.95 | 39.54 | 10.0 | 84.0 | 30.95 | 0.0 |  |
| 41566208 | 40.25 | 38.95 | 10.0 | 89.0 | 35.295 | 0.0 |  |
| 41582592 | 39.9 | 38.91 | 12.0 | 91.0 | 35.035 | 0.0 |  |
| 41598976 | 37.98 | 38.93 | 17.0 | 95.0 | 34.155 | 1.0 |  |
| 41615360 | 41.73 | 39.0 | 12.0 | 93.0 | 36.82 | 0.0 |  |
| 41730048 | 40.22 | 38.87 | 11.0 | 78.0 | 35.22 | 0.0 |  |
| 41746432 | 37.65 | 38.75 | 15.0 | 79.0 | 32.695 | 0.0 |  |
| 41762816 | 35.98 | 38.67 | 6.0 | 71.0 | 31.025 | 0.0 |  |
| 41779200 | 39.0 | 38.79 | 14.0 | 80.0 | 34.0 | 0.0 |  |
