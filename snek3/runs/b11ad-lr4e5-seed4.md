# b11ad-lr4e5-seed4

step **31,309,824** · 1911 evals · trailing **92.79** · peak **93.99** @24,133,632 · sef **61.0** · best30 **95.8** @24,608,768

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
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
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 4e-05 |
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

![b11ad-lr4e5-seed4](b11ad-lr4e5-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.02 | 0.02 | 0.0 | 1.0 | -0.525 | 0.0 |  |
| 32768 | 8.34 | 4.18 | 2.0 | 18.0 | 3.34 | 0.0 |  |
| 49152 | 9.65 | 6.0 | 4.0 | 24.0 | 4.65 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 31129600 | 92.32 | 91.59 | 49.0 | 95.0 | 181.37 | 90.0 |  |
| 31145984 | 93.25 | 92.74 | 43.0 | 95.0 | 186.28 | 94.0 |  |
| 31162368 | 92.69 | 92.71 | 55.0 | 95.0 | 183.73 | 92.0 |  |
| 31178752 | 92.65 | 92.8 | 55.0 | 95.0 | 184.685 | 93.0 |  |
| 31195136 | 91.77 | 92.6 | 53.0 | 95.0 | 180.82 | 90.0 |  |
| 31211520 | 93.13 | 92.78 | 55.0 | 95.0 | 184.125 | 92.0 |  |
| 31227904 | 92.92 | 92.76 | 43.0 | 95.0 | 184.955 | 93.0 |  |
| 31244288 | 94.06 | 92.37 | 54.0 | 95.0 | 190.075 | 97.0 |  |
| 31260672 | 92.04 | 92.71 | 49.0 | 95.0 | 180.095 | 89.0 |  |
| 31277056 | 92.17 | 91.88 | 44.0 | 95.0 | 181.22 | 90.0 |  |
| 31293440 | 94.26 | 92.46 | 66.0 | 95.0 | 188.24 | 95.0 |  |
| 31309824 | 93.05 | 92.79 | 44.0 | 95.0 | 186.08 | 94.0 |  |
