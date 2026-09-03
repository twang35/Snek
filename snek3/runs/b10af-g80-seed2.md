# b10af-g80-seed2

step **37,011,456** · 2259 evals · trailing **50.22** · peak **83.56** @16,154,624 · sef **0.0** · best30 **24.4** @14,761,984

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
| seed | 2 |
| torch_threads | 1 |

![b10af-g80-seed2](b10af-g80-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 3.01 | 3.01 | 0.0 | 8.0 | -1.09 | 0.0 |  |
| 32768 | 15.88 | 9.45 | 0.0 | 29.0 | 11.645 | 0.0 |  |
| 49152 | 32.88 | 21.81 | 0.0 | 68.0 | 28.105 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 36831232 | 48.87 | 50.06 | 15.0 | 95.0 | 53.815 | 9.0 |  |
| 36847616 | 46.95 | 50.18 | 10.0 | 95.0 | 48.73 | 6.0 |  |
| 36864000 | 47.41 | 51.59 | 6.0 | 95.0 | 51.27 | 8.0 |  |
| 36880384 | 44.32 | 51.71 | 7.0 | 95.0 | 47.095 | 7.0 |  |
| 36896768 | 45.68 | 51.28 | 10.0 | 95.0 | 46.465 | 5.0 |  |
| 36913152 | 51.57 | 50.96 | 11.0 | 95.0 | 54.75 | 7.0 |  |
| 36929536 | 46.51 | 51.05 | 3.0 | 95.0 | 50.415 | 8.0 |  |
| 36945920 | 54.66 | 51.82 | 12.0 | 95.0 | 61.185 | 10.0 |  |
| 36962304 | 51.86 | 51.96 | 8.0 | 95.0 | 57.21 | 9.0 |  |
| 36978688 | 50.78 | 50.4 | 9.0 | 95.0 | 57.17 | 10.0 |  |
| 36995072 | 53.95 | 50.24 | 5.0 | 95.0 | 59.435 | 9.0 |  |
| 37011456 | 54.21 | 50.22 | 13.0 | 95.0 | 65.665 | 15.0 |  |
