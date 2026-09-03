# b10ag-g80-seed3

step **35,815,424** · 2186 evals · trailing **59.41** · peak **86.18** @7,913,472 · sef **0.0** · best30 **41.8** @17,498,112

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
| seed | 3 |
| torch_threads | 1 |

![b10ag-g80-seed3](b10ag-g80-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.13 | 0.13 | 0.0 | 1.0 | -0.37 | 0.0 |  |
| 32768 | 1.3 | 0.72 | 0.0 | 6.0 | 0.8 | 0.0 |  |
| 49152 | 17.33 | 6.25 | 0.0 | 47.0 | 15.435 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 35635200 | 63.14 | 59.61 | 16.0 | 95.0 | 71.245 | 11.0 |  |
| 35651584 | 57.06 | 63.66 | 3.0 | 95.0 | 61.775 | 8.0 |  |
| 35667968 | 58.06 | 62.26 | 6.0 | 95.0 | 65.67 | 11.0 |  |
| 35684352 | 64.65 | 61.38 | 10.0 | 95.0 | 67.6 | 6.0 |  |
| 35700736 | 57.73 | 60.38 | 12.0 | 95.0 | 61.315 | 7.0 |  |
| 35717120 | 61.05 | 61.5 | 9.0 | 95.0 | 66.625 | 9.0 |  |
| 35733504 | 62.6 | 60.2 | 15.0 | 95.0 | 68.355 | 9.0 |  |
| 35749888 | 58.59 | 59.94 | 6.0 | 95.0 | 63.305 | 8.0 |  |
| 35766272 | 63.83 | 59.7 | 8.0 | 95.0 | 71.845 | 11.0 |  |
| 35782656 | 60.89 | 61.01 | 9.0 | 95.0 | 67.64 | 10.0 |  |
| 35799040 | 58.27 | 59.23 | 14.0 | 95.0 | 62.08 | 7.0 |  |
| 35815424 | 63.61 | 59.41 | 16.0 | 95.0 | 70.315 | 10.0 |  |
