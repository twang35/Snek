# b8c-ent003-seed3

step **100,007,936** · 6104 evals · trailing **93.2** · peak **94.55** @95,633,408 · sef **87.5** · best30 **97.4** @95,518,720

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
| fc_layers | (200, 100) |
| graph_eval_episodes | 100 |
| max_steps | 100007936 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.003 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 8 |
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
| seed | 3 |
| torch_threads | 1 |

![b8c-ent003-seed3](b8c-ent003-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 17.4 | 17.4 | 0.0 | 37.0 | 13.03 | 0.0 |  |
| 32768 | 21.38 | 19.39 | 7.0 | 42.0 | 16.425 | 0.0 |  |
| 49152 | 25.28 | 21.35 | 7.0 | 45.0 | 20.28 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99827712 | 92.07 | 93.18 | 5.0 | 95.0 | 184.875 | 94.0 |  |
| 99844096 | 94.16 | 93.28 | 49.0 | 95.0 | 188.095 | 95.0 |  |
| 99860480 | 94.86 | 93.27 | 87.0 | 95.0 | 190.785 | 97.0 |  |
| 99876864 | 93.85 | 93.29 | 40.0 | 95.0 | 189.775 | 97.0 |  |
| 99893248 | 91.73 | 93.19 | 3.0 | 95.0 | 182.59 | 92.0 |  |
| 99909632 | 94.32 | 93.27 | 62.0 | 95.0 | 189.295 | 96.0 |  |
| 99926016 | 93.57 | 93.19 | 42.0 | 95.0 | 186.42 | 94.0 |  |
| 99942400 | 93.27 | 93.17 | 51.0 | 95.0 | 186.075 | 94.0 |  |
| 99958784 | 93.15 | 93.17 | 42.0 | 95.0 | 182.02 | 90.0 |  |
| 99975168 | 94.83 | 93.21 | 85.0 | 95.0 | 191.795 | 98.0 |  |
| 99991552 | 94.64 | 93.2 | 72.0 | 95.0 | 189.525 | 96.0 |  |
| 100007936 | 93.96 | 93.2 | 54.0 | 95.0 | 188.89 | 96.0 |  |
