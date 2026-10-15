# b8o-lam95-seed3

step **100,007,936** · 6104 evals · trailing **93.92** · peak **94.29** @73,318,400 · sef **90.9** · best30 **96.0** @6,815,744

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
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 8 |
| ppo_gae_lambda | 0.95 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 16.8 |
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

![b8o-lam95-seed3](b8o-lam95-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 17.35 | 17.35 | 0.0 | 35.0 | 12.755 | 0.0 |  |
| 32768 | 21.6 | 19.48 | 0.0 | 44.0 | 17.275 | 0.0 |  |
| 49152 | 28.38 | 22.44 | 11.0 | 48.0 | 23.425 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99827712 | 94.2 | 93.91 | 36.0 | 95.0 | 190.125 | 97.0 |  |
| 99844096 | 93.39 | 93.87 | 71.0 | 95.0 | 174.755 | 83.0 |  |
| 99860480 | 93.88 | 93.93 | 36.0 | 95.0 | 184.56 | 92.0 |  |
| 99876864 | 94.41 | 93.99 | 85.0 | 95.0 | 185.09 | 92.0 |  |
| 99893248 | 92.0 | 93.91 | 1.0 | 95.0 | 185.935 | 95.0 |  |
| 99909632 | 93.9 | 93.89 | 3.0 | 95.0 | 188.785 | 96.0 |  |
| 99926016 | 92.04 | 93.84 | 1.0 | 95.0 | 184.98 | 94.0 |  |
| 99942400 | 94.28 | 93.84 | 52.0 | 95.0 | 188.08 | 95.0 |  |
| 99958784 | 95.0 | 93.89 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 99975168 | 94.35 | 93.97 | 72.0 | 95.0 | 188.33 | 95.0 |  |
| 99991552 | 94.47 | 93.96 | 81.0 | 95.0 | 187.23 | 94.0 |  |
| 100007936 | 93.63 | 93.92 | 15.0 | 95.0 | 186.525 | 94.0 |  |
