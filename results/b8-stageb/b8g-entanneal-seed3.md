# b8g-entanneal-seed3

step **100,007,936** · 6104 evals · trailing **93.44** · peak **94.62** @20,103,168 · sef **89.8** · best30 **98.0** @20,070,400

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
| ppo_entropy_coef_final | 0.001 |
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

![b8g-entanneal-seed3](b8g-entanneal-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 12.83 | 12.83 | 0.0 | 32.0 | 9.09 | 0.0 |  |
| 32768 | 22.42 | 17.62 | 7.0 | 37.0 | 17.51 | 0.0 |  |
| 49152 | 25.92 | 20.39 | 12.0 | 49.0 | 21.01 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99827712 | 93.96 | 93.45 | 20.0 | 95.0 | 189.93 | 97.0 |  |
| 99844096 | 95.0 | 93.65 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 99860480 | 95.0 | 93.76 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 99876864 | 94.35 | 93.8 | 62.0 | 95.0 | 189.28 | 96.0 |  |
| 99893248 | 94.31 | 93.76 | 26.0 | 95.0 | 192.27 | 99.0 |  |
| 99909632 | 92.69 | 93.58 | 18.0 | 95.0 | 183.595 | 92.0 |  |
| 99926016 | 93.87 | 93.75 | 36.0 | 95.0 | 189.75 | 97.0 |  |
| 99942400 | 93.02 | 93.74 | 25.0 | 95.0 | 188.9 | 97.0 |  |
| 99958784 | 92.71 | 93.73 | 28.0 | 95.0 | 185.47 | 94.0 |  |
| 99975168 | 90.02 | 93.64 | 38.0 | 95.0 | 165.19 | 77.0 |  |
| 99991552 | 89.4 | 93.42 | 32.0 | 95.0 | 161.405 | 74.0 |  |
| 100007936 | 92.66 | 93.44 | 28.0 | 95.0 | 181.44 | 90.0 |  |
