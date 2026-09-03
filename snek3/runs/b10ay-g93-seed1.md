# b10ay-g93-seed1

step **21,839,872** · 1325 evals · trailing **93.6** · peak **93.99** @16,826,368 · sef **22.7** · best30 **87.7** @17,088,512

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.93 |
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
| ppo_horizon | 11.3 |
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

![b10ay-g93-seed1](b10ay-g93-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 3.93 | 3.93 | 0.0 | 15.0 | 3.43 | 0.0 |  |
| 32768 | 21.41 | 25.08 | 1.0 | 78.0 | 19.65 | 0.0 |  |
| 49152 | 51.66 | 43.46 | 11.0 | 84.0 | 48.055 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 21528576 | 94.26 | 93.13 | 72.0 | 95.0 | 179.33 | 86.0 |  |
| 21544960 | 94.06 | 93.21 | 51.0 | 95.0 | 176.145 | 83.0 |  |
| 21561344 | 93.76 | 93.44 | 29.0 | 95.0 | 179.825 | 87.0 |  |
| 21577728 | 94.47 | 93.47 | 86.0 | 95.0 | 180.535 | 87.0 |  |
| 21594112 | 94.27 | 93.63 | 82.0 | 95.0 | 177.35 | 84.0 |  |
| 21610496 | 94.19 | 93.53 | 75.0 | 95.0 | 181.25 | 88.0 |  |
| 21626880 | 94.42 | 93.58 | 76.0 | 95.0 | 180.485 | 87.0 |  |
| 21643264 | 93.91 | 93.5 | 76.0 | 95.0 | 175.995 | 83.0 |  |
| 21725184 | 93.57 | 93.61 | 22.0 | 95.0 | 177.645 | 85.0 |  |
| 21741568 | 93.69 | 93.62 | 55.0 | 95.0 | 180.75 | 88.0 |  |
| 21823488 | 92.61 | 93.58 | 44.0 | 95.0 | 166.735 | 75.0 |  |
| 21839872 | 93.47 | 93.6 | 42.0 | 95.0 | 172.57 | 80.0 |  |
