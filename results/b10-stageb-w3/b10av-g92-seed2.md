# b10av-g92-seed2

step **50,003,968** · 3052 evals · trailing **91.18** · peak **93.8** @38,617,088 · sef **10.8** · best30 **86.6** @38,420,480

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.92 |
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
| ppo_horizon | 10.2 |
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

![b10av-g92-seed2](b10av-g92-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.9 | 2.9 | 0.0 | 7.0 | -0.885 | 0.0 |  |
| 32768 | 10.62 | 6.76 | 0.0 | 19.0 | 5.935 | 0.0 |  |
| 49152 | 26.04 | 13.19 | 0.0 | 51.0 | 21.31 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 91.45 | 89.55 | 28.0 | 95.0 | 159.47 | 69.0 |  |
| 49840128 | 88.67 | 89.54 | 8.0 | 95.0 | 160.49 | 73.0 |  |
| 49856512 | 91.67 | 89.74 | 29.0 | 95.0 | 162.675 | 72.0 |  |
| 49872896 | 89.8 | 90.38 | 26.0 | 95.0 | 161.71 | 73.0 |  |
| 49889280 | 92.82 | 90.13 | 41.0 | 95.0 | 165.86 | 74.0 |  |
| 49905664 | 91.05 | 90.61 | 18.0 | 95.0 | 160.925 | 71.0 |  |
| 49922048 | 93.76 | 90.74 | 59.0 | 95.0 | 174.85 | 82.0 |  |
| 49938432 | 92.4 | 90.96 | 37.0 | 95.0 | 161.505 | 70.0 |  |
| 49954816 | 93.15 | 90.82 | 19.0 | 95.0 | 169.085 | 77.0 |  |
| 49971200 | 93.76 | 91.35 | 82.0 | 95.0 | 163.905 | 71.0 |  |
| 49987584 | 92.35 | 90.54 | 26.0 | 95.0 | 171.36 | 80.0 |  |
| 50003968 | 94.25 | 91.18 | 84.0 | 95.0 | 176.245 | 83.0 |  |
