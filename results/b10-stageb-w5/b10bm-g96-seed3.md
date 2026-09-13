# b10bm-g96-seed3

step **50,003,968** · 3052 evals · trailing **92.25** · peak **94.2** @39,632,896 · sef **56.3** · best30 **94.6** @21,397,504

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.96 |
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
| ppo_horizon | 16.9 |
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

![b10bm-g96-seed3](b10bm-g96-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.07 | 0.07 | 0.0 | 1.0 | -0.52 | 0.0 |  |
| 32768 | 0.34 | 0.21 | 0.0 | 3.0 | -0.16 | 0.0 |  |
| 49152 | 2.6 | 1.0 | 0.0 | 18.0 | 1.965 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.67 | 92.67 | 70.0 | 95.0 | 163.81 | 72.0 |  |
| 49840128 | 93.26 | 92.59 | 73.0 | 95.0 | 172.36 | 80.0 |  |
| 49856512 | 92.01 | 92.6 | 65.0 | 95.0 | 159.125 | 68.0 |  |
| 49872896 | 92.51 | 92.44 | 36.0 | 95.0 | 157.68 | 66.0 |  |
| 49889280 | 91.33 | 92.49 | 8.0 | 95.0 | 160.48 | 70.0 |  |
| 49905664 | 92.62 | 92.33 | 37.0 | 95.0 | 165.75 | 74.0 |  |
| 49922048 | 91.3 | 92.43 | 12.0 | 95.0 | 162.44 | 72.0 |  |
| 49938432 | 93.05 | 92.29 | 77.0 | 95.0 | 168.17 | 76.0 |  |
| 49954816 | 92.32 | 92.35 | 72.0 | 95.0 | 162.465 | 71.0 |  |
| 49971200 | 92.42 | 92.41 | 76.0 | 95.0 | 144.655 | 53.0 |  |
| 49987584 | 91.89 | 92.3 | 52.0 | 95.0 | 159.05 | 68.0 |  |
| 50003968 | 91.29 | 92.25 | 56.0 | 95.0 | 144.52 | 54.0 |  |
