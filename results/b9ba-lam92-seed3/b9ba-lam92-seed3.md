# b9ba-lam92-seed3

step **50,003,968** · 3052 evals · trailing **93.95** · peak **94.44** @35,454,976 · sef **91.2** · best30 **96.7** @22,855,680

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
| ppo_gae_lambda | 0.92 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 11.2 |
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

![b9ba-lam92-seed3](b9ba-lam92-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.03 | 0.03 | 0.0 | 1.0 | -2.54 | 0.0 |  |
| 32768 | 1.25 | 0.64 | 0.0 | 6.0 | 0.75 | 0.0 |  |
| 49152 | 18.76 | 20.05 | 1.0 | 45.0 | 14.975 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.55 | 93.81 | 6.0 | 95.0 | 184.585 | 93.0 |  |
| 49840128 | 94.92 | 93.84 | 90.0 | 95.0 | 191.93 | 98.0 |  |
| 49856512 | 95.0 | 93.96 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49872896 | 94.89 | 93.95 | 84.0 | 95.0 | 192.895 | 99.0 |  |
| 49889280 | 94.88 | 93.93 | 84.0 | 95.0 | 191.845 | 98.0 |  |
| 49905664 | 94.13 | 93.95 | 52.0 | 95.0 | 189.105 | 96.0 |  |
| 49922048 | 94.68 | 93.96 | 83.0 | 95.0 | 190.695 | 97.0 |  |
| 49938432 | 94.89 | 93.96 | 84.0 | 95.0 | 192.895 | 99.0 |  |
| 49954816 | 94.94 | 93.98 | 90.0 | 95.0 | 191.905 | 98.0 |  |
| 49971200 | 93.53 | 93.92 | 68.0 | 95.0 | 182.535 | 90.0 |  |
| 49987584 | 92.9 | 93.92 | 10.0 | 95.0 | 180.955 | 89.0 |  |
| 50003968 | 93.43 | 93.95 | 18.0 | 95.0 | 185.465 | 93.0 |  |
