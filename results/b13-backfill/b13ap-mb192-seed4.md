# b13ap-mb192-seed4

step **50,003,968** · 3052 evals · trailing **93.41** · peak **94.46** @39,190,528 · sef **90.2** · best30 **97.9** @18,137,088

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
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 192 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b13ap-mb192-seed4](b13ap-mb192-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.24 | 0.24 | 0.0 | 2.0 | -0.485 | 0.0 |  |
| 32768 | 18.53 | 14.08 | 2.0 | 35.0 | 13.8 | 0.0 |  |
| 49152 | 23.47 | 11.85 | 5.0 | 41.0 | 18.47 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.18 | 93.79 | 80.0 | 95.0 | 185.175 | 92.0 |  |
| 49840128 | 91.99 | 93.96 | 4.0 | 95.0 | 178.055 | 87.0 |  |
| 49856512 | 93.17 | 93.76 | 67.0 | 95.0 | 178.24 | 86.0 |  |
| 49872896 | 93.79 | 93.83 | 62.0 | 95.0 | 185.825 | 93.0 |  |
| 49889280 | 92.99 | 93.74 | 50.0 | 95.0 | 181.995 | 90.0 |  |
| 49905664 | 91.17 | 93.63 | 22.0 | 95.0 | 174.16 | 84.0 |  |
| 49922048 | 90.72 | 93.5 | 64.0 | 95.0 | 161.815 | 72.0 |  |
| 49938432 | 92.28 | 93.41 | 32.0 | 95.0 | 175.36 | 84.0 |  |
| 49954816 | 93.4 | 93.47 | 14.0 | 95.0 | 180.46 | 88.0 |  |
| 49971200 | 94.2 | 93.48 | 66.0 | 95.0 | 187.23 | 94.0 |  |
| 49987584 | 92.82 | 93.44 | 29.0 | 95.0 | 181.735 | 90.0 |  |
| 50003968 | 93.49 | 93.41 | 16.0 | 95.0 | 183.49 | 91.0 |  |
