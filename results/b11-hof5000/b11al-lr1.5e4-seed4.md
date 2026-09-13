# b11al-lr1.5e4-seed4

step **50,003,968** · 3052 evals · trailing **94.03** · peak **94.61** @31,064,064 · sef **87.7** · best30 **98.4** @31,375,360

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
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.00015 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b11al-lr1.5e4-seed4](b11al-lr1.5e4-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.59 | 0.59 | 0.0 | 2.0 | -1.89 | 0.0 |  |
| 32768 | 11.73 | 6.16 | 2.0 | 26.0 | 6.73 | 0.0 |  |
| 49152 | 20.95 | 11.09 | 9.0 | 42.0 | 15.95 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.32 | 94.21 | 12.0 | 95.0 | 186.35 | 94.0 |  |
| 49840128 | 93.17 | 94.28 | 30.0 | 95.0 | 184.21 | 92.0 |  |
| 49856512 | 94.82 | 94.25 | 83.0 | 95.0 | 191.83 | 98.0 |  |
| 49872896 | 93.85 | 94.25 | 8.0 | 95.0 | 190.86 | 98.0 |  |
| 49889280 | 93.49 | 94.21 | 55.0 | 95.0 | 186.52 | 94.0 |  |
| 49905664 | 93.75 | 94.24 | 60.0 | 95.0 | 184.79 | 92.0 |  |
| 49922048 | 93.78 | 94.11 | 52.0 | 95.0 | 184.775 | 92.0 |  |
| 49938432 | 92.47 | 94.14 | 69.0 | 95.0 | 175.55 | 84.0 |  |
| 49954816 | 93.2 | 94.08 | 66.0 | 95.0 | 179.265 | 87.0 |  |
| 49971200 | 93.52 | 94.05 | 70.0 | 95.0 | 181.575 | 89.0 |  |
| 49987584 | 94.2 | 94.11 | 57.0 | 95.0 | 190.215 | 97.0 |  |
| 50003968 | 94.11 | 94.03 | 53.0 | 95.0 | 190.125 | 97.0 |  |
