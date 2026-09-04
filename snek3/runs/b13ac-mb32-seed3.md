# b13ac-mb32-seed3

step **50,003,968** · 3052 evals · trailing **93.77** · peak **94.1** @49,577,984 · sef **82.7** · best30 **97.2** @12,189,696

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
| ppo_minibatch | 32 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b13ac-mb32-seed3](b13ac-mb32-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.01 | 0.01 | 0.0 | 1.0 | -4.99 | 0.0 |  |
| 32768 | 2.08 | 1.04 | 0.0 | 11.0 | 1.445 | 0.0 |  |
| 49152 | 17.1 | 6.4 | 1.0 | 41.0 | 13.045 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.69 | 94.01 | 12.0 | 95.0 | 187.58 | 95.0 |  |
| 49840128 | 93.16 | 93.9 | 40.0 | 95.0 | 182.935 | 91.0 |  |
| 49856512 | 93.55 | 93.93 | 19.0 | 95.0 | 186.445 | 94.0 |  |
| 49872896 | 93.55 | 93.92 | 6.0 | 95.0 | 186.58 | 94.0 |  |
| 49889280 | 93.65 | 93.81 | 22.0 | 95.0 | 185.55 | 93.0 |  |
| 49905664 | 93.79 | 93.95 | 49.0 | 95.0 | 185.645 | 93.0 |  |
| 49922048 | 94.26 | 93.73 | 45.0 | 95.0 | 190.23 | 97.0 |  |
| 49938432 | 94.64 | 93.8 | 67.0 | 95.0 | 190.565 | 97.0 |  |
| 49954816 | 93.8 | 93.79 | 55.0 | 95.0 | 187.645 | 95.0 |  |
| 49971200 | 91.45 | 93.83 | 32.0 | 95.0 | 174.895 | 85.0 |  |
| 49987584 | 92.15 | 93.71 | 49.0 | 95.0 | 179.755 | 89.0 |  |
| 50003968 | 93.7 | 93.77 | 56.0 | 95.0 | 186.46 | 94.0 |  |
