# b13am-mb192-seed1

step **50,003,968** · 3052 evals · trailing **93.08** · peak **94.65** @33,734,656 · sef **92.8** · best30 **98.1** @33,734,656

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
| seed | 1 |
| torch_threads | 1 |

![b13am-mb192-seed1](b13am-mb192-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 18.93 | 18.93 | 0.0 | 35.0 | 14.02 | 0.0 |  |
| 32768 | 27.55 | 23.24 | 1.0 | 61.0 | 22.91 | 0.0 |  |
| 49152 | 23.99 | 23.49 | 3.0 | 53.0 | 19.035 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.15 | 93.05 | 69.0 | 95.0 | 187.18 | 94.0 |  |
| 49840128 | 94.47 | 93.14 | 74.0 | 95.0 | 187.5 | 94.0 |  |
| 49856512 | 94.42 | 93.03 | 80.0 | 95.0 | 187.45 | 94.0 |  |
| 49872896 | 93.89 | 93.14 | 67.0 | 95.0 | 186.92 | 94.0 |  |
| 49889280 | 94.28 | 93.12 | 70.0 | 95.0 | 188.305 | 95.0 |  |
| 49905664 | 93.32 | 92.97 | 22.0 | 95.0 | 186.305 | 94.0 |  |
| 49922048 | 94.54 | 92.97 | 78.0 | 95.0 | 188.565 | 95.0 |  |
| 49938432 | 93.93 | 93.0 | 78.0 | 95.0 | 183.975 | 91.0 |  |
| 49954816 | 94.36 | 93.07 | 67.0 | 95.0 | 188.385 | 95.0 |  |
| 49971200 | 94.48 | 93.04 | 70.0 | 95.0 | 190.495 | 97.0 |  |
| 49987584 | 94.35 | 93.1 | 80.0 | 95.0 | 185.39 | 92.0 |  |
| 50003968 | 93.69 | 93.08 | 34.0 | 95.0 | 186.675 | 94.0 |  |
