# b13ai-mb128-seed1

step **50,003,968** · 3052 evals · trailing **93.89** · peak **94.45** @13,746,176 · sef **90.2** · best30 **97.8** @38,453,248

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
| ppo_minibatch | 128 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

![b13ai-mb128-seed1](b13ai-mb128-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 18.97 | 18.97 | 1.0 | 41.0 | 15.725 | 0.0 |  |
| 32768 | 47.28 | 34.77 | 19.0 | 88.0 | 42.325 | 0.0 |  |
| 49152 | 33.36 | 26.16 | 10.0 | 68.0 | 28.36 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.73 | 94.06 | 12.0 | 95.0 | 188.75 | 96.0 |  |
| 49840128 | 93.51 | 94.0 | 14.0 | 95.0 | 187.49 | 95.0 |  |
| 49856512 | 93.93 | 93.99 | 8.0 | 95.0 | 190.94 | 98.0 |  |
| 49872896 | 94.86 | 93.98 | 81.0 | 95.0 | 192.865 | 99.0 |  |
| 49889280 | 93.68 | 93.96 | 12.0 | 95.0 | 187.705 | 95.0 |  |
| 49905664 | 94.63 | 93.97 | 58.0 | 95.0 | 192.635 | 99.0 |  |
| 49922048 | 94.76 | 93.98 | 80.0 | 95.0 | 191.77 | 98.0 |  |
| 49938432 | 92.89 | 94.05 | 22.0 | 95.0 | 185.785 | 94.0 |  |
| 49954816 | 92.12 | 93.95 | 14.0 | 95.0 | 184.02 | 93.0 |  |
| 49971200 | 92.39 | 93.97 | 24.0 | 95.0 | 184.29 | 93.0 |  |
| 49987584 | 93.58 | 93.99 | 14.0 | 95.0 | 187.605 | 95.0 |  |
| 50003968 | 93.02 | 93.89 | 24.0 | 95.0 | 185.915 | 94.0 |  |
