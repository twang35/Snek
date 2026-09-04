# b13bc-mb2048-seed1

step **50,003,968** · 3052 evals · trailing **93.89** · peak **94.31** @47,464,448 · sef **70.5** · best30 **97.7** @49,102,848

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
| ppo_minibatch | 2048 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

![b13bc-mb2048-seed1](b13bc-mb2048-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 10.85 | 13.97 | 0.0 | 29.0 | 9.45 | 0.0 |  |
| 32768 | 15.43 | 14.74 | 1.0 | 36.0 | 12.005 | 0.0 |  |
| 49152 | 14.05 | 14.05 | 2.0 | 23.0 | 9.05 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.82 | 93.7 | 54.0 | 95.0 | 186.85 | 94.0 |  |
| 49840128 | 93.74 | 93.63 | 6.0 | 95.0 | 190.75 | 98.0 |  |
| 49856512 | 94.31 | 93.71 | 26.0 | 95.0 | 192.315 | 99.0 |  |
| 49872896 | 93.32 | 93.7 | 56.0 | 95.0 | 186.35 | 94.0 |  |
| 49889280 | 94.21 | 93.68 | 50.0 | 95.0 | 191.22 | 98.0 |  |
| 49905664 | 94.74 | 93.7 | 69.0 | 95.0 | 192.745 | 99.0 |  |
| 49922048 | 95.0 | 93.8 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49938432 | 93.99 | 93.87 | 26.0 | 95.0 | 191.0 | 98.0 |  |
| 49954816 | 93.72 | 93.83 | 6.0 | 95.0 | 189.735 | 97.0 |  |
| 49971200 | 93.42 | 93.69 | 26.0 | 95.0 | 188.44 | 96.0 |  |
| 49987584 | 93.5 | 93.83 | 8.0 | 95.0 | 188.52 | 96.0 |  |
| 50003968 | 93.84 | 93.89 | 26.0 | 95.0 | 189.855 | 97.0 |  |
