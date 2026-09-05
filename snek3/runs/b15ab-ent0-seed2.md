# b15ab-ent0-seed2

step **50,003,968** · 3052 evals · trailing **93.97** · peak **94.51** @43,778,048 · sef **95.0** · best30 **97.5** @39,305,216

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
| ppo_entropy_coef | 0.0 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
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

![b15ab-ent0-seed2](b15ab-ent0-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.62 | 1.62 | 0.0 | 6.0 | -0.95 | 0.0 |  |
| 32768 | 15.54 | 8.58 | 4.0 | 28.0 | 10.855 | 0.0 |  |
| 49152 | 25.05 | 17.14 | 4.0 | 54.0 | 20.14 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.45 | 93.99 | 80.0 | 95.0 | 187.48 | 94.0 |  |
| 49840128 | 93.24 | 93.94 | 36.0 | 95.0 | 186.225 | 94.0 |  |
| 49856512 | 94.15 | 93.97 | 34.0 | 95.0 | 191.115 | 98.0 |  |
| 49872896 | 94.25 | 93.98 | 45.0 | 95.0 | 190.22 | 97.0 |  |
| 49889280 | 94.81 | 93.98 | 76.0 | 95.0 | 192.815 | 99.0 |  |
| 49905664 | 94.86 | 94.01 | 83.0 | 95.0 | 191.87 | 98.0 |  |
| 49922048 | 93.83 | 93.97 | 18.0 | 95.0 | 186.815 | 94.0 |  |
| 49938432 | 93.9 | 93.98 | 27.0 | 95.0 | 188.83 | 96.0 |  |
| 49954816 | 93.07 | 93.98 | 17.0 | 95.0 | 187.955 | 96.0 |  |
| 49971200 | 94.66 | 93.94 | 71.0 | 95.0 | 191.625 | 98.0 |  |
| 49987584 | 94.14 | 93.98 | 53.0 | 95.0 | 188.075 | 95.0 |  |
| 50003968 | 93.91 | 93.97 | 42.0 | 95.0 | 188.885 | 96.0 |  |
