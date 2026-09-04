# b13bd-mb2048-seed2

step **50,003,968** · 3052 evals · trailing **93.44** · peak **94.38** @48,513,024 · sef **82.7** · best30 **97.6** @48,037,888

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
| seed | 2 |
| torch_threads | 1 |

![b13bd-mb2048-seed2](b13bd-mb2048-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.6 | 0.6 | 0.0 | 4.0 | -0.125 | 0.0 |  |
| 32768 | 6.6 | 3.6 | 0.0 | 16.0 | 2.77 | 0.0 |  |
| 49152 | 9.79 | 7.19 | 1.0 | 23.0 | 4.97 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.01 | 93.44 | 6.0 | 95.0 | 184.045 | 93.0 |  |
| 49840128 | 94.38 | 93.41 | 64.0 | 95.0 | 190.395 | 97.0 |  |
| 49856512 | 93.78 | 93.37 | 60.0 | 95.0 | 187.805 | 95.0 |  |
| 49872896 | 92.95 | 93.38 | 58.0 | 95.0 | 181.005 | 89.0 |  |
| 49889280 | 92.44 | 93.37 | 14.0 | 95.0 | 182.485 | 91.0 |  |
| 49905664 | 94.31 | 93.46 | 57.0 | 95.0 | 191.32 | 98.0 |  |
| 49922048 | 93.82 | 93.38 | 44.0 | 95.0 | 188.84 | 96.0 |  |
| 49938432 | 91.86 | 93.38 | 57.0 | 95.0 | 179.915 | 89.0 |  |
| 49954816 | 94.14 | 93.41 | 58.0 | 95.0 | 189.16 | 96.0 |  |
| 49971200 | 93.84 | 93.41 | 49.0 | 95.0 | 188.86 | 96.0 |  |
| 49987584 | 93.0 | 93.37 | 49.0 | 95.0 | 186.03 | 94.0 |  |
| 50003968 | 95.0 | 93.44 | 95.0 | 95.0 | 194.0 | 100.0 |  |
