# b11ay-lr1e3-seed1

step **50,003,968** · 3052 evals · trailing **94.11** · peak **94.16** @13,107,200 · sef **90.4** · best30 **97.6** @13,172,736

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
| ppo_learning_rate | 0.001 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

![b11ay-lr1e3-seed1](b11ay-lr1e3-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 12.77 | 12.77 | 0.0 | 34.0 | 10.56 | 0.0 |  |
| 32768 | 51.4 | 34.6 | 15.0 | 87.0 | 46.625 | 0.0 |  |
| 49152 | 40.59 | 29.68 | 15.0 | 74.0 | 35.68 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.6 | 93.74 | 73.0 | 95.0 | 190.615 | 97.0 |  |
| 49840128 | 94.58 | 93.86 | 65.0 | 95.0 | 191.59 | 98.0 |  |
| 49856512 | 95.0 | 93.56 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49872896 | 91.51 | 93.54 | 6.0 | 95.0 | 184.45 | 94.0 |  |
| 49889280 | 94.94 | 93.59 | 89.0 | 95.0 | 192.945 | 99.0 |  |
| 49905664 | 93.75 | 93.71 | 38.0 | 95.0 | 188.77 | 96.0 |  |
| 49922048 | 94.8 | 93.73 | 80.0 | 95.0 | 191.81 | 98.0 |  |
| 49938432 | 93.99 | 94.11 | 4.0 | 95.0 | 190.955 | 98.0 |  |
| 49954816 | 93.76 | 93.88 | 28.0 | 95.0 | 187.785 | 95.0 |  |
| 49971200 | 94.77 | 93.94 | 72.0 | 95.0 | 192.775 | 99.0 |  |
| 49987584 | 94.45 | 94.06 | 40.0 | 95.0 | 192.455 | 99.0 |  |
| 50003968 | 93.94 | 94.11 | 13.0 | 95.0 | 190.95 | 98.0 |  |
