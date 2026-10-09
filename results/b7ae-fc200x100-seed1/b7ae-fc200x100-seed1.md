# b7ae-fc200x100-seed1

step **50,003,968** · 3052 evals · trailing **93.81** · peak **94.54** @43,204,608 · sef **94.5** · best30 **97.3** @7,487,488

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
| fc_layers | (200, 100) |
| graph_eval_episodes | 100 |
| max_steps | 50003968 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

![b7ae-fc200x100-seed1](b7ae-fc200x100-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 10.87 | 10.87 | 0.0 | 22.0 | 6.545 | 0.0 |  |
| 32768 | 29.08 | 26.12 | 0.0 | 62.0 | 25.295 | 0.0 |  |
| 49152 | 32.73 | 21.8 | 1.0 | 71.0 | 27.865 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.89 | 93.72 | 12.0 | 95.0 | 189.815 | 97.0 |  |
| 49840128 | 94.35 | 93.73 | 62.0 | 95.0 | 187.335 | 94.0 |  |
| 49856512 | 93.89 | 93.79 | 11.0 | 95.0 | 188.82 | 96.0 |  |
| 49872896 | 94.79 | 93.82 | 74.0 | 95.0 | 192.75 | 99.0 |  |
| 49889280 | 92.88 | 93.87 | 3.0 | 95.0 | 184.78 | 93.0 |  |
| 49905664 | 94.6 | 93.88 | 66.0 | 95.0 | 191.52 | 98.0 |  |
| 49922048 | 93.19 | 93.82 | 7.0 | 95.0 | 185.135 | 93.0 |  |
| 49938432 | 93.16 | 93.8 | 32.0 | 95.0 | 182.03 | 90.0 |  |
| 49954816 | 92.66 | 93.78 | 5.0 | 95.0 | 182.705 | 91.0 |  |
| 49971200 | 93.06 | 93.83 | 7.0 | 95.0 | 179.94 | 88.0 |  |
| 49987584 | 93.6 | 93.88 | 25.0 | 95.0 | 188.485 | 96.0 |  |
| 50003968 | 94.26 | 93.81 | 64.0 | 95.0 | 189.28 | 96.0 |  |
