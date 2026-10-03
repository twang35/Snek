# b12aq-ep6-seed1

step **50,003,968** · 3052 evals · trailing **93.85** · peak **94.47** @11,911,168 · sef **89.3** · best30 **97.8** @47,808,512

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
| ppo_epochs | 6 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.0003 |
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

![b12aq-ep6-seed1](b12aq-ep6-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 12.79 | 12.79 | 4.0 | 27.0 | 8.645 | 0.0 |  |
| 32768 | 38.79 | 32.06 | 13.0 | 78.0 | 33.835 | 0.0 |  |
| 49152 | 32.41 | 22.6 | 7.0 | 50.0 | 27.455 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.98 | 93.99 | 93.0 | 95.0 | 192.94 | 99.0 |  |
| 49840128 | 94.86 | 94.01 | 84.0 | 95.0 | 191.87 | 98.0 |  |
| 49856512 | 94.81 | 93.86 | 81.0 | 95.0 | 191.82 | 98.0 |  |
| 49872896 | 94.16 | 93.95 | 64.0 | 95.0 | 188.185 | 95.0 |  |
| 49889280 | 94.72 | 93.84 | 74.0 | 95.0 | 191.73 | 98.0 |  |
| 49905664 | 94.26 | 93.87 | 60.0 | 95.0 | 189.28 | 96.0 |  |
| 49922048 | 93.85 | 93.88 | 10.0 | 95.0 | 189.865 | 97.0 |  |
| 49938432 | 93.34 | 93.93 | 28.0 | 95.0 | 184.38 | 92.0 |  |
| 49954816 | 94.59 | 93.83 | 78.0 | 95.0 | 189.565 | 96.0 |  |
| 49971200 | 93.31 | 93.8 | 70.0 | 95.0 | 178.38 | 86.0 |  |
| 49987584 | 93.7 | 93.77 | 72.0 | 95.0 | 181.665 | 89.0 |  |
| 50003968 | 93.32 | 93.85 | 73.0 | 95.0 | 181.375 | 89.0 |  |
