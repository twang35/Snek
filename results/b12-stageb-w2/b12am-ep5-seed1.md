# b12am-ep5-seed1

step **50,003,968** · 3052 evals · trailing **94.52** · peak **94.62** @35,831,808 · sef **88.0** · best30 **98.3** @35,979,264

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
| ppo_epochs | 5 |
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

![b12am-ep5-seed1](b12am-ep5-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 9.85 | 9.85 | 1.0 | 28.0 | 8.585 | 0.0 |  |
| 32768 | 38.8 | 26.54 | 12.0 | 70.0 | 33.8 | 0.0 |  |
| 49152 | 33.69 | 28.33 | 2.0 | 75.0 | 28.825 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.61 | 94.45 | 78.0 | 95.0 | 190.58 | 97.0 |  |
| 49840128 | 94.32 | 94.36 | 27.0 | 95.0 | 192.28 | 99.0 |  |
| 49856512 | 93.77 | 94.38 | 26.0 | 95.0 | 189.74 | 97.0 |  |
| 49872896 | 95.0 | 94.43 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49889280 | 93.11 | 94.47 | 14.0 | 95.0 | 187.09 | 95.0 |  |
| 49905664 | 94.28 | 94.37 | 32.0 | 95.0 | 191.29 | 98.0 |  |
| 49922048 | 94.6 | 94.36 | 70.0 | 95.0 | 190.57 | 97.0 |  |
| 49938432 | 94.8 | 94.57 | 88.0 | 95.0 | 190.815 | 97.0 |  |
| 49954816 | 94.75 | 94.49 | 82.0 | 95.0 | 190.72 | 97.0 |  |
| 49971200 | 94.21 | 94.55 | 27.0 | 95.0 | 190.135 | 97.0 |  |
| 49987584 | 94.08 | 94.55 | 55.0 | 95.0 | 189.055 | 96.0 |  |
| 50003968 | 95.0 | 94.52 | 95.0 | 95.0 | 194.0 | 100.0 |  |
