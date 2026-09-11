# b13ae-mb64-seed1

step **50,003,968** · 3052 evals · trailing **93.34** · peak **94.39** @38,699,008 · sef **91.0** · best30 **97.9** @34,586,624

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
| ppo_minibatch | 64 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

![b13ae-mb64-seed1](b13ae-mb64-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.97 | 2.97 | 0.0 | 12.0 | 2.47 | 0.0 |  |
| 32768 | 36.28 | 26.21 | 11.0 | 80.0 | 31.325 | 0.0 |  |
| 49152 | 29.65 | 16.31 | 10.0 | 70.0 | 24.65 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.44 | 93.44 | 66.0 | 95.0 | 188.33 | 95.0 |  |
| 49840128 | 94.49 | 93.21 | 73.0 | 95.0 | 190.505 | 97.0 |  |
| 49856512 | 93.6 | 93.23 | 35.0 | 95.0 | 186.585 | 94.0 |  |
| 49872896 | 94.77 | 93.32 | 82.0 | 95.0 | 191.78 | 98.0 |  |
| 49889280 | 94.81 | 93.35 | 76.0 | 95.0 | 192.815 | 99.0 |  |
| 49905664 | 94.78 | 93.23 | 73.0 | 95.0 | 192.785 | 99.0 |  |
| 49922048 | 94.64 | 93.39 | 73.0 | 95.0 | 190.655 | 97.0 |  |
| 49938432 | 93.58 | 93.3 | 13.0 | 95.0 | 187.56 | 95.0 |  |
| 49954816 | 93.81 | 93.31 | 21.0 | 95.0 | 189.78 | 97.0 |  |
| 49971200 | 93.89 | 93.43 | 56.0 | 95.0 | 184.885 | 92.0 |  |
| 49987584 | 94.82 | 93.34 | 77.0 | 95.0 | 192.825 | 99.0 |  |
| 50003968 | 94.31 | 93.34 | 40.0 | 95.0 | 191.275 | 98.0 |  |
