# b10bx-g995-seed2

step **50,003,968** · 3052 evals · trailing **94.32** · peak **94.54** @21,905,408 · sef **92.8** · best30 **97.9** @36,847,616

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.995 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 50003968 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 40.2 |
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

![b10bx-g995-seed2](b10bx-g995-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.52 | 1.52 | 0.0 | 5.0 | -0.915 | 0.0 |  |
| 32768 | 7.51 | 4.51 | 0.0 | 19.0 | 3.41 | 0.0 |  |
| 49152 | 23.9 | 14.16 | 8.0 | 41.0 | 18.9 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.94 | 94.28 | 6.0 | 95.0 | 189.91 | 97.0 |  |
| 49840128 | 94.56 | 94.31 | 51.0 | 95.0 | 192.52 | 99.0 |  |
| 49856512 | 94.64 | 94.33 | 74.0 | 95.0 | 190.655 | 97.0 |  |
| 49872896 | 94.65 | 94.38 | 86.0 | 95.0 | 187.68 | 94.0 |  |
| 49889280 | 93.23 | 94.39 | 10.0 | 95.0 | 189.245 | 97.0 |  |
| 49905664 | 93.16 | 94.34 | 8.0 | 95.0 | 186.19 | 94.0 |  |
| 49922048 | 94.82 | 94.24 | 86.0 | 95.0 | 191.83 | 98.0 |  |
| 49938432 | 94.34 | 94.28 | 60.0 | 95.0 | 189.36 | 96.0 |  |
| 49954816 | 94.43 | 94.3 | 66.0 | 95.0 | 190.445 | 97.0 |  |
| 49971200 | 94.6 | 94.28 | 70.0 | 95.0 | 190.615 | 97.0 |  |
| 49987584 | 93.75 | 94.26 | 64.0 | 95.0 | 185.74 | 93.0 |  |
| 50003968 | 94.55 | 94.32 | 70.0 | 95.0 | 191.56 | 98.0 |  |
