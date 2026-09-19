# b11ae-lr1e4-seed1

step **50,003,968** · 3052 evals · trailing **94.03** · peak **94.52** @38,240,256 · sef **81.9** · best30 **98.2** @38,223,872

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
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.0001 |
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

![b11ae-lr1e4-seed1](b11ae-lr1e4-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 10.29 | 16.62 | 1.0 | 32.0 | 9.16 | 0.0 |  |
| 32768 | 16.92 | 16.92 | 2.0 | 37.0 | 11.92 | 0.0 |  |
| 49152 | 16.82 | 17.11 | 2.0 | 41.0 | 11.82 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.97 | 93.88 | 92.0 | 95.0 | 192.975 | 99.0 |  |
| 49840128 | 94.79 | 94.04 | 74.0 | 95.0 | 192.795 | 99.0 |  |
| 49856512 | 93.6 | 94.04 | 8.0 | 95.0 | 187.625 | 95.0 |  |
| 49872896 | 93.45 | 93.87 | 56.0 | 95.0 | 187.475 | 95.0 |  |
| 49889280 | 94.6 | 93.98 | 55.0 | 95.0 | 192.605 | 99.0 |  |
| 49905664 | 95.0 | 94.01 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49922048 | 93.94 | 93.87 | 55.0 | 95.0 | 189.955 | 97.0 |  |
| 49938432 | 93.89 | 93.97 | 56.0 | 95.0 | 188.91 | 96.0 |  |
| 49954816 | 94.69 | 94.08 | 73.0 | 95.0 | 191.7 | 98.0 |  |
| 49971200 | 94.49 | 94.11 | 56.0 | 95.0 | 191.5 | 98.0 |  |
| 49987584 | 93.37 | 94.05 | 10.0 | 95.0 | 188.39 | 96.0 |  |
| 50003968 | 93.32 | 94.03 | 6.0 | 95.0 | 188.34 | 96.0 |  |
