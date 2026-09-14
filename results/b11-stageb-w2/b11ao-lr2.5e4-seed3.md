# b11ao-lr2.5e4-seed3

step **50,003,968** · 3052 evals · trailing **94.11** · peak **94.53** @33,030,144 · sef **90.8** · best30 **98.3** @32,980,992

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
| ppo_learning_rate | 0.00025 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b11ao-lr2.5e4-seed3](b11ao-lr2.5e4-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.11 | 0.11 | 0.0 | 2.0 | -3.54 | 0.0 |  |
| 32768 | 1.99 | 1.05 | 0.0 | 9.0 | 1.175 | 0.0 |  |
| 49152 | 10.12 | 4.07 | 0.0 | 25.0 | 6.29 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.42 | 93.87 | 68.0 | 95.0 | 189.44 | 96.0 |  |
| 49840128 | 94.72 | 93.83 | 70.0 | 95.0 | 191.64 | 98.0 |  |
| 49856512 | 94.64 | 93.85 | 59.0 | 95.0 | 192.645 | 99.0 |  |
| 49872896 | 94.82 | 93.96 | 82.0 | 95.0 | 191.83 | 98.0 |  |
| 49889280 | 94.57 | 93.92 | 61.0 | 95.0 | 191.58 | 98.0 |  |
| 49905664 | 94.95 | 93.95 | 90.0 | 95.0 | 192.955 | 99.0 |  |
| 49922048 | 94.15 | 93.98 | 31.0 | 95.0 | 190.12 | 97.0 |  |
| 49938432 | 95.0 | 94.01 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49954816 | 95.0 | 94.01 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49971200 | 94.03 | 94.07 | 44.0 | 95.0 | 189.05 | 96.0 |  |
| 49987584 | 94.62 | 94.12 | 59.0 | 95.0 | 191.63 | 98.0 |  |
| 50003968 | 94.22 | 94.11 | 59.0 | 95.0 | 190.19 | 97.0 |  |
