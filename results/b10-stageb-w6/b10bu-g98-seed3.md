# b10bu-g98-seed3

step **50,003,968** · 3052 evals · trailing **94.38** · peak **94.5** @45,465,600 · sef **87.3** · best30 **97.4** @23,674,880

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.98 |
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
| ppo_horizon | 25.3 |
| ppo_learning_rate | 0.0003 |
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

![b10bu-g98-seed3](b10bu-g98-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.11 | 0.11 | 0.0 | 1.0 | -0.84 | 0.0 |  |
| 32768 | 1.08 | 0.6 | 0.0 | 7.0 | 0.58 | 0.0 |  |
| 49152 | 17.39 | 6.19 | 0.0 | 45.0 | 12.975 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.8 | 94.16 | 78.0 | 95.0 | 191.81 | 98.0 |  |
| 49840128 | 94.89 | 94.16 | 90.0 | 95.0 | 190.905 | 97.0 |  |
| 49856512 | 94.97 | 94.26 | 92.0 | 95.0 | 192.975 | 99.0 |  |
| 49872896 | 94.49 | 94.25 | 60.0 | 95.0 | 190.505 | 97.0 |  |
| 49889280 | 94.44 | 94.18 | 73.0 | 95.0 | 190.455 | 97.0 |  |
| 49905664 | 94.26 | 94.23 | 63.0 | 95.0 | 189.28 | 96.0 |  |
| 49922048 | 94.65 | 94.4 | 65.0 | 95.0 | 191.66 | 98.0 |  |
| 49938432 | 94.65 | 94.36 | 72.0 | 95.0 | 190.665 | 97.0 |  |
| 49954816 | 94.54 | 94.33 | 70.0 | 95.0 | 190.51 | 97.0 |  |
| 49971200 | 94.44 | 94.42 | 56.0 | 95.0 | 191.45 | 98.0 |  |
| 49987584 | 93.49 | 94.38 | 8.0 | 95.0 | 188.51 | 96.0 |  |
| 50003968 | 94.78 | 94.38 | 76.0 | 95.0 | 191.79 | 98.0 |  |
