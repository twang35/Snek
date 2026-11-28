# b7aa-fc320-seed1

step **50,003,968** · 3052 evals · trailing **94.27** · peak **94.57** @35,241,984 · sef **94.0** · best30 **97.8** @31,784,960

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

![b7aa-fc320-seed1](b7aa-fc320-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 10.71 | 20.88 | 0.0 | 24.0 | 9.625 | 0.0 |  |
| 32768 | 43.62 | 31.73 | 8.0 | 84.0 | 38.755 | 0.0 |  |
| 49152 | 38.92 | 29.35 | 11.0 | 81.0 | 33.92 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.02 | 94.26 | 28.0 | 95.0 | 184.015 | 92.0 |  |
| 49840128 | 94.28 | 94.38 | 68.0 | 95.0 | 188.26 | 95.0 |  |
| 49856512 | 94.44 | 94.36 | 74.0 | 95.0 | 189.46 | 96.0 |  |
| 49872896 | 93.84 | 94.36 | 17.0 | 95.0 | 188.815 | 96.0 |  |
| 49889280 | 93.93 | 94.21 | 11.0 | 95.0 | 188.86 | 96.0 |  |
| 49905664 | 94.23 | 94.25 | 18.0 | 95.0 | 192.235 | 99.0 |  |
| 49922048 | 94.84 | 94.21 | 82.0 | 95.0 | 191.85 | 98.0 |  |
| 49938432 | 93.76 | 94.18 | 10.0 | 95.0 | 189.775 | 97.0 |  |
| 49954816 | 94.01 | 94.15 | 19.0 | 95.0 | 189.935 | 97.0 |  |
| 49971200 | 93.52 | 94.1 | 22.0 | 95.0 | 184.47 | 92.0 |  |
| 49987584 | 93.78 | 94.09 | 22.0 | 95.0 | 186.765 | 94.0 |  |
| 50003968 | 95.0 | 94.27 | 95.0 | 95.0 | 194.0 | 100.0 |  |
