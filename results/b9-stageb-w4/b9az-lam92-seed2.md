# b9az-lam92-seed2

step **50,003,968** · 3052 evals · trailing **93.48** · peak **94.42** @25,935,872 · sef **89.3** · best30 **96.9** @14,958,592

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
| ppo_gae_lambda | 0.92 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 11.2 |
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

![b9az-lam92-seed2](b9az-lam92-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.63 | 2.63 | 1.0 | 8.0 | -0.795 | 0.0 |  |
| 32768 | 12.52 | 7.57 | 0.0 | 25.0 | 8.87 | 0.0 |  |
| 49152 | 25.68 | 17.02 | 1.0 | 47.0 | 20.995 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.74 | 93.79 | 84.0 | 95.0 | 189.76 | 96.0 |  |
| 49840128 | 94.03 | 93.78 | 72.0 | 95.0 | 187.06 | 94.0 |  |
| 49856512 | 94.79 | 93.31 | 80.0 | 95.0 | 191.8 | 98.0 |  |
| 49872896 | 93.35 | 93.34 | 22.0 | 95.0 | 187.375 | 95.0 |  |
| 49889280 | 94.68 | 93.31 | 77.0 | 95.0 | 191.69 | 98.0 |  |
| 49905664 | 94.63 | 93.34 | 72.0 | 95.0 | 190.645 | 97.0 |  |
| 49922048 | 93.94 | 93.37 | 4.0 | 95.0 | 189.955 | 97.0 |  |
| 49938432 | 92.11 | 93.31 | 12.0 | 95.0 | 184.055 | 93.0 |  |
| 49954816 | 94.01 | 93.34 | 62.0 | 95.0 | 187.04 | 94.0 |  |
| 49971200 | 93.95 | 93.63 | 8.0 | 95.0 | 189.965 | 97.0 |  |
| 49987584 | 94.72 | 93.57 | 70.0 | 95.0 | 191.73 | 98.0 |  |
| 50003968 | 94.97 | 93.48 | 92.0 | 95.0 | 192.975 | 99.0 |  |
