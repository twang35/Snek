# b10bn-g96-seed4

step **50,003,968** · 3052 evals · trailing **92.34** · peak **94.12** @14,434,304 · sef **67.1** · best30 **93.4** @29,523,968

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.96 |
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
| ppo_horizon | 16.9 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b10bn-g96-seed4](b10bn-g96-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.63 | 2.63 | 0.0 | 10.0 | 0.69 | 0.0 |  |
| 32768 | 1.31 | 1.97 | 0.0 | 14.0 | 0.765 | 0.0 |  |
| 49152 | 5.58 | 16.75 | 1.0 | 38.0 | 4.495 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.24 | 92.4 | 1.0 | 95.0 | 173.335 | 81.0 |  |
| 49840128 | 93.54 | 93.07 | 73.0 | 95.0 | 168.615 | 76.0 |  |
| 49856512 | 94.12 | 93.07 | 77.0 | 95.0 | 177.155 | 84.0 |  |
| 49872896 | 88.39 | 92.85 | 3.0 | 95.0 | 167.445 | 80.0 |  |
| 49889280 | 90.97 | 92.72 | 2.0 | 95.0 | 171.02 | 81.0 |  |
| 49905664 | 93.8 | 92.71 | 36.0 | 95.0 | 180.86 | 88.0 |  |
| 49922048 | 91.77 | 92.68 | 1.0 | 95.0 | 169.785 | 79.0 |  |
| 49938432 | 93.23 | 92.35 | 38.0 | 95.0 | 178.3 | 86.0 |  |
| 49954816 | 93.32 | 92.32 | 16.0 | 95.0 | 177.395 | 85.0 |  |
| 49971200 | 92.36 | 92.35 | 1.0 | 95.0 | 180.37 | 89.0 |  |
| 49987584 | 93.09 | 92.34 | 3.0 | 95.0 | 185.125 | 93.0 |  |
| 50003968 | 93.08 | 92.34 | 8.0 | 95.0 | 183.125 | 91.0 |  |
