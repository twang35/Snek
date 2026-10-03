# b7as-fc160x160-seed3

step **50,003,968** · 3052 evals · trailing **93.78** · peak **94.6** @43,499,520 · sef **94.6** · best30 **98.1** @43,466,752

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
| fc_layers | (160, 160) |
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
| seed | 3 |
| torch_threads | 1 |

![b7as-fc160x160-seed3](b7as-fc160x160-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 5.19 | 27.91 | 0.0 | 24.0 | 0.64 | 0.0 |  |
| 32768 | 34.58 | 34.14 | 1.0 | 58.0 | 29.67 | 0.0 |  |
| 49152 | 36.44 | 36.44 | 6.0 | 66.0 | 31.44 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.95 | 93.45 | 90.0 | 95.0 | 192.955 | 99.0 |  |
| 49840128 | 94.43 | 93.64 | 67.0 | 95.0 | 187.415 | 94.0 |  |
| 49856512 | 94.89 | 93.51 | 84.0 | 95.0 | 192.895 | 99.0 |  |
| 49872896 | 93.08 | 93.41 | 9.0 | 95.0 | 185.025 | 93.0 |  |
| 49889280 | 92.83 | 93.45 | 1.0 | 95.0 | 187.805 | 96.0 |  |
| 49905664 | 93.51 | 93.4 | 6.0 | 95.0 | 184.505 | 92.0 |  |
| 49922048 | 93.63 | 93.41 | 66.0 | 95.0 | 182.635 | 90.0 |  |
| 49938432 | 94.26 | 93.43 | 73.0 | 95.0 | 186.25 | 93.0 |  |
| 49954816 | 94.03 | 93.42 | 33.0 | 95.0 | 190.995 | 98.0 |  |
| 49971200 | 94.77 | 93.43 | 82.0 | 95.0 | 190.785 | 97.0 |  |
| 49987584 | 94.63 | 93.49 | 84.0 | 95.0 | 187.66 | 94.0 |  |
| 50003968 | 94.67 | 93.78 | 82.0 | 95.0 | 189.69 | 96.0 |  |
