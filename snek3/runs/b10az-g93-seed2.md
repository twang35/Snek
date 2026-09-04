# b10az-g93-seed2

step **50,003,968** · 3052 evals · trailing **91.74** · peak **94.01** @21,331,968 · sef **16.1** · best30 **86.9** @48,250,880

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.93 |
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
| ppo_horizon | 11.3 |
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

![b10az-g93-seed2](b10az-g93-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.37 | 2.37 | 0.0 | 6.0 | -1.145 | 0.0 |  |
| 32768 | 9.47 | 5.92 | 0.0 | 21.0 | 4.83 | 0.0 |  |
| 49152 | 20.68 | 10.84 | 0.0 | 43.0 | 15.815 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.61 | 91.75 | 44.0 | 95.0 | 167.69 | 75.0 |  |
| 49840128 | 91.77 | 92.03 | 43.0 | 95.0 | 161.915 | 71.0 |  |
| 49856512 | 93.24 | 92.42 | 47.0 | 95.0 | 169.355 | 77.0 |  |
| 49872896 | 92.24 | 92.4 | 67.0 | 95.0 | 146.465 | 55.0 |  |
| 49889280 | 92.53 | 92.41 | 57.0 | 95.0 | 144.72 | 53.0 |  |
| 49905664 | 93.1 | 91.94 | 63.0 | 95.0 | 150.31 | 58.0 |  |
| 49922048 | 91.85 | 91.83 | 67.0 | 95.0 | 142.095 | 51.0 |  |
| 49938432 | 92.5 | 92.31 | 64.0 | 95.0 | 136.775 | 45.0 |  |
| 49954816 | 91.66 | 92.1 | 42.0 | 95.0 | 139.915 | 49.0 |  |
| 49971200 | 92.4 | 92.38 | 41.0 | 95.0 | 149.565 | 58.0 |  |
| 49987584 | 92.87 | 92.34 | 6.0 | 95.0 | 165.005 | 73.0 |  |
| 50003968 | 93.0 | 91.74 | 72.0 | 95.0 | 160.16 | 68.0 |  |
