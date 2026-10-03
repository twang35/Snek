# b9ag-lam50-seed3

step **50,003,968** · 3052 evals · trailing **93.31** · peak **94.27** @9,306,112 · sef **83.7** · best30 **93.7** @38,649,856

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
| ppo_gae_lambda | 0.5 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 2.0 |
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

![b9ag-lam50-seed3](b9ag-lam50-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.1 | 0.1 | 0.0 | 1.0 | -0.4 | 0.0 |  |
| 32768 | 0.68 | 0.39 | 0.0 | 6.0 | 0.18 | 0.0 |  |
| 49152 | 20.76 | 7.18 | 1.0 | 56.0 | 18.28 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.63 | 93.28 | 11.0 | 95.0 | 185.62 | 93.0 |  |
| 49840128 | 93.23 | 93.41 | 3.0 | 95.0 | 185.265 | 93.0 |  |
| 49856512 | 92.55 | 93.44 | 8.0 | 95.0 | 179.61 | 88.0 |  |
| 49872896 | 91.38 | 93.46 | 1.0 | 95.0 | 179.39 | 89.0 |  |
| 49889280 | 91.43 | 93.34 | 5.0 | 95.0 | 176.455 | 86.0 |  |
| 49905664 | 93.08 | 93.36 | 8.0 | 95.0 | 183.125 | 91.0 |  |
| 49922048 | 94.26 | 93.42 | 78.0 | 95.0 | 184.26 | 91.0 |  |
| 49938432 | 93.6 | 93.41 | 17.0 | 95.0 | 186.63 | 94.0 |  |
| 49954816 | 93.05 | 93.32 | 13.0 | 95.0 | 185.085 | 93.0 |  |
| 49971200 | 93.84 | 93.39 | 22.0 | 95.0 | 185.83 | 93.0 |  |
| 49987584 | 93.8 | 93.37 | 50.0 | 95.0 | 182.715 | 90.0 |  |
| 50003968 | 94.43 | 93.31 | 79.0 | 95.0 | 186.375 | 93.0 |  |
