# b7ax-fc100x100-seed4

step **50,003,968** · 3052 evals · trailing **92.36** · peak **94.51** @44,957,696 · sef **94.6** · best30 **98.0** @27,688,960

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
| fc_layers | (100, 100) |
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
| seed | 4 |
| torch_threads | 1 |

![b7ax-fc100x100-seed4](b7ax-fc100x100-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 8.68 | 28.21 | 0.0 | 34.0 | 7.55 | 0.0 |  |
| 32768 | 54.05 | 35.37 | 0.0 | 94.0 | 49.41 | 0.0 |  |
| 49152 | 36.6 | 36.6 | 20.0 | 57.0 | 31.6 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 90.39 | 92.61 | 5.0 | 95.0 | 175.46 | 86.0 |  |
| 49840128 | 91.67 | 92.75 | 13.0 | 95.0 | 179.725 | 89.0 |  |
| 49856512 | 93.81 | 92.52 | 6.0 | 95.0 | 187.835 | 95.0 |  |
| 49872896 | 89.66 | 92.37 | 1.0 | 95.0 | 177.715 | 89.0 |  |
| 49889280 | 91.65 | 92.54 | 7.0 | 95.0 | 184.68 | 94.0 |  |
| 49905664 | 92.98 | 92.32 | 57.0 | 95.0 | 182.03 | 90.0 |  |
| 49922048 | 94.73 | 92.32 | 77.0 | 95.0 | 191.74 | 98.0 |  |
| 49938432 | 94.89 | 92.55 | 84.0 | 95.0 | 192.895 | 99.0 |  |
| 49954816 | 94.38 | 92.33 | 70.0 | 95.0 | 189.4 | 96.0 |  |
| 49971200 | 93.79 | 92.33 | 16.0 | 95.0 | 187.815 | 95.0 |  |
| 49987584 | 92.26 | 92.33 | 3.0 | 95.0 | 186.285 | 95.0 |  |
| 50003968 | 94.92 | 92.36 | 90.0 | 95.0 | 191.93 | 98.0 |  |
