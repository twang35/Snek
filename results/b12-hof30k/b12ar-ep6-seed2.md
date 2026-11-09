# b12ar-ep6-seed2

step **50,003,968** · 3052 evals · trailing **94.01** · peak **94.59** @24,264,704 · sef **93.0** · best30 **98.2** @8,847,360

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
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 6 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
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

![b12ar-ep6-seed2](b12ar-ep6-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.98 | 0.98 | 0.0 | 4.0 | -0.33 | 0.0 |  |
| 32768 | 11.78 | 6.38 | 3.0 | 28.0 | 6.825 | 0.0 |  |
| 49152 | 25.34 | 12.7 | 6.0 | 44.0 | 20.34 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.63 | 94.13 | 73.0 | 95.0 | 191.55 | 98.0 |  |
| 49840128 | 93.13 | 93.96 | 28.0 | 95.0 | 186.025 | 94.0 |  |
| 49856512 | 93.89 | 94.18 | 15.0 | 95.0 | 187.78 | 95.0 |  |
| 49872896 | 94.64 | 94.19 | 80.0 | 95.0 | 190.565 | 97.0 |  |
| 49889280 | 93.13 | 94.21 | 10.0 | 95.0 | 188.06 | 96.0 |  |
| 49905664 | 93.01 | 94.1 | 45.0 | 95.0 | 184.955 | 93.0 |  |
| 49922048 | 93.46 | 94.08 | 16.0 | 95.0 | 185.36 | 93.0 |  |
| 49938432 | 93.91 | 94.08 | 65.0 | 95.0 | 185.765 | 93.0 |  |
| 49954816 | 94.35 | 94.09 | 37.0 | 95.0 | 191.315 | 98.0 |  |
| 49971200 | 94.23 | 94.11 | 45.0 | 95.0 | 191.195 | 98.0 |  |
| 49987584 | 92.13 | 94.0 | 8.0 | 95.0 | 186.02 | 95.0 |  |
| 50003968 | 94.28 | 94.01 | 38.0 | 95.0 | 190.205 | 97.0 |  |
