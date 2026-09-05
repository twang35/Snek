# b15ag-ent001-seed3

step **50,003,968** · 3052 evals · trailing **94.21** · peak **94.51** @49,299,456 · sef **94.1** · best30 **98.2** @6,275,072

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
| ppo_entropy_coef | 0.001 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
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

![b15ag-ent001-seed3](b15ag-ent001-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.09 | 0.09 | 0.0 | 1.0 | -3.425 | 0.0 |  |
| 32768 | 1.96 | 1.02 | 0.0 | 9.0 | 1.415 | 0.0 |  |
| 49152 | 19.79 | 7.28 | 3.0 | 35.0 | 15.015 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.2 | 94.02 | 1.0 | 95.0 | 188.175 | 96.0 |  |
| 49840128 | 94.85 | 94.07 | 80.0 | 95.0 | 192.81 | 99.0 |  |
| 49856512 | 95.0 | 94.1 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49872896 | 94.83 | 94.19 | 83.0 | 95.0 | 191.84 | 98.0 |  |
| 49889280 | 94.7 | 94.09 | 65.0 | 95.0 | 192.705 | 99.0 |  |
| 49905664 | 94.02 | 94.18 | 21.0 | 95.0 | 190.985 | 98.0 |  |
| 49922048 | 93.46 | 94.13 | 35.0 | 95.0 | 184.455 | 92.0 |  |
| 49938432 | 93.52 | 94.13 | 20.0 | 95.0 | 186.55 | 94.0 |  |
| 49954816 | 94.92 | 94.16 | 87.0 | 95.0 | 192.925 | 99.0 |  |
| 49971200 | 93.86 | 94.11 | 38.0 | 95.0 | 186.845 | 94.0 |  |
| 49987584 | 93.37 | 94.15 | 26.0 | 95.0 | 183.325 | 91.0 |  |
| 50003968 | 94.26 | 94.21 | 58.0 | 95.0 | 189.235 | 96.0 |  |
