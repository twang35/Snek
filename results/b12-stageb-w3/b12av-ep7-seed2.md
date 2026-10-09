# b12av-ep7-seed2

step **50,003,968** · 3052 evals · trailing **94.04** · peak **94.61** @32,178,176 · sef **92.1** · best30 **98.6** @32,292,864

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
| ppo_epochs | 7 |
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

![b12av-ep7-seed2](b12av-ep7-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.03 | 1.03 | 0.0 | 5.0 | -0.325 | 0.0 |  |
| 32768 | 8.94 | 4.98 | 0.0 | 15.0 | 4.03 | 0.0 |  |
| 49152 | 24.92 | 15.33 | 8.0 | 44.0 | 19.92 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.76 | 94.23 | 3.0 | 95.0 | 187.785 | 95.0 |  |
| 49840128 | 94.13 | 94.22 | 73.0 | 95.0 | 187.16 | 94.0 |  |
| 49856512 | 94.48 | 94.22 | 81.0 | 95.0 | 186.515 | 93.0 |  |
| 49872896 | 94.43 | 94.18 | 72.0 | 95.0 | 189.45 | 96.0 |  |
| 49889280 | 94.42 | 94.22 | 74.0 | 95.0 | 188.445 | 95.0 |  |
| 49905664 | 94.1 | 94.2 | 27.0 | 95.0 | 190.07 | 97.0 |  |
| 49922048 | 91.59 | 94.07 | 3.0 | 95.0 | 179.645 | 89.0 |  |
| 49938432 | 94.45 | 94.17 | 80.0 | 95.0 | 189.47 | 96.0 |  |
| 49954816 | 93.09 | 94.03 | 1.0 | 95.0 | 184.085 | 92.0 |  |
| 49971200 | 94.44 | 94.08 | 68.0 | 95.0 | 190.455 | 97.0 |  |
| 49987584 | 94.09 | 94.08 | 65.0 | 95.0 | 188.07 | 95.0 |  |
| 50003968 | 93.2 | 94.04 | 62.0 | 95.0 | 184.24 | 92.0 |  |
