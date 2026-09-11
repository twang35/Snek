# b13ba-mb1024-seed3

step **50,003,968** · 3052 evals · trailing **94.49** · peak **94.56** @49,987,584 · sef **86.5** · best30 **98.0** @36,716,544

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
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 1024 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b13ba-mb1024-seed3](b13ba-mb1024-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.26 | 0.26 | 0.0 | 3.0 | -0.285 | 0.0 |  |
| 32768 | 1.36 | 0.81 | 1.0 | 3.0 | -0.355 | 0.0 |  |
| 49152 | 7.29 | 2.97 | 0.0 | 17.0 | 2.56 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.91 | 94.54 | 86.0 | 95.0 | 192.915 | 99.0 |  |
| 49840128 | 94.73 | 94.54 | 68.0 | 95.0 | 192.735 | 99.0 |  |
| 49856512 | 93.74 | 94.54 | 22.0 | 95.0 | 190.75 | 98.0 |  |
| 49872896 | 94.34 | 94.54 | 57.0 | 95.0 | 191.35 | 98.0 |  |
| 49889280 | 94.82 | 94.5 | 83.0 | 95.0 | 191.83 | 98.0 |  |
| 49905664 | 94.96 | 94.52 | 91.0 | 95.0 | 192.965 | 99.0 |  |
| 49922048 | 94.23 | 94.5 | 18.0 | 95.0 | 192.235 | 99.0 |  |
| 49938432 | 94.05 | 94.5 | 57.0 | 95.0 | 187.08 | 94.0 |  |
| 49954816 | 94.75 | 94.54 | 82.0 | 95.0 | 190.765 | 97.0 |  |
| 49971200 | 95.0 | 94.54 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49987584 | 94.78 | 94.56 | 73.0 | 95.0 | 192.785 | 99.0 |  |
| 50003968 | 92.89 | 94.49 | 8.0 | 95.0 | 182.89 | 91.0 |  |
