# b10am-g90-seed1

step **50,003,968** · 3052 evals · trailing **91.43** · peak **93.62** @32,899,072 · sef **3.9** · best30 **81.6** @26,329,088

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.9 |
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
| ppo_horizon | 8.5 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

![b10am-g90-seed1](b10am-g90-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 3.52 | 3.52 | 0.0 | 12.0 | 3.02 | 0.0 |  |
| 32768 | 52.22 | 41.75 | 1.0 | 82.0 | 48.165 | 0.0 |  |
| 49152 | 56.61 | 39.13 | 1.0 | 95.0 | 54.09 | 1.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.66 | 91.59 | 63.0 | 95.0 | 170.77 | 78.0 |  |
| 49840128 | 87.37 | 91.78 | 1.0 | 95.0 | 158.42 | 72.0 |  |
| 49856512 | 89.3 | 91.63 | 1.0 | 95.0 | 152.435 | 64.0 |  |
| 49872896 | 91.53 | 91.38 | 14.0 | 95.0 | 162.625 | 72.0 |  |
| 49889280 | 92.29 | 91.37 | 51.0 | 95.0 | 154.475 | 63.0 |  |
| 49905664 | 88.87 | 91.48 | 3.0 | 95.0 | 149.925 | 62.0 |  |
| 49922048 | 91.5 | 91.48 | 17.0 | 95.0 | 153.595 | 63.0 |  |
| 49938432 | 91.42 | 91.49 | 13.0 | 95.0 | 156.545 | 66.0 |  |
| 49954816 | 92.78 | 91.48 | 14.0 | 95.0 | 166.86 | 75.0 |  |
| 49971200 | 89.82 | 91.45 | 15.0 | 95.0 | 152.865 | 64.0 |  |
| 49987584 | 90.66 | 91.56 | 15.0 | 95.0 | 151.715 | 62.0 |  |
| 50003968 | 90.1 | 91.43 | 15.0 | 95.0 | 150.295 | 61.0 |  |
