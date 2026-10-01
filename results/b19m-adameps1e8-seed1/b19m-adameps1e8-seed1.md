# b19m-adameps1e8-seed1

step **50,003,968** · 3052 evals · trailing **94.24** · peak **94.5** @10,256,384 · sef **93.2** · best30 **97.9** @10,289,152

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
| ppo_adam_epsilon | 1e-08 |
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
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
| seed | 1 |
| torch_threads | 1 |

![b19m-adameps1e8-seed1](b19m-adameps1e8-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 17.24 | 26.66 | 3.0 | 40.0 | 15.522 | 0.0 |  |
| 32768 | 47.92 | 34.13 | 12.0 | 91.0 | 42.908 | 0.0 |  |
| 49152 | 33.76 | 31.37 | 6.0 | 57.0 | 28.735 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.99 | 94.02 | 42.0 | 95.0 | 188.696 | 96.0 |  |
| 49840128 | 94.45 | 94.03 | 70.0 | 95.0 | 190.157 | 97.0 |  |
| 49856512 | 94.74 | 94.09 | 81.0 | 95.0 | 189.44 | 96.0 |  |
| 49872896 | 94.55 | 94.21 | 67.0 | 95.0 | 189.226 | 96.0 |  |
| 49889280 | 94.95 | 94.2 | 90.0 | 95.0 | 192.66 | 99.0 |  |
| 49905664 | 94.49 | 94.18 | 64.0 | 95.0 | 190.166 | 97.0 |  |
| 49922048 | 93.95 | 94.24 | 42.0 | 95.0 | 187.678 | 95.0 |  |
| 49938432 | 94.05 | 94.19 | 58.0 | 95.0 | 187.726 | 95.0 |  |
| 49954816 | 93.68 | 94.2 | 68.0 | 95.0 | 185.405 | 93.0 |  |
| 49971200 | 94.9 | 94.25 | 86.0 | 95.0 | 191.561 | 98.0 |  |
| 49987584 | 94.57 | 94.2 | 58.0 | 95.0 | 191.277 | 98.0 |  |
| 50003968 | 94.67 | 94.24 | 71.0 | 95.0 | 191.379 | 98.0 |  |
