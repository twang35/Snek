# b7af-fc200x100-seed2

step **50,003,968** · 3052 evals · trailing **93.53** · peak **94.48** @25,231,360 · sef **93.7** · best30 **97.8** @41,975,808

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
| fc_layers | (200, 100) |
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
| seed | 2 |
| torch_threads | 1 |

![b7af-fc200x100-seed2](b7af-fc200x100-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 11.35 | 11.35 | 1.0 | 23.0 | 6.35 | 0.0 |  |
| 32768 | 27.43 | 19.39 | 9.0 | 43.0 | 22.43 | 0.0 |  |
| 49152 | 29.24 | 22.67 | 2.0 | 50.0 | 24.285 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.4 | 93.91 | 3.0 | 95.0 | 184.305 | 92.0 |  |
| 49840128 | 93.82 | 93.71 | 7.0 | 95.0 | 189.835 | 97.0 |  |
| 49856512 | 94.96 | 93.71 | 91.0 | 95.0 | 192.965 | 99.0 |  |
| 49872896 | 93.09 | 93.7 | 5.0 | 95.0 | 189.105 | 97.0 |  |
| 49889280 | 94.97 | 93.71 | 92.0 | 95.0 | 192.93 | 99.0 |  |
| 49905664 | 92.16 | 93.58 | 11.0 | 95.0 | 183.2 | 92.0 |  |
| 49922048 | 89.96 | 93.76 | 3.0 | 95.0 | 176.975 | 88.0 |  |
| 49938432 | 92.61 | 93.64 | 5.0 | 95.0 | 179.49 | 88.0 |  |
| 49954816 | 92.98 | 93.53 | 3.0 | 95.0 | 178.82 | 87.0 |  |
| 49971200 | 92.81 | 93.49 | 1.0 | 95.0 | 184.8 | 93.0 |  |
| 49987584 | 92.97 | 93.51 | 3.0 | 95.0 | 187.945 | 96.0 |  |
| 50003968 | 93.78 | 93.53 | 6.0 | 95.0 | 189.795 | 97.0 |  |
