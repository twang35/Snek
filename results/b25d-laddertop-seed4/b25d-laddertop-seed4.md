# b25d-laddertop-seed4

step **200,015,872** · 3052 evals · trailing **94.38** · peak **94.8** @44,367,872 · sef **98.2** · best30 **99.3** @141,230,080

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.999 |
| eval_interval | 65536 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 200015872 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_anneal_fraction | 0.8 |
| ppo_clip | 0.2 |
| ppo_clip_final | 0.001 |
| ppo_discount_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.99 |
| ppo_gae_lambda_final | None |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 91.0 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 512 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 65536 |
| ppo_value_loss | mse |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b25d-laddertop-seed4](b25d-laddertop-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 65536 | 9.78 | 9.78 | 1.0 | 22.0 | 4.767 | 0.0 |  |
| 131072 | 23.16 | 16.47 | 10.0 | 36.0 | 18.123 | 0.0 |  |
| 196608 | 25.5 | 19.48 | 5.0 | 49.0 | 20.475 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 199294976 | 95.0 | 94.3 | 95.0 | 95.0 | 193.745 | 100.0 |  |
| 199360512 | 93.94 | 94.31 | 16.0 | 95.0 | 190.69 | 98.0 |  |
| 199426048 | 94.21 | 94.29 | 16.0 | 95.0 | 191.956 | 99.0 |  |
| 199491584 | 94.7 | 94.31 | 65.0 | 95.0 | 192.447 | 99.0 |  |
| 199557120 | 94.09 | 94.29 | 7.0 | 95.0 | 190.848 | 98.0 |  |
| 199622656 | 95.0 | 94.29 | 95.0 | 95.0 | 193.742 | 100.0 |  |
| 199688192 | 95.0 | 94.31 | 95.0 | 95.0 | 193.749 | 100.0 |  |
| 199753728 | 94.95 | 94.34 | 90.0 | 95.0 | 192.691 | 99.0 |  |
| 199819264 | 95.0 | 94.37 | 95.0 | 95.0 | 193.743 | 100.0 |  |
| 199884800 | 94.13 | 94.35 | 65.0 | 95.0 | 189.879 | 97.0 |  |
| 199950336 | 95.0 | 94.38 | 95.0 | 95.0 | 193.742 | 100.0 |  |
| 200015872 | 94.62 | 94.38 | 57.0 | 95.0 | 192.319 | 99.0 |  |
