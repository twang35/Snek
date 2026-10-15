# b26o-pen0-seed15

step **50,003,968** · 1526 evals · trailing **94.2** · peak **94.67** @41,680,896 · sef **92.9** · best30 **98.0** @41,517,056

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 32768 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 50003968 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_anneal_fraction | 0.5 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_discount_final | 0.999 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | 0.001 |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.95 |
| ppo_gae_lambda_final | 0.999 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 16.8 |
| ppo_horizon_final | 500.3 |
| ppo_learning_rate | 0.00025 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 512 |
| ppo_normalize_adv | True |
| ppo_rollout | 256 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 32768 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 15 |
| torch_threads | 1 |

![b26o-pen0-seed15](b26o-pen0-seed15.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.13 | 0.13 | 0.0 | 2.0 | -0.419 | 0.0 |  |
| 65536 | 5.38 | 2.75 | 0.0 | 30.0 | 4.347 | 0.0 |  |
| 98304 | 28.78 | 21.24 | 5.0 | 57.0 | 24.446 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 93.71 | 94.16 | 50.0 | 95.0 | 187.436 | 95.0 |  |
| 49676288 | 94.54 | 94.18 | 69.0 | 95.0 | 191.262 | 98.0 |  |
| 49709056 | 94.04 | 94.18 | 59.0 | 95.0 | 188.771 | 96.0 |  |
| 49741824 | 94.09 | 94.18 | 62.0 | 95.0 | 189.82 | 97.0 |  |
| 49774592 | 94.31 | 94.21 | 52.0 | 95.0 | 191.031 | 98.0 |  |
| 49807360 | 94.62 | 94.18 | 57.0 | 95.0 | 192.345 | 99.0 |  |
| 49840128 | 93.47 | 94.15 | 56.0 | 95.0 | 185.171 | 93.0 |  |
| 49872896 | 93.98 | 94.19 | 46.0 | 95.0 | 189.709 | 97.0 |  |
| 49905664 | 93.3 | 94.17 | 58.0 | 95.0 | 185.039 | 93.0 |  |
| 49938432 | 94.43 | 94.19 | 60.0 | 95.0 | 190.156 | 97.0 |  |
| 49971200 | 93.31 | 94.16 | 50.0 | 95.0 | 186.044 | 94.0 |  |
| 50003968 | 93.99 | 94.2 | 53.0 | 95.0 | 187.717 | 95.0 |  |
