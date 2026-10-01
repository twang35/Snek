# b26c-pen01-seed3

step **50,003,968** · 1526 evals · trailing **94.59** · peak **94.65** @44,400,640 · sef **91.9** · best30 **98.7** @48,529,408

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
| seed | 3 |
| torch_threads | 1 |

![b26c-pen01-seed3](b26c-pen01-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.23 | 0.23 | 0.0 | 2.0 | -4.776 | 0.0 |  |
| 65536 | 14.25 | 7.24 | 0.0 | 38.0 | 10.899 | 0.0 |  |
| 98304 | 30.15 | 14.88 | 0.0 | 58.0 | 25.402 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.12 | 94.45 | 59.0 | 95.0 | 187.81 | 95.0 |  |
| 49676288 | 95.0 | 94.54 | 95.0 | 95.0 | 193.721 | 100.0 |  |
| 49709056 | 94.59 | 94.54 | 65.0 | 95.0 | 190.31 | 97.0 |  |
| 49741824 | 95.0 | 94.54 | 95.0 | 95.0 | 193.722 | 100.0 |  |
| 49774592 | 94.63 | 94.56 | 58.0 | 95.0 | 192.346 | 99.0 |  |
| 49807360 | 94.67 | 94.58 | 65.0 | 95.0 | 191.402 | 98.0 |  |
| 49840128 | 95.0 | 94.52 | 95.0 | 95.0 | 193.721 | 100.0 |  |
| 49872896 | 94.39 | 94.55 | 63.0 | 95.0 | 191.124 | 98.0 |  |
| 49905664 | 94.25 | 94.56 | 62.0 | 95.0 | 188.968 | 96.0 |  |
| 49938432 | 94.75 | 94.56 | 70.0 | 95.0 | 192.471 | 99.0 |  |
| 49971200 | 95.0 | 94.53 | 95.0 | 95.0 | 193.721 | 100.0 |  |
| 50003968 | 94.66 | 94.59 | 78.0 | 95.0 | 191.391 | 98.0 |  |
