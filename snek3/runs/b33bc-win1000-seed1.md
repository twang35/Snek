# b33bc-win1000-seed1

step **50,003,968** · 1526 evals · trailing **94.55** · peak **94.94** @38,961,152 · sef **77.1** · best30 **99.8** @38,862,848

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
| init_from | None |
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
| seed | 1 |
| torch_threads | 1 |

![b33bc-win1000-seed1](b33bc-win1000-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 6.63 | 6.63 | 0.0 | 16.0 | 1.974 | 0.0 |  |
| 65536 | 28.99 | 17.81 | 0.0 | 52.0 | 24.039 | 0.0 |  |
| 98304 | 29.26 | 21.63 | 9.0 | 55.0 | 24.248 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 95.0 | 94.54 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 49676288 | 94.6 | 94.51 | 55.0 | 95.0 | 192.366 | 99.0 |  |
| 49709056 | 95.0 | 94.54 | 95.0 | 95.0 | 193.756 | 100.0 |  |
| 49741824 | 94.52 | 94.53 | 47.0 | 95.0 | 192.278 | 99.0 |  |
| 49774592 | 95.0 | 94.54 | 95.0 | 95.0 | 193.755 | 100.0 |  |
| 49807360 | 95.0 | 94.54 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 49840128 | 93.92 | 94.52 | 30.0 | 95.0 | 190.685 | 98.0 |  |
| 49872896 | 94.74 | 94.56 | 69.0 | 95.0 | 192.494 | 99.0 |  |
| 49905664 | 95.0 | 94.56 | 95.0 | 95.0 | 193.752 | 100.0 |  |
| 49938432 | 95.0 | 94.56 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49971200 | 94.25 | 94.55 | 20.0 | 95.0 | 191.969 | 99.0 |  |
| 50003968 | 94.11 | 94.55 | 38.0 | 95.0 | 190.871 | 98.0 |  |
