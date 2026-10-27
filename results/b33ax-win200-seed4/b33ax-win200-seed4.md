# b33ax-win200-seed4

step **50,003,968** · 1526 evals · trailing **94.72** · peak **94.85** @39,944,192 · sef **88.7** · best30 **99.7** @39,878,656

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
| seed | 4 |
| torch_threads | 1 |

![b33ax-win200-seed4](b33ax-win200-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.28 | 4.28 | 0.0 | 10.0 | 1.66 | 0.0 |  |
| 65536 | 12.49 | 8.38 | 0.0 | 38.0 | 9.032 | 0.0 |  |
| 98304 | 23.57 | 13.45 | 0.0 | 50.0 | 18.654 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 95.0 | 94.76 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 49676288 | 95.0 | 94.76 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 49709056 | 93.98 | 94.73 | 39.0 | 95.0 | 190.666 | 98.0 |  |
| 49741824 | 95.0 | 94.74 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 49774592 | 94.13 | 94.69 | 8.0 | 95.0 | 191.905 | 99.0 |  |
| 49807360 | 94.21 | 94.71 | 16.0 | 95.0 | 191.975 | 99.0 |  |
| 49840128 | 95.0 | 94.74 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 49872896 | 95.0 | 94.72 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 49905664 | 94.69 | 94.73 | 64.0 | 95.0 | 192.453 | 99.0 |  |
| 49938432 | 95.0 | 94.72 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 49971200 | 94.69 | 94.72 | 64.0 | 95.0 | 192.452 | 99.0 |  |
| 50003968 | 95.0 | 94.72 | 95.0 | 95.0 | 193.763 | 100.0 |  |
