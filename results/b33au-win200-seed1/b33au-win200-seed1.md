# b33au-win200-seed1

step **50,003,968** · 1526 evals · trailing **94.73** · peak **94.82** @47,874,048 · sef **86.6** · best30 **99.6** @25,296,896

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

![b33au-win200-seed1](b33au-win200-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 7.04 | 7.04 | 0.0 | 17.0 | 2.25 | 0.0 |  |
| 65536 | 27.95 | 17.5 | 0.0 | 56.0 | 23.137 | 0.0 |  |
| 98304 | 27.42 | 20.8 | 1.0 | 48.0 | 22.415 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.33 | 94.7 | 28.0 | 95.0 | 192.057 | 99.0 |  |
| 49676288 | 94.64 | 94.67 | 59.0 | 95.0 | 192.408 | 99.0 |  |
| 49709056 | 95.0 | 94.7 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 49741824 | 95.0 | 94.7 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 49774592 | 95.0 | 94.7 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 49807360 | 94.56 | 94.73 | 51.0 | 95.0 | 192.284 | 99.0 |  |
| 49840128 | 94.99 | 94.74 | 94.0 | 95.0 | 192.711 | 99.0 |  |
| 49872896 | 94.25 | 94.71 | 20.0 | 95.0 | 191.971 | 99.0 |  |
| 49905664 | 94.86 | 94.76 | 81.0 | 95.0 | 192.627 | 99.0 |  |
| 49938432 | 93.96 | 94.72 | 37.0 | 95.0 | 190.728 | 98.0 |  |
| 49971200 | 95.0 | 94.73 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 50003968 | 95.0 | 94.73 | 95.0 | 95.0 | 193.757 | 100.0 |  |
