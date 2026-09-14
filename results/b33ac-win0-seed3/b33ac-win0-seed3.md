# b33ac-win0-seed3

step **50,003,968** · 1526 evals · trailing **94.66** · peak **94.86** @25,296,896 · sef **80.5** · best30 **99.6** @25,100,288

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
| seed | 3 |
| torch_threads | 1 |

![b33ac-win0-seed3](b33ac-win0-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 2.86 | 2.86 | 0.0 | 10.0 | 1.041 | 0.0 |  |
| 65536 | 18.63 | 10.74 | 0.0 | 48.0 | 14.262 | 0.0 |  |
| 98304 | 27.09 | 16.19 | 1.0 | 49.0 | 22.383 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.95 | 94.55 | 90.0 | 95.0 | 192.668 | 99.0 |  |
| 49676288 | 95.0 | 94.55 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 49709056 | 95.0 | 94.56 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 49741824 | 94.22 | 94.57 | 17.0 | 95.0 | 191.978 | 99.0 |  |
| 49774592 | 95.0 | 94.6 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 49807360 | 94.22 | 94.62 | 18.0 | 95.0 | 190.948 | 98.0 |  |
| 49840128 | 95.0 | 94.63 | 95.0 | 95.0 | 193.765 | 100.0 |  |
| 49872896 | 93.92 | 94.62 | 32.0 | 95.0 | 190.604 | 98.0 |  |
| 49905664 | 95.0 | 94.62 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 49938432 | 95.0 | 94.64 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49971200 | 94.68 | 94.65 | 63.0 | 95.0 | 192.44 | 99.0 |  |
| 50003968 | 95.0 | 94.66 | 95.0 | 95.0 | 193.767 | 100.0 |  |
