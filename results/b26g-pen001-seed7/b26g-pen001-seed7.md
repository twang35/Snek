# b26g-pen001-seed7

step **50,003,968** · 1526 evals · trailing **94.55** · peak **94.6** @40,632,320 · sef **91.3** · best30 **98.3** @40,402,944

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
| seed | 7 |
| torch_threads | 1 |

![b26g-pen001-seed7](b26g-pen001-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 17.62 | 17.62 | 0.0 | 38.0 | 13.309 | 0.0 |  |
| 65536 | 28.46 | 23.04 | 0.0 | 49.0 | 23.667 | 0.0 |  |
| 98304 | 30.51 | 25.53 | 9.0 | 51.0 | 25.503 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.1 | 94.56 | 60.0 | 95.0 | 187.779 | 95.0 |  |
| 49676288 | 95.0 | 94.58 | 95.0 | 95.0 | 193.701 | 100.0 |  |
| 49709056 | 94.48 | 94.57 | 64.0 | 95.0 | 191.202 | 98.0 |  |
| 49741824 | 94.91 | 94.57 | 89.0 | 95.0 | 191.622 | 98.0 |  |
| 49774592 | 94.75 | 94.56 | 70.0 | 95.0 | 192.457 | 99.0 |  |
| 49807360 | 94.67 | 94.58 | 67.0 | 95.0 | 191.352 | 98.0 |  |
| 49840128 | 94.64 | 94.58 | 66.0 | 95.0 | 191.356 | 98.0 |  |
| 49872896 | 95.0 | 94.57 | 95.0 | 95.0 | 193.706 | 100.0 |  |
| 49905664 | 94.13 | 94.56 | 63.0 | 95.0 | 187.822 | 95.0 |  |
| 49938432 | 95.0 | 94.57 | 95.0 | 95.0 | 193.713 | 100.0 |  |
| 49971200 | 94.32 | 94.55 | 64.0 | 95.0 | 189.049 | 96.0 |  |
| 50003968 | 94.87 | 94.55 | 86.0 | 95.0 | 191.586 | 98.0 |  |
