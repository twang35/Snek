# b26d-pen01-seed4

step **50,003,968** · 1526 evals · trailing **94.47** · peak **94.66** @25,690,112 · sef **95.2** · best30 **98.8** @31,457,280

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
| seed | 4 |
| torch_threads | 1 |

![b26d-pen01-seed4](b26d-pen01-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 8.17 | 8.17 | 1.0 | 20.0 | 4.777 | 0.0 |  |
| 65536 | 56.94 | 42.4 | 0.0 | 93.0 | 53.807 | 0.0 |  |
| 98304 | 51.22 | 37.06 | 14.0 | 95.0 | 47.36 | 1.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.22 | 94.42 | 24.0 | 95.0 | 190.955 | 98.0 |  |
| 49676288 | 93.9 | 94.4 | 58.0 | 95.0 | 188.641 | 96.0 |  |
| 49709056 | 95.0 | 94.47 | 95.0 | 95.0 | 193.723 | 100.0 |  |
| 49741824 | 93.6 | 94.44 | 24.0 | 95.0 | 190.344 | 98.0 |  |
| 49774592 | 94.78 | 94.52 | 73.0 | 95.0 | 192.496 | 99.0 |  |
| 49807360 | 93.56 | 94.48 | 14.0 | 95.0 | 189.3 | 97.0 |  |
| 49840128 | 94.91 | 94.43 | 86.0 | 95.0 | 192.629 | 99.0 |  |
| 49872896 | 94.63 | 94.44 | 58.0 | 95.0 | 192.361 | 99.0 |  |
| 49905664 | 94.56 | 94.46 | 58.0 | 95.0 | 191.278 | 98.0 |  |
| 49938432 | 94.58 | 94.48 | 53.0 | 95.0 | 192.307 | 99.0 |  |
| 49971200 | 94.75 | 94.44 | 70.0 | 95.0 | 192.477 | 99.0 |  |
| 50003968 | 94.94 | 94.47 | 89.0 | 95.0 | 192.662 | 99.0 |  |
