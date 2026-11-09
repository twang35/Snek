# b26n-pen0-seed14

step **50,003,968** · 1526 evals · trailing **94.49** · peak **94.64** @45,088,768 · sef **93.3** · best30 **98.7** @49,020,928

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
| seed | 14 |
| torch_threads | 1 |

![b26n-pen0-seed14](b26n-pen0-seed14.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 7.7 | 7.7 | 1.0 | 24.0 | 6.33 | 0.0 |  |
| 65536 | 14.53 | 11.12 | 1.0 | 45.0 | 12.963 | 0.0 |  |
| 98304 | 39.7 | 24.64 | 11.0 | 75.0 | 34.763 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.0 | 94.5 | 58.0 | 95.0 | 187.734 | 95.0 |  |
| 49676288 | 93.25 | 94.44 | 56.0 | 95.0 | 185.988 | 94.0 |  |
| 49709056 | 94.71 | 94.51 | 78.0 | 95.0 | 190.436 | 97.0 |  |
| 49741824 | 94.87 | 94.44 | 82.0 | 95.0 | 192.595 | 99.0 |  |
| 49774592 | 94.66 | 94.46 | 61.0 | 95.0 | 192.368 | 99.0 |  |
| 49807360 | 94.27 | 94.44 | 58.0 | 95.0 | 190.996 | 98.0 |  |
| 49840128 | 94.3 | 94.49 | 58.0 | 95.0 | 190.037 | 97.0 |  |
| 49872896 | 94.63 | 94.44 | 58.0 | 95.0 | 192.363 | 99.0 |  |
| 49905664 | 95.0 | 94.5 | 95.0 | 95.0 | 193.714 | 100.0 |  |
| 49938432 | 94.32 | 94.51 | 58.0 | 95.0 | 191.039 | 98.0 |  |
| 49971200 | 94.25 | 94.5 | 34.0 | 95.0 | 190.943 | 98.0 |  |
| 50003968 | 94.47 | 94.49 | 73.0 | 95.0 | 188.168 | 95.0 |  |
