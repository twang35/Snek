# b26j-pen0001-seed10

step **46,563,328** · 1418 evals · trailing **94.42** · peak **94.7** @43,024,384 · sef **93.7** · best30 **98.7** @39,157,760

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
| seed | 10 |
| torch_threads | 1 |

![b26j-pen0001-seed10](b26j-pen0001-seed10.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.73 | 0.73 | 0.0 | 6.0 | 0.175 | 0.0 |  |
| 65536 | 6.16 | 3.45 | 1.0 | 28.0 | 5.548 | 0.0 |  |
| 98304 | 40.45 | 15.78 | 1.0 | 72.0 | 35.9 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 46104576 | 94.97 | 94.38 | 92.0 | 95.0 | 192.687 | 99.0 |  |
| 46137344 | 93.45 | 94.31 | 14.0 | 95.0 | 188.128 | 96.0 |  |
| 46170112 | 94.42 | 94.33 | 63.0 | 95.0 | 191.138 | 98.0 |  |
| 46202880 | 94.27 | 94.37 | 58.0 | 95.0 | 190.994 | 98.0 |  |
| 46235648 | 94.43 | 94.32 | 64.0 | 95.0 | 191.165 | 98.0 |  |
| 46268416 | 94.21 | 94.39 | 33.0 | 95.0 | 189.896 | 97.0 |  |
| 46301184 | 94.65 | 94.45 | 73.0 | 95.0 | 191.383 | 98.0 |  |
| 46432256 | 94.78 | 94.42 | 84.0 | 95.0 | 191.498 | 98.0 |  |
| 46465024 | 94.37 | 94.4 | 58.0 | 95.0 | 191.046 | 98.0 |  |
| 46497792 | 95.0 | 94.4 | 95.0 | 95.0 | 193.709 | 100.0 |  |
| 46530560 | 94.83 | 94.47 | 85.0 | 95.0 | 191.547 | 98.0 |  |
| 46563328 | 94.57 | 94.42 | 63.0 | 95.0 | 190.302 | 97.0 |  |
