# b28j-hist8a25-seed10

step **200,015,872** · 6104 evals · trailing **94.84** · peak **94.93** @185,401,344 · sef **97.8** · best30 **99.9** @185,401,344

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
| max_steps | 200015872 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_anneal_fraction | 0.25 |
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

![b28j-hist8a25-seed10](b28j-hist8a25-seed10.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 11.18 | 11.18 | 1.0 | 31.0 | 7.697 | 0.0 |  |
| 65536 | 30.41 | 26.29 | 1.0 | 57.0 | 25.657 | 0.0 |  |
| 98304 | 30.76 | 20.97 | 3.0 | 52.0 | 25.742 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 199655424 | 95.0 | 94.87 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 199688192 | 95.0 | 94.87 | 95.0 | 95.0 | 193.778 | 100.0 |  |
| 199720960 | 94.33 | 94.85 | 28.0 | 95.0 | 192.07 | 99.0 |  |
| 199753728 | 95.0 | 94.85 | 95.0 | 95.0 | 193.777 | 100.0 |  |
| 199786496 | 95.0 | 94.85 | 95.0 | 95.0 | 193.781 | 100.0 |  |
| 199819264 | 95.0 | 94.85 | 95.0 | 95.0 | 193.784 | 100.0 |  |
| 199852032 | 94.62 | 94.84 | 57.0 | 95.0 | 192.406 | 99.0 |  |
| 199884800 | 95.0 | 94.84 | 95.0 | 95.0 | 193.781 | 100.0 |  |
| 199917568 | 94.4 | 94.84 | 35.0 | 95.0 | 192.184 | 99.0 |  |
| 199950336 | 95.0 | 94.86 | 95.0 | 95.0 | 193.784 | 100.0 |  |
| 199983104 | 95.0 | 94.84 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 200015872 | 94.27 | 94.84 | 22.0 | 95.0 | 192.057 | 99.0 |  |
