# b27c-hist0-seed3

step **100,007,936** · 3052 evals · trailing **94.75** · peak **94.81** @83,427,328 · sef **95.8** · best30 **99.3** @92,602,368

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
| max_steps | 100007936 |
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

![b27c-hist0-seed3](b27c-hist0-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.2 | 0.2 | 0.0 | 3.0 | -4.805 | 0.0 |  |
| 65536 | 11.91 | 6.05 | 0.0 | 41.0 | 8.836 | 0.0 |  |
| 98304 | 33.56 | 15.22 | 8.0 | 62.0 | 28.705 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.95 | 94.77 | 90.0 | 95.0 | 192.673 | 99.0 |  |
| 99680256 | 94.56 | 94.72 | 62.0 | 95.0 | 190.283 | 97.0 |  |
| 99713024 | 95.0 | 94.76 | 95.0 | 95.0 | 193.734 | 100.0 |  |
| 99745792 | 95.0 | 94.72 | 95.0 | 95.0 | 193.724 | 100.0 |  |
| 99778560 | 94.8 | 94.72 | 75.0 | 95.0 | 192.49 | 99.0 |  |
| 99811328 | 95.0 | 94.72 | 95.0 | 95.0 | 193.718 | 100.0 |  |
| 99844096 | 94.01 | 94.72 | 23.0 | 95.0 | 189.66 | 97.0 |  |
| 99876864 | 94.97 | 94.78 | 92.0 | 95.0 | 192.687 | 99.0 |  |
| 99909632 | 94.87 | 94.74 | 82.0 | 95.0 | 192.586 | 99.0 |  |
| 99942400 | 94.91 | 94.74 | 86.0 | 95.0 | 192.624 | 99.0 |  |
| 99975168 | 94.87 | 94.76 | 82.0 | 95.0 | 192.579 | 99.0 |  |
| 100007936 | 93.87 | 94.75 | 61.0 | 95.0 | 186.545 | 94.0 |  |
