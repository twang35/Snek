# b31e-mse-seed5

step **100,007,936** · 3052 evals · trailing **94.63** · peak **94.92** @51,478,528 · sef **97.5** · best30 **99.9** @51,511,296

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
| ppo_value_loss | mse |
| ppo_vf_coef | 0.5 |
| seed | 5 |
| torch_threads | 1 |

![b31e-mse-seed5](b31e-mse-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 6.31 | 6.31 | 0.0 | 19.0 | 2.186 | 0.0 |  |
| 65536 | 20.09 | 13.2 | 2.0 | 40.0 | 15.23 | 0.0 |  |
| 98304 | 23.11 | 16.5 | 3.0 | 40.0 | 18.211 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.19 | 94.65 | 14.0 | 95.0 | 191.971 | 99.0 |  |
| 99680256 | 94.17 | 94.64 | 12.0 | 95.0 | 191.952 | 99.0 |  |
| 99713024 | 95.0 | 94.66 | 95.0 | 95.0 | 193.774 | 100.0 |  |
| 99745792 | 95.0 | 94.66 | 95.0 | 95.0 | 193.773 | 100.0 |  |
| 99778560 | 94.67 | 94.65 | 62.0 | 95.0 | 192.455 | 99.0 |  |
| 99811328 | 94.08 | 94.61 | 3.0 | 95.0 | 191.869 | 99.0 |  |
| 99844096 | 95.0 | 94.64 | 95.0 | 95.0 | 193.771 | 100.0 |  |
| 99876864 | 95.0 | 94.64 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 99909632 | 95.0 | 94.64 | 95.0 | 95.0 | 193.776 | 100.0 |  |
| 99942400 | 94.45 | 94.63 | 40.0 | 95.0 | 192.227 | 99.0 |  |
| 99975168 | 94.31 | 94.6 | 26.0 | 95.0 | 192.047 | 99.0 |  |
| 100007936 | 95.0 | 94.63 | 95.0 | 95.0 | 193.774 | 100.0 |  |
