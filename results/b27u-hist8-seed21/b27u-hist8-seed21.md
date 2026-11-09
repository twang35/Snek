# b27u-hist8-seed21

step **100,007,936** · 3052 evals · trailing **94.84** · peak **94.9** @76,775,424 · sef **95.8** · best30 **99.8** @76,906,496

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
| seed | 21 |
| torch_threads | 1 |

![b27u-hist8-seed21](b27u-hist8-seed21.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 3.69 | 3.69 | 0.0 | 9.0 | 2.611 | 0.0 |  |
| 65536 | 27.83 | 19.39 | 1.0 | 55.0 | 22.904 | 0.0 |  |
| 98304 | 26.65 | 15.17 | 1.0 | 54.0 | 21.765 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.82 | 95.0 | 95.0 | 193.768 | 100.0 |  |
| 99680256 | 95.0 | 94.84 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 99713024 | 95.0 | 94.82 | 95.0 | 95.0 | 193.773 | 100.0 |  |
| 99745792 | 95.0 | 94.82 | 95.0 | 95.0 | 193.771 | 100.0 |  |
| 99778560 | 95.0 | 94.82 | 95.0 | 95.0 | 193.778 | 100.0 |  |
| 99811328 | 95.0 | 94.84 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 99844096 | 95.0 | 94.84 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 99876864 | 95.0 | 94.84 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99909632 | 94.7 | 94.84 | 65.0 | 95.0 | 192.475 | 99.0 |  |
| 99942400 | 95.0 | 94.84 | 95.0 | 95.0 | 193.774 | 100.0 |  |
| 99975168 | 94.25 | 94.82 | 20.0 | 95.0 | 191.989 | 99.0 |  |
| 100007936 | 95.0 | 94.84 | 95.0 | 95.0 | 193.773 | 100.0 |  |
