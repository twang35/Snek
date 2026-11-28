# b31d-mse-seed4

step **100,007,936** · 3052 evals · trailing **94.52** · peak **94.9** @82,739,200 · sef **97.1** · best30 **99.8** @82,739,200

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
| seed | 4 |
| torch_threads | 1 |

![b31d-mse-seed4](b31d-mse-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.71 | 4.71 | 0.0 | 10.0 | 1.908 | 0.0 |  |
| 65536 | 13.6 | 9.15 | 0.0 | 31.0 | 9.604 | 0.0 |  |
| 98304 | 23.07 | 13.79 | 4.0 | 44.0 | 18.073 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.1 | 94.62 | 46.0 | 95.0 | 190.868 | 98.0 |  |
| 99680256 | 94.15 | 94.59 | 45.0 | 95.0 | 190.879 | 98.0 |  |
| 99713024 | 94.65 | 94.53 | 60.0 | 95.0 | 192.427 | 99.0 |  |
| 99745792 | 94.49 | 94.63 | 44.0 | 95.0 | 192.214 | 99.0 |  |
| 99778560 | 95.0 | 94.65 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99811328 | 93.48 | 94.58 | 12.0 | 95.0 | 190.22 | 98.0 |  |
| 99844096 | 93.42 | 94.52 | 12.0 | 95.0 | 190.207 | 98.0 |  |
| 99876864 | 94.88 | 94.54 | 83.0 | 95.0 | 192.647 | 99.0 |  |
| 99909632 | 94.5 | 94.51 | 45.0 | 95.0 | 192.274 | 99.0 |  |
| 99942400 | 95.0 | 94.55 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99975168 | 95.0 | 94.52 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 100007936 | 94.25 | 94.52 | 20.0 | 95.0 | 192.019 | 99.0 |  |
