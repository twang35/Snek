# b27i-hist4-seed9

step **100,007,936** · 3052 evals · trailing **94.77** · peak **94.89** @72,089,600 · sef **97.0** · best30 **99.8** @95,223,808

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
| seed | 9 |
| torch_threads | 1 |

![b27i-hist4-seed9](b27i-hist4-seed9.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 5.1 | 5.1 | 0.0 | 18.0 | 3.683 | 0.0 |  |
| 65536 | 30.47 | 17.79 | 0.0 | 48.0 | 25.534 | 0.0 |  |
| 98304 | 34.42 | 24.75 | 15.0 | 60.0 | 29.393 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.82 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 99680256 | 95.0 | 94.82 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 99713024 | 94.19 | 94.79 | 14.0 | 95.0 | 191.952 | 99.0 |  |
| 99745792 | 94.25 | 94.8 | 20.0 | 95.0 | 192.019 | 99.0 |  |
| 99778560 | 94.68 | 94.78 | 63.0 | 95.0 | 192.445 | 99.0 |  |
| 99811328 | 95.0 | 94.8 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 99844096 | 95.0 | 94.8 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 99876864 | 95.0 | 94.77 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 99909632 | 95.0 | 94.77 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 99942400 | 95.0 | 94.8 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 99975168 | 95.0 | 94.8 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 100007936 | 93.77 | 94.77 | 10.0 | 95.0 | 190.494 | 98.0 |  |
