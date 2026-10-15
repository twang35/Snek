# b27w-hist8-seed23

step **100,007,936** · 3052 evals · trailing **94.89** · peak **94.91** @99,778,560 · sef **94.6** · best30 **99.8** @99,811,328

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
| seed | 23 |
| torch_threads | 1 |

![b27w-hist8-seed23](b27w-hist8-seed23.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 1.53 | 1.53 | 0.0 | 4.0 | -3.478 | 0.0 |  |
| 65536 | 6.78 | 4.16 | 2.0 | 20.0 | 2.653 | 0.0 |  |
| 98304 | 15.69 | 8.0 | 3.0 | 32.0 | 10.937 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.86 | 95.0 | 95.0 | 193.78 | 100.0 |  |
| 99680256 | 94.08 | 94.83 | 48.0 | 95.0 | 190.858 | 98.0 |  |
| 99713024 | 95.0 | 94.86 | 95.0 | 95.0 | 193.768 | 100.0 |  |
| 99745792 | 94.23 | 94.85 | 18.0 | 95.0 | 192.016 | 99.0 |  |
| 99778560 | 95.0 | 94.91 | 95.0 | 95.0 | 193.778 | 100.0 |  |
| 99811328 | 95.0 | 94.9 | 95.0 | 95.0 | 193.781 | 100.0 |  |
| 99844096 | 94.82 | 94.89 | 77.0 | 95.0 | 192.558 | 99.0 |  |
| 99876864 | 95.0 | 94.91 | 95.0 | 95.0 | 193.778 | 100.0 |  |
| 99909632 | 94.49 | 94.89 | 44.0 | 95.0 | 192.225 | 99.0 |  |
| 99942400 | 94.11 | 94.86 | 41.0 | 95.0 | 190.897 | 98.0 |  |
| 99975168 | 95.0 | 94.9 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 100007936 | 95.0 | 94.89 | 95.0 | 95.0 | 193.779 | 100.0 |  |
