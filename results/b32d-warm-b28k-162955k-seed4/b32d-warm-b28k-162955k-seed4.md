# b32d-warm-b28k-162955k-seed4

step **100,007,936** · 3052 evals · trailing **94.74** · peak **95.0** @32,768 · sef **100.0** · best30 **99.9** @983,040

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.999 |
| eval_interval | 32768 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| init_from | b28k-hist8a25-seed11@162955264 |
| max_steps | 100007936 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_anneal_fraction | 0.5 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_discount_final | 1.0 |
| ppo_entropy_coef | 0.001 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.999 |
| ppo_gae_lambda_final | 1.0 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 500.3 |
| ppo_horizon_final | inf |
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

![b32d-warm-b28k-162955k-seed4](b32d-warm-b28k-162955k-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 95.0 | 95.0 | 95.0 | 95.0 | 193.781 | 100.0 |  |
| 65536 | 95.0 | 95.0 | 95.0 | 95.0 | 193.785 | 100.0 |  |
| 98304 | 95.0 | 95.0 | 95.0 | 95.0 | 193.784 | 100.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.73 | 95.0 | 95.0 | 193.782 | 100.0 |  |
| 99680256 | 94.49 | 94.73 | 44.0 | 95.0 | 192.277 | 99.0 |  |
| 99713024 | 95.0 | 94.78 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 99745792 | 94.43 | 94.76 | 38.0 | 95.0 | 192.169 | 99.0 |  |
| 99778560 | 94.21 | 94.73 | 16.0 | 95.0 | 191.991 | 99.0 |  |
| 99811328 | 95.0 | 94.76 | 95.0 | 95.0 | 193.786 | 100.0 |  |
| 99844096 | 94.49 | 94.73 | 44.0 | 95.0 | 192.279 | 99.0 |  |
| 99876864 | 95.0 | 94.73 | 95.0 | 95.0 | 193.78 | 100.0 |  |
| 99909632 | 94.21 | 94.74 | 16.0 | 95.0 | 191.991 | 99.0 |  |
| 99942400 | 95.0 | 94.74 | 95.0 | 95.0 | 193.783 | 100.0 |  |
| 99975168 | 95.0 | 94.74 | 95.0 | 95.0 | 193.783 | 100.0 |  |
| 100007936 | 95.0 | 94.74 | 95.0 | 95.0 | 193.78 | 100.0 |  |
