# b32h-warm-b28o-136479k-seed8

step **100,007,936** · 3052 evals · trailing **94.59** · peak **94.97** @393,216 · sef **100.0** · best30 **99.9** @37,617,664

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
| init_from | b28o-hist8a25-seed15@136478720 |
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
| seed | 8 |
| torch_threads | 1 |

![b32h-warm-b28o-136479k-seed8](b32h-warm-b28o-136479k-seed8.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 94.67 | 94.67 | 62.0 | 95.0 | 192.455 | 99.0 |  |
| 65536 | 95.0 | 94.84 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 98304 | 95.0 | 94.89 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 93.75 | 94.48 | 20.0 | 95.0 | 190.447 | 98.0 |  |
| 99680256 | 94.99 | 94.47 | 94.0 | 95.0 | 192.722 | 99.0 |  |
| 99713024 | 95.0 | 94.48 | 95.0 | 95.0 | 193.781 | 100.0 |  |
| 99745792 | 94.43 | 94.46 | 38.0 | 95.0 | 192.172 | 99.0 |  |
| 99778560 | 95.0 | 94.5 | 95.0 | 95.0 | 193.778 | 100.0 |  |
| 99811328 | 94.16 | 94.55 | 11.0 | 95.0 | 191.938 | 99.0 |  |
| 99844096 | 95.0 | 94.53 | 95.0 | 95.0 | 193.783 | 100.0 |  |
| 99876864 | 94.51 | 94.52 | 46.0 | 95.0 | 192.289 | 99.0 |  |
| 99909632 | 95.0 | 94.56 | 95.0 | 95.0 | 193.773 | 100.0 |  |
| 99942400 | 95.0 | 94.55 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 99975168 | 95.0 | 94.55 | 95.0 | 95.0 | 193.777 | 100.0 |  |
| 100007936 | 94.27 | 94.59 | 22.0 | 95.0 | 192.048 | 99.0 |  |
