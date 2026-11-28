# b32f-warm-b28m-138314k-seed6

step **100,007,936** · 3052 evals · trailing **94.69** · peak **95.0** @65,536 · sef **100.0** · best30 **99.9** @2,686,976

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
| init_from | b28m-hist8a25-seed13@138313728 |
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
| seed | 6 |
| torch_threads | 1 |

![b32f-warm-b28m-138314k-seed6](b32f-warm-b28m-138314k-seed6.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 94.02 | 94.4 | 10.0 | 95.0 | 190.797 | 98.0 |  |
| 65536 | 95.0 | 95.0 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 98304 | 94.18 | 94.59 | 13.0 | 95.0 | 191.915 | 99.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.59 | 94.76 | 54.0 | 95.0 | 192.372 | 99.0 |  |
| 99680256 | 95.0 | 94.68 | 95.0 | 95.0 | 193.776 | 100.0 |  |
| 99713024 | 93.69 | 94.64 | 20.0 | 95.0 | 190.43 | 98.0 |  |
| 99745792 | 94.45 | 94.74 | 46.0 | 95.0 | 191.228 | 98.0 |  |
| 99778560 | 94.4 | 94.72 | 35.0 | 95.0 | 192.137 | 99.0 |  |
| 99811328 | 94.29 | 94.7 | 24.0 | 95.0 | 192.028 | 99.0 |  |
| 99844096 | 94.49 | 94.68 | 44.0 | 95.0 | 192.276 | 99.0 |  |
| 99876864 | 95.0 | 94.7 | 95.0 | 95.0 | 193.778 | 100.0 |  |
| 99909632 | 93.85 | 94.6 | 37.0 | 95.0 | 190.59 | 98.0 |  |
| 99942400 | 95.0 | 94.6 | 95.0 | 95.0 | 193.773 | 100.0 |  |
| 99975168 | 94.37 | 94.68 | 32.0 | 95.0 | 192.106 | 99.0 |  |
| 100007936 | 95.0 | 94.69 | 95.0 | 95.0 | 193.774 | 100.0 |  |
