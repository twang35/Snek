# b27t-hist8-seed20

step **100,007,936** · 3052 evals · trailing **94.77** · peak **94.89** @62,423,040 · sef **96.4** · best30 **99.8** @92,340,224

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
| seed | 20 |
| torch_threads | 1 |

![b27t-hist8-seed20](b27t-hist8-seed20.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.07 | 0.07 | 0.0 | 1.0 | -0.479 | 0.0 |  |
| 65536 | 0.94 | 0.51 | 0.0 | 8.0 | 0.256 | 0.0 |  |
| 98304 | 8.62 | 3.21 | 0.0 | 27.0 | 5.688 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.78 | 95.0 | 95.0 | 193.785 | 100.0 |  |
| 99680256 | 95.0 | 94.8 | 95.0 | 95.0 | 193.773 | 100.0 |  |
| 99713024 | 95.0 | 94.8 | 95.0 | 95.0 | 193.783 | 100.0 |  |
| 99745792 | 95.0 | 94.8 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 99778560 | 95.0 | 94.8 | 95.0 | 95.0 | 193.779 | 100.0 |  |
| 99811328 | 95.0 | 94.8 | 95.0 | 95.0 | 193.773 | 100.0 |  |
| 99844096 | 95.0 | 94.8 | 95.0 | 95.0 | 193.77 | 100.0 |  |
| 99876864 | 94.37 | 94.75 | 32.0 | 95.0 | 192.1 | 99.0 |  |
| 99909632 | 94.11 | 94.77 | 6.0 | 95.0 | 191.898 | 99.0 |  |
| 99942400 | 95.0 | 94.77 | 95.0 | 95.0 | 193.776 | 100.0 |  |
| 99975168 | 95.0 | 94.77 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 100007936 | 95.0 | 94.77 | 95.0 | 95.0 | 193.782 | 100.0 |  |
