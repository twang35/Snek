# b27s-hist8-seed19

step **100,007,936** · 3052 evals · trailing **94.7** · peak **94.96** @59,277,312 · sef **96.4** · best30 **99.9** @59,277,312

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
| seed | 19 |
| torch_threads | 1 |

![b27s-hist8-seed19](b27s-hist8-seed19.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.43 | 0.43 | 0.0 | 4.0 | -0.175 | 0.0 |  |
| 65536 | 13.65 | 13.33 | 0.0 | 50.0 | 11.278 | 0.0 |  |
| 98304 | 25.92 | 13.18 | 6.0 | 43.0 | 20.872 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.6 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99680256 | 95.0 | 94.6 | 95.0 | 95.0 | 193.774 | 100.0 |  |
| 99713024 | 93.64 | 94.62 | 10.0 | 95.0 | 190.382 | 98.0 |  |
| 99745792 | 95.0 | 94.65 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99778560 | 95.0 | 94.65 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 99811328 | 95.0 | 94.66 | 95.0 | 95.0 | 193.783 | 100.0 |  |
| 99844096 | 95.0 | 94.69 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 99876864 | 95.0 | 94.69 | 95.0 | 95.0 | 193.774 | 100.0 |  |
| 99909632 | 95.0 | 94.69 | 95.0 | 95.0 | 193.776 | 100.0 |  |
| 99942400 | 95.0 | 94.71 | 95.0 | 95.0 | 193.771 | 100.0 |  |
| 99975168 | 93.75 | 94.67 | 25.0 | 95.0 | 190.446 | 98.0 |  |
| 100007936 | 95.0 | 94.7 | 95.0 | 95.0 | 193.774 | 100.0 |  |
