# b51g-lstmhist0-seed7

step **100,007,936** · 3052 evals · trailing **94.54** · peak **94.81** @20,643,840 · sef **96.0** · best30 **99.5** @41,385,984

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
| init_from | None |
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
| ppo_recurrent | lstm |
| ppo_recurrent_hidden | 128 |
| ppo_rollout | 256 |
| ppo_seq_minibatch | 2 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 32768 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 7 |
| torch_threads | 1 |

![b51g-lstmhist0-seed7](b51g-lstmhist0-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 10.46 | 10.46 | 0.0 | 41.0 | 8.971 | 0.0 |  |
| 65536 | 37.27 | 23.87 | 4.0 | 75.0 | 32.218 | 0.0 |  |
| 98304 | 28.59 | 25.44 | 12.0 | 51.0 | 23.55 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.43 | 94.48 | 38.0 | 95.0 | 192.171 | 99.0 |  |
| 99680256 | 95.0 | 94.5 | 95.0 | 95.0 | 193.78 | 100.0 |  |
| 99713024 | 95.0 | 94.52 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 99745792 | 95.0 | 94.52 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 99778560 | 95.0 | 94.53 | 95.0 | 95.0 | 193.777 | 100.0 |  |
| 99811328 | 93.22 | 94.48 | 10.0 | 95.0 | 188.92 | 97.0 |  |
| 99844096 | 94.43 | 94.46 | 38.0 | 95.0 | 192.172 | 99.0 |  |
| 99876864 | 95.0 | 94.46 | 95.0 | 95.0 | 193.776 | 100.0 |  |
| 99909632 | 94.47 | 94.47 | 42.0 | 95.0 | 192.21 | 99.0 |  |
| 99942400 | 94.27 | 94.47 | 22.0 | 95.0 | 192.008 | 99.0 |  |
| 99975168 | 95.0 | 94.54 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 100007936 | 94.53 | 94.54 | 48.0 | 95.0 | 192.271 | 99.0 |  |
