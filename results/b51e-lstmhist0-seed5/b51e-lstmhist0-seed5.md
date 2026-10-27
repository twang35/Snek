# b51e-lstmhist0-seed5

step **100,007,936** · 3052 evals · trailing **94.4** · peak **94.84** @53,313,536 · sef **96.7** · best30 **99.5** @54,132,736

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
| seed | 5 |
| torch_threads | 1 |

![b51e-lstmhist0-seed5](b51e-lstmhist0-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 19.54 | 19.54 | 0.0 | 40.0 | 15.0 | 0.0 |  |
| 65536 | 30.09 | 24.81 | 2.0 | 60.0 | 25.097 | 0.0 |  |
| 98304 | 29.44 | 26.36 | 6.0 | 54.0 | 24.616 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.16 | 95.0 | 95.0 | 193.782 | 100.0 |  |
| 99680256 | 94.17 | 94.16 | 12.0 | 95.0 | 191.917 | 99.0 |  |
| 99713024 | 94.44 | 94.21 | 48.0 | 95.0 | 191.141 | 98.0 |  |
| 99745792 | 95.0 | 94.23 | 95.0 | 95.0 | 193.781 | 100.0 |  |
| 99778560 | 93.58 | 94.24 | 1.0 | 95.0 | 190.328 | 98.0 |  |
| 99811328 | 95.0 | 94.27 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 99844096 | 95.0 | 94.33 | 95.0 | 95.0 | 193.785 | 100.0 |  |
| 99876864 | 94.89 | 94.37 | 84.0 | 95.0 | 192.671 | 99.0 |  |
| 99909632 | 94.1 | 94.4 | 47.0 | 95.0 | 190.806 | 98.0 |  |
| 99942400 | 94.29 | 94.38 | 57.0 | 95.0 | 190.996 | 98.0 |  |
| 99975168 | 95.0 | 94.41 | 95.0 | 95.0 | 193.785 | 100.0 |  |
| 100007936 | 94.23 | 94.4 | 18.0 | 95.0 | 191.971 | 99.0 |  |
