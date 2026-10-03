# b33bf-win1000-seed4

step **50,003,968** · 1526 evals · trailing **94.77** · peak **94.89** @48,594,944 · sef **75.8** · best30 **99.7** @48,660,480

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
| max_steps | 50003968 |
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
| seed | 4 |
| torch_threads | 1 |

![b33bf-win1000-seed4](b33bf-win1000-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.15 | 4.15 | 0.0 | 11.0 | 1.526 | 0.0 |  |
| 65536 | 12.27 | 8.21 | 0.0 | 26.0 | 8.809 | 0.0 |  |
| 98304 | 23.66 | 13.36 | 0.0 | 49.0 | 18.832 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.4 | 94.62 | 35.0 | 95.0 | 192.163 | 99.0 |  |
| 49676288 | 95.0 | 94.64 | 95.0 | 95.0 | 193.765 | 100.0 |  |
| 49709056 | 95.0 | 94.66 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 49741824 | 94.49 | 94.64 | 44.0 | 95.0 | 192.251 | 99.0 |  |
| 49774592 | 94.62 | 94.64 | 57.0 | 95.0 | 192.386 | 99.0 |  |
| 49807360 | 93.95 | 94.61 | 27.0 | 95.0 | 190.721 | 98.0 |  |
| 49840128 | 95.0 | 94.64 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49872896 | 95.0 | 94.66 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 49905664 | 95.0 | 94.73 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 49938432 | 95.0 | 94.76 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 49971200 | 95.0 | 94.77 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 50003968 | 95.0 | 94.77 | 95.0 | 95.0 | 193.762 | 100.0 |  |
