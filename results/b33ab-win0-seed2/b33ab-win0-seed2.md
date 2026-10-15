# b33ab-win0-seed2

step **50,003,968** · 1526 evals · trailing **94.58** · peak **94.88** @42,205,184 · sef **84.9** · best30 **99.8** @42,205,184

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
| seed | 2 |
| torch_threads | 1 |

![b33ab-win0-seed2](b33ab-win0-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 11.75 | 11.75 | 0.0 | 28.0 | 7.266 | 0.0 |  |
| 65536 | 34.97 | 26.38 | 2.0 | 65.0 | 30.136 | 0.0 |  |
| 98304 | 32.43 | 22.09 | 12.0 | 68.0 | 27.444 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.99 | 94.62 | 94.0 | 95.0 | 192.699 | 99.0 |  |
| 49676288 | 94.95 | 94.62 | 90.0 | 95.0 | 192.699 | 99.0 |  |
| 49709056 | 95.0 | 94.56 | 95.0 | 95.0 | 193.755 | 100.0 |  |
| 49741824 | 94.19 | 94.59 | 14.0 | 95.0 | 191.949 | 99.0 |  |
| 49774592 | 95.0 | 94.59 | 95.0 | 95.0 | 193.75 | 100.0 |  |
| 49807360 | 93.94 | 94.55 | 16.0 | 95.0 | 190.648 | 98.0 |  |
| 49840128 | 93.23 | 94.57 | 14.0 | 95.0 | 186.878 | 95.0 |  |
| 49872896 | 94.37 | 94.56 | 32.0 | 95.0 | 192.129 | 99.0 |  |
| 49905664 | 95.0 | 94.58 | 95.0 | 95.0 | 193.757 | 100.0 |  |
| 49938432 | 95.0 | 94.6 | 95.0 | 95.0 | 193.754 | 100.0 |  |
| 49971200 | 94.29 | 94.6 | 58.0 | 95.0 | 191.049 | 98.0 |  |
| 50003968 | 94.36 | 94.58 | 31.0 | 95.0 | 192.084 | 99.0 |  |
