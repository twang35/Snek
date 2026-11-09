# b33ay-win300-seed1

step **50,003,968** · 1526 evals · trailing **94.76** · peak **94.92** @40,108,032 · sef **87.9** · best30 **99.8** @40,075,264

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
| seed | 1 |
| torch_threads | 1 |

![b33ay-win300-seed1](b33ay-win300-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 6.39 | 6.39 | 0.0 | 23.0 | 1.691 | 0.0 |  |
| 65536 | 28.06 | 20.29 | 0.0 | 56.0 | 23.2 | 0.0 |  |
| 98304 | 29.13 | 22.5 | 2.0 | 57.0 | 24.119 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.26 | 94.78 | 53.0 | 95.0 | 190.986 | 98.0 |  |
| 49676288 | 95.0 | 94.78 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 49709056 | 95.0 | 94.78 | 95.0 | 95.0 | 193.749 | 100.0 |  |
| 49741824 | 94.29 | 94.77 | 24.0 | 95.0 | 192.053 | 99.0 |  |
| 49774592 | 95.0 | 94.77 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 49807360 | 95.0 | 94.77 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 49840128 | 95.0 | 94.77 | 95.0 | 95.0 | 193.757 | 100.0 |  |
| 49872896 | 95.0 | 94.77 | 95.0 | 95.0 | 193.753 | 100.0 |  |
| 49905664 | 94.35 | 94.74 | 30.0 | 95.0 | 192.114 | 99.0 |  |
| 49938432 | 94.05 | 94.79 | 46.0 | 95.0 | 190.813 | 98.0 |  |
| 49971200 | 93.88 | 94.75 | 32.0 | 95.0 | 190.638 | 98.0 |  |
| 50003968 | 95.0 | 94.76 | 95.0 | 95.0 | 193.752 | 100.0 |  |
