# b33aa-win0-seed1

step **50,003,968** · 1526 evals · trailing **94.48** · peak **94.89** @42,860,544 · sef **83.1** · best30 **99.8** @42,827,776

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

![b33aa-win0-seed1](b33aa-win0-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 7.26 | 7.26 | 0.0 | 23.0 | 2.558 | 0.0 |  |
| 65536 | 28.03 | 22.48 | 1.0 | 60.0 | 23.256 | 0.0 |  |
| 98304 | 28.71 | 20.63 | 1.0 | 56.0 | 23.699 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.93 | 94.58 | 88.0 | 95.0 | 192.688 | 99.0 |  |
| 49676288 | 94.21 | 94.5 | 16.0 | 95.0 | 191.934 | 99.0 |  |
| 49709056 | 95.0 | 94.52 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 49741824 | 95.0 | 94.54 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 49774592 | 94.41 | 94.52 | 36.0 | 95.0 | 192.13 | 99.0 |  |
| 49807360 | 93.76 | 94.51 | 32.0 | 95.0 | 190.532 | 98.0 |  |
| 49840128 | 94.17 | 94.53 | 12.0 | 95.0 | 191.934 | 99.0 |  |
| 49872896 | 94.03 | 94.5 | 1.0 | 95.0 | 190.752 | 98.0 |  |
| 49905664 | 95.0 | 94.53 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 49938432 | 95.0 | 94.56 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 49971200 | 94.13 | 94.55 | 8.0 | 95.0 | 191.891 | 99.0 |  |
| 50003968 | 92.98 | 94.48 | 3.0 | 95.0 | 188.711 | 97.0 |  |
