# b33ar-win100-seed2

step **50,003,968** · 1526 evals · trailing **94.8** · peak **94.85** @45,449,216 · sef **91.9** · best30 **99.6** @31,096,832

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

![b33ar-win100-seed2](b33ar-win100-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 12.44 | 12.44 | 0.0 | 28.0 | 8.042 | 0.0 |  |
| 65536 | 36.96 | 24.7 | 2.0 | 60.0 | 32.076 | 0.0 |  |
| 98304 | 35.7 | 28.37 | 1.0 | 62.0 | 30.795 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 95.0 | 94.76 | 95.0 | 95.0 | 193.755 | 100.0 |  |
| 49676288 | 95.0 | 94.8 | 95.0 | 95.0 | 193.757 | 100.0 |  |
| 49709056 | 95.0 | 94.74 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49741824 | 95.0 | 94.74 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49774592 | 94.61 | 94.79 | 56.0 | 95.0 | 192.363 | 99.0 |  |
| 49807360 | 95.0 | 94.8 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 49840128 | 94.66 | 94.79 | 61.0 | 95.0 | 192.42 | 99.0 |  |
| 49872896 | 95.0 | 94.79 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49905664 | 94.64 | 94.79 | 59.0 | 95.0 | 192.363 | 99.0 |  |
| 49938432 | 95.0 | 94.79 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 49971200 | 95.0 | 94.8 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 50003968 | 95.0 | 94.8 | 95.0 | 95.0 | 193.76 | 100.0 |  |
