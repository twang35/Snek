# b33az-win300-seed2

step **50,003,968** · 1526 evals · trailing **94.69** · peak **94.88** @37,322,752 · sef **91.0** · best30 **99.8** @37,453,824

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

![b33az-win300-seed2](b33az-win300-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 12.58 | 12.58 | 1.0 | 28.0 | 8.093 | 0.0 |  |
| 65536 | 37.52 | 27.38 | 1.0 | 73.0 | 32.709 | 0.0 |  |
| 98304 | 32.75 | 28.72 | 1.0 | 60.0 | 27.901 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 95.0 | 94.74 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 49676288 | 95.0 | 94.77 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 49709056 | 94.15 | 94.75 | 10.0 | 95.0 | 191.927 | 99.0 |  |
| 49741824 | 95.0 | 94.77 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 49774592 | 93.73 | 94.74 | 30.0 | 95.0 | 190.42 | 98.0 |  |
| 49807360 | 94.45 | 94.72 | 40.0 | 95.0 | 192.175 | 99.0 |  |
| 49840128 | 95.0 | 94.72 | 95.0 | 95.0 | 193.773 | 100.0 |  |
| 49872896 | 95.0 | 94.73 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 49905664 | 93.66 | 94.71 | 8.0 | 95.0 | 190.392 | 98.0 |  |
| 49938432 | 95.0 | 94.73 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 49971200 | 95.0 | 94.71 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 50003968 | 94.49 | 94.69 | 44.0 | 95.0 | 192.214 | 99.0 |  |
