# b33am-win50-seed1

step **50,003,968** · 1526 evals · trailing **94.61** · peak **94.82** @46,333,952 · sef **92.3** · best30 **99.6** @45,875,200

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

![b33am-win50-seed1](b33am-win50-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 6.46 | 6.46 | 0.0 | 20.0 | 1.849 | 0.0 |  |
| 65536 | 28.05 | 17.25 | 0.0 | 51.0 | 23.246 | 0.0 |  |
| 98304 | 29.2 | 22.6 | 10.0 | 53.0 | 24.145 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.39 | 94.5 | 56.0 | 95.0 | 191.165 | 98.0 |  |
| 49676288 | 93.96 | 94.51 | 24.0 | 95.0 | 190.732 | 98.0 |  |
| 49709056 | 94.77 | 94.5 | 72.0 | 95.0 | 192.536 | 99.0 |  |
| 49741824 | 95.0 | 94.55 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 49774592 | 94.58 | 94.58 | 53.0 | 95.0 | 192.305 | 99.0 |  |
| 49807360 | 95.0 | 94.61 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49840128 | 93.72 | 94.55 | 12.0 | 95.0 | 190.499 | 98.0 |  |
| 49872896 | 95.0 | 94.61 | 95.0 | 95.0 | 193.754 | 100.0 |  |
| 49905664 | 94.65 | 94.6 | 60.0 | 95.0 | 192.372 | 99.0 |  |
| 49938432 | 93.22 | 94.54 | 22.0 | 95.0 | 188.906 | 97.0 |  |
| 49971200 | 94.34 | 94.52 | 29.0 | 95.0 | 192.064 | 99.0 |  |
| 50003968 | 94.58 | 94.61 | 53.0 | 95.0 | 192.305 | 99.0 |  |
