# b33ba-win300-seed3

step **50,003,968** · 1526 evals · trailing **94.72** · peak **94.84** @42,237,952 · sef **89.1** · best30 **99.6** @42,762,240

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
| seed | 3 |
| torch_threads | 1 |

![b33ba-win300-seed3](b33ba-win300-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 2.69 | 2.69 | 0.0 | 11.0 | 0.561 | 0.0 |  |
| 65536 | 16.85 | 9.77 | 0.0 | 45.0 | 13.076 | 0.0 |  |
| 98304 | 27.03 | 15.52 | 1.0 | 59.0 | 22.191 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.47 | 94.65 | 42.0 | 95.0 | 192.195 | 99.0 |  |
| 49676288 | 95.0 | 94.67 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 49709056 | 95.0 | 94.69 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 49741824 | 95.0 | 94.67 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 49774592 | 95.0 | 94.69 | 95.0 | 95.0 | 193.754 | 100.0 |  |
| 49807360 | 94.38 | 94.67 | 33.0 | 95.0 | 192.114 | 99.0 |  |
| 49840128 | 94.48 | 94.65 | 43.0 | 95.0 | 192.247 | 99.0 |  |
| 49872896 | 95.0 | 94.69 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 49905664 | 95.0 | 94.72 | 95.0 | 95.0 | 193.757 | 100.0 |  |
| 49938432 | 94.63 | 94.67 | 58.0 | 95.0 | 192.394 | 99.0 |  |
| 49971200 | 94.92 | 94.75 | 87.0 | 95.0 | 192.687 | 99.0 |  |
| 50003968 | 95.0 | 94.72 | 95.0 | 95.0 | 193.762 | 100.0 |  |
