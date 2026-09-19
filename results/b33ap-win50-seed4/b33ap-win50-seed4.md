# b33ap-win50-seed4

step **50,003,968** · 1526 evals · trailing **94.67** · peak **94.82** @30,474,240 · sef **90.3** · best30 **99.6** @43,941,888

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

![b33ap-win50-seed4](b33ap-win50-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.42 | 4.42 | 0.0 | 10.0 | 1.486 | 0.0 |  |
| 65536 | 10.7 | 7.56 | 0.0 | 36.0 | 7.562 | 0.0 |  |
| 98304 | 23.85 | 12.99 | 0.0 | 46.0 | 19.021 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 95.0 | 94.67 | 95.0 | 95.0 | 193.774 | 100.0 |  |
| 49676288 | 95.0 | 94.69 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 49709056 | 95.0 | 94.72 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 49741824 | 94.64 | 94.7 | 59.0 | 95.0 | 192.41 | 99.0 |  |
| 49774592 | 95.0 | 94.7 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 49807360 | 95.0 | 94.7 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 49840128 | 95.0 | 94.7 | 95.0 | 95.0 | 193.77 | 100.0 |  |
| 49872896 | 94.36 | 94.7 | 31.0 | 95.0 | 192.08 | 99.0 |  |
| 49905664 | 95.0 | 94.7 | 95.0 | 95.0 | 193.771 | 100.0 |  |
| 49938432 | 93.73 | 94.66 | 18.0 | 95.0 | 190.459 | 98.0 |  |
| 49971200 | 94.23 | 94.67 | 18.0 | 95.0 | 191.96 | 99.0 |  |
| 50003968 | 95.0 | 94.67 | 95.0 | 95.0 | 193.762 | 100.0 |  |
