# b33ad-win0-seed4

step **50,003,968** · 1526 evals · trailing **94.62** · peak **94.94** @38,797,312 · sef **81.9** · best30 **99.9** @39,092,224

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

![b33ad-win0-seed4](b33ad-win0-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 3.95 | 3.95 | 0.0 | 9.0 | 1.551 | 0.0 |  |
| 65536 | 10.52 | 7.23 | 0.0 | 33.0 | 7.251 | 0.0 |  |
| 98304 | 23.95 | 12.81 | 2.0 | 48.0 | 18.946 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 95.0 | 94.46 | 95.0 | 95.0 | 193.757 | 100.0 |  |
| 49676288 | 95.0 | 94.55 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 49709056 | 95.0 | 94.58 | 95.0 | 95.0 | 193.751 | 100.0 |  |
| 49741824 | 95.0 | 94.58 | 95.0 | 95.0 | 193.757 | 100.0 |  |
| 49774592 | 95.0 | 94.58 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49807360 | 95.0 | 94.6 | 95.0 | 95.0 | 193.765 | 100.0 |  |
| 49840128 | 95.0 | 94.62 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 49872896 | 95.0 | 94.62 | 95.0 | 95.0 | 193.75 | 100.0 |  |
| 49905664 | 95.0 | 94.62 | 95.0 | 95.0 | 193.756 | 100.0 |  |
| 49938432 | 94.28 | 94.62 | 52.0 | 95.0 | 190.966 | 98.0 |  |
| 49971200 | 94.42 | 94.62 | 37.0 | 95.0 | 192.187 | 99.0 |  |
| 50003968 | 94.42 | 94.62 | 37.0 | 95.0 | 192.146 | 99.0 |  |
