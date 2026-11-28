# b33ak-win30-seed3

step **50,003,968** · 1526 evals · trailing **94.69** · peak **94.84** @45,809,664 · sef **90.6** · best30 **99.5** @45,416,448

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

![b33ak-win30-seed3](b33ak-win30-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 3.25 | 3.25 | 0.0 | 10.0 | 0.761 | 0.0 |  |
| 65536 | 15.05 | 17.84 | 0.0 | 50.0 | 11.417 | 0.0 |  |
| 98304 | 26.73 | 14.99 | 2.0 | 46.0 | 21.929 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 95.0 | 94.63 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 49676288 | 94.19 | 94.6 | 14.0 | 95.0 | 191.914 | 99.0 |  |
| 49709056 | 95.0 | 94.65 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49741824 | 95.0 | 94.63 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 49774592 | 95.0 | 94.7 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49807360 | 94.61 | 94.65 | 56.0 | 95.0 | 192.33 | 99.0 |  |
| 49840128 | 94.65 | 94.69 | 60.0 | 95.0 | 192.377 | 99.0 |  |
| 49872896 | 94.25 | 94.66 | 20.0 | 95.0 | 192.019 | 99.0 |  |
| 49905664 | 95.0 | 94.68 | 95.0 | 95.0 | 193.765 | 100.0 |  |
| 49938432 | 94.7 | 94.68 | 66.0 | 95.0 | 191.382 | 98.0 |  |
| 49971200 | 93.92 | 94.66 | 4.0 | 95.0 | 190.689 | 98.0 |  |
| 50003968 | 94.58 | 94.69 | 53.0 | 95.0 | 192.307 | 99.0 |  |
