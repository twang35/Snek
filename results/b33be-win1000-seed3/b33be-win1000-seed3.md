# b33be-win1000-seed3

step **50,003,968** · 1526 evals · trailing **94.54** · peak **94.9** @38,830,080 · sef **80.5** · best30 **99.7** @19,529,728

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

![b33be-win1000-seed3](b33be-win1000-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 2.47 | 2.47 | 0.0 | 8.0 | 0.564 | 0.0 |  |
| 65536 | 18.64 | 15.78 | 0.0 | 49.0 | 14.588 | 0.0 |  |
| 98304 | 27.82 | 18.79 | 2.0 | 52.0 | 22.975 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.23 | 94.57 | 18.0 | 95.0 | 191.999 | 99.0 |  |
| 49676288 | 94.6 | 94.58 | 55.0 | 95.0 | 192.327 | 99.0 |  |
| 49709056 | 94.63 | 94.6 | 58.0 | 95.0 | 192.401 | 99.0 |  |
| 49741824 | 94.5 | 94.59 | 45.0 | 95.0 | 192.259 | 99.0 |  |
| 49774592 | 94.04 | 94.57 | 47.0 | 95.0 | 189.775 | 97.0 |  |
| 49807360 | 94.46 | 94.52 | 41.0 | 95.0 | 192.185 | 99.0 |  |
| 49840128 | 95.0 | 94.57 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 49872896 | 94.18 | 94.54 | 53.0 | 95.0 | 190.907 | 98.0 |  |
| 49905664 | 93.96 | 94.49 | 26.0 | 95.0 | 190.69 | 98.0 |  |
| 49938432 | 94.43 | 94.51 | 38.0 | 95.0 | 192.158 | 99.0 |  |
| 49971200 | 92.85 | 94.48 | 33.0 | 95.0 | 187.498 | 96.0 |  |
| 50003968 | 95.0 | 94.54 | 95.0 | 95.0 | 193.767 | 100.0 |  |
