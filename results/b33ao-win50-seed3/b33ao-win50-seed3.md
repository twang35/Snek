# b33ao-win50-seed3

step **50,003,968** · 1526 evals · trailing **94.57** · peak **94.85** @29,687,808 · sef **92.5** · best30 **99.7** @29,884,416

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

![b33ao-win50-seed3](b33ao-win50-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 2.67 | 2.67 | 0.0 | 12.0 | 0.542 | 0.0 |  |
| 65536 | 16.9 | 9.79 | 0.0 | 43.0 | 13.257 | 0.0 |  |
| 98304 | 25.8 | 15.12 | 1.0 | 49.0 | 21.096 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 93.55 | 94.58 | 22.0 | 95.0 | 190.276 | 98.0 |  |
| 49676288 | 94.25 | 94.63 | 26.0 | 95.0 | 190.926 | 98.0 |  |
| 49709056 | 93.66 | 94.59 | 19.0 | 95.0 | 190.347 | 98.0 |  |
| 49741824 | 94.49 | 94.63 | 44.0 | 95.0 | 192.205 | 99.0 |  |
| 49774592 | 94.65 | 94.63 | 60.0 | 95.0 | 192.363 | 99.0 |  |
| 49807360 | 93.56 | 94.63 | 20.0 | 95.0 | 190.241 | 98.0 |  |
| 49840128 | 95.0 | 94.58 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 49872896 | 94.31 | 94.58 | 26.0 | 95.0 | 192.019 | 99.0 |  |
| 49905664 | 95.0 | 94.58 | 95.0 | 95.0 | 193.756 | 100.0 |  |
| 49938432 | 95.0 | 94.58 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49971200 | 95.0 | 94.6 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 50003968 | 94.34 | 94.57 | 29.0 | 95.0 | 192.058 | 99.0 |  |
