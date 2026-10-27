# b33an-win50-seed2

step **50,003,968** · 1526 evals · trailing **94.5** · peak **94.77** @19,857,408 · sef **92.7** · best30 **99.5** @48,791,552

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

![b33an-win50-seed2](b33an-win50-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 13.3 | 13.3 | 0.0 | 30.0 | 9.078 | 0.0 |  |
| 65536 | 36.14 | 27.28 | 4.0 | 80.0 | 31.206 | 0.0 |  |
| 98304 | 34.29 | 29.03 | 1.0 | 76.0 | 29.521 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 95.0 | 94.48 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 49676288 | 95.0 | 94.48 | 95.0 | 95.0 | 193.771 | 100.0 |  |
| 49709056 | 95.0 | 94.47 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 49741824 | 94.34 | 94.45 | 29.0 | 95.0 | 192.112 | 99.0 |  |
| 49774592 | 94.37 | 94.46 | 32.0 | 95.0 | 192.13 | 99.0 |  |
| 49807360 | 93.11 | 94.45 | 24.0 | 95.0 | 188.892 | 97.0 |  |
| 49840128 | 94.26 | 94.43 | 21.0 | 95.0 | 192.033 | 99.0 |  |
| 49872896 | 95.0 | 94.48 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 49905664 | 95.0 | 94.49 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 49938432 | 94.36 | 94.49 | 31.0 | 95.0 | 192.086 | 99.0 |  |
| 49971200 | 94.61 | 94.48 | 56.0 | 95.0 | 192.375 | 99.0 |  |
| 50003968 | 95.0 | 94.5 | 95.0 | 95.0 | 193.764 | 100.0 |  |
