# b33ag-win10-seed3

step **50,003,968** · 1526 evals · trailing **94.33** · peak **94.92** @44,498,944 · sef **90.8** · best30 **99.8** @44,367,872

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

![b33ag-win10-seed3](b33ag-win10-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 2.76 | 2.76 | 0.0 | 11.0 | 0.631 | 0.0 |  |
| 65536 | 18.06 | 15.92 | 0.0 | 42.0 | 13.925 | 0.0 |  |
| 98304 | 26.93 | 14.84 | 1.0 | 45.0 | 22.138 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 92.92 | 94.36 | 3.0 | 95.0 | 188.641 | 97.0 |  |
| 49676288 | 95.0 | 94.38 | 95.0 | 95.0 | 193.755 | 100.0 |  |
| 49709056 | 94.62 | 94.39 | 57.0 | 95.0 | 192.333 | 99.0 |  |
| 49741824 | 94.65 | 94.42 | 60.0 | 95.0 | 192.409 | 99.0 |  |
| 49774592 | 94.46 | 94.42 | 41.0 | 95.0 | 192.162 | 99.0 |  |
| 49807360 | 95.0 | 94.42 | 95.0 | 95.0 | 193.742 | 100.0 |  |
| 49840128 | 93.62 | 94.37 | 3.0 | 95.0 | 190.334 | 98.0 |  |
| 49872896 | 95.0 | 94.39 | 95.0 | 95.0 | 193.75 | 100.0 |  |
| 49905664 | 94.01 | 94.39 | 28.0 | 95.0 | 190.729 | 98.0 |  |
| 49938432 | 94.13 | 94.41 | 8.0 | 95.0 | 191.884 | 99.0 |  |
| 49971200 | 93.41 | 94.32 | 47.0 | 95.0 | 188.084 | 96.0 |  |
| 50003968 | 93.87 | 94.33 | 3.0 | 95.0 | 190.59 | 98.0 |  |
