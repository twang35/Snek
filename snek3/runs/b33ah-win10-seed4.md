# b33ah-win10-seed4

step **50,003,968** · 1526 evals · trailing **94.77** · peak **94.82** @49,643,520 · sef **89.9** · best30 **99.6** @47,775,744

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

![b33ah-win10-seed4](b33ah-win10-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.11 | 4.11 | 0.0 | 9.0 | 1.576 | 0.0 |  |
| 65536 | 11.2 | 7.65 | 0.0 | 24.0 | 7.702 | 0.0 |  |
| 98304 | 23.2 | 12.84 | 0.0 | 42.0 | 18.332 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.76 | 94.82 | 71.0 | 95.0 | 192.479 | 99.0 |  |
| 49676288 | 95.0 | 94.7 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 49709056 | 95.0 | 94.75 | 95.0 | 95.0 | 193.755 | 100.0 |  |
| 49741824 | 94.56 | 94.8 | 51.0 | 95.0 | 192.32 | 99.0 |  |
| 49774592 | 94.43 | 94.82 | 38.0 | 95.0 | 192.152 | 99.0 |  |
| 49807360 | 94.66 | 94.82 | 61.0 | 95.0 | 192.419 | 99.0 |  |
| 49840128 | 95.0 | 94.82 | 95.0 | 95.0 | 193.754 | 100.0 |  |
| 49872896 | 94.36 | 94.8 | 31.0 | 95.0 | 192.115 | 99.0 |  |
| 49905664 | 94.5 | 94.79 | 45.0 | 95.0 | 192.209 | 99.0 |  |
| 49938432 | 93.9 | 94.78 | 16.0 | 95.0 | 190.666 | 98.0 |  |
| 49971200 | 93.42 | 94.72 | 37.0 | 95.0 | 188.141 | 96.0 |  |
| 50003968 | 94.25 | 94.77 | 20.0 | 95.0 | 192.008 | 99.0 |  |
