# b33al-win30-seed4

step **50,003,968** · 1526 evals · trailing **94.77** · peak **94.79** @49,741,824 · sef **88.9** · best30 **99.5** @48,988,160

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

![b33al-win30-seed4](b33al-win30-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.75 | 4.75 | 0.0 | 12.0 | 1.68 | 0.0 |  |
| 65536 | 11.41 | 8.08 | 0.0 | 27.0 | 7.998 | 0.0 |  |
| 98304 | 23.78 | 13.31 | 0.0 | 43.0 | 18.951 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 95.0 | 94.72 | 95.0 | 95.0 | 193.757 | 100.0 |  |
| 49676288 | 95.0 | 94.7 | 95.0 | 95.0 | 193.755 | 100.0 |  |
| 49709056 | 94.57 | 94.75 | 52.0 | 95.0 | 192.335 | 99.0 |  |
| 49741824 | 95.0 | 94.79 | 95.0 | 95.0 | 193.746 | 100.0 |  |
| 49774592 | 95.0 | 94.79 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 49807360 | 94.01 | 94.75 | 38.0 | 95.0 | 190.733 | 98.0 |  |
| 49840128 | 95.0 | 94.75 | 95.0 | 95.0 | 193.757 | 100.0 |  |
| 49872896 | 95.0 | 94.76 | 95.0 | 95.0 | 193.76 | 100.0 |  |
| 49905664 | 94.51 | 94.76 | 46.0 | 95.0 | 192.277 | 99.0 |  |
| 49938432 | 95.0 | 94.76 | 95.0 | 95.0 | 193.757 | 100.0 |  |
| 49971200 | 95.0 | 94.76 | 95.0 | 95.0 | 193.752 | 100.0 |  |
| 50003968 | 94.36 | 94.77 | 31.0 | 95.0 | 192.073 | 99.0 |  |
