# b33as-win100-seed3

step **50,003,968** · 1526 evals · trailing **94.48** · peak **94.86** @32,112,640 · sef **91.7** · best30 **99.6** @31,883,264

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

![b33as-win100-seed3](b33as-win100-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 2.66 | 2.66 | 0.0 | 11.0 | 0.577 | 0.0 |  |
| 65536 | 14.87 | 8.77 | 0.0 | 46.0 | 11.239 | 0.0 |  |
| 98304 | 27.33 | 14.95 | 1.0 | 54.0 | 22.487 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 95.0 | 94.53 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 49676288 | 94.43 | 94.51 | 38.0 | 95.0 | 192.159 | 99.0 |  |
| 49709056 | 94.25 | 94.51 | 20.0 | 95.0 | 191.969 | 99.0 |  |
| 49741824 | 95.0 | 94.56 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 49774592 | 93.74 | 94.49 | 15.0 | 95.0 | 189.466 | 97.0 |  |
| 49807360 | 93.7 | 94.51 | 10.0 | 95.0 | 190.433 | 98.0 |  |
| 49840128 | 94.37 | 94.51 | 32.0 | 95.0 | 192.088 | 99.0 |  |
| 49872896 | 95.0 | 94.51 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 49905664 | 94.27 | 94.48 | 22.0 | 95.0 | 192.036 | 99.0 |  |
| 49938432 | 95.0 | 94.51 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 49971200 | 94.5 | 94.53 | 45.0 | 95.0 | 192.262 | 99.0 |  |
| 50003968 | 94.61 | 94.48 | 56.0 | 95.0 | 192.333 | 99.0 |  |
