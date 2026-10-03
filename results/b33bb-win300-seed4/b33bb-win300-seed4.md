# b33bb-win300-seed4

step **50,003,968** · 1526 evals · trailing **94.68** · peak **94.84** @20,512,768 · sef **86.8** · best30 **99.7** @21,233,664

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

![b33bb-win300-seed4](b33bb-win300-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.12 | 4.12 | 0.0 | 13.0 | 1.499 | 0.0 |  |
| 65536 | 11.66 | 7.89 | 0.0 | 28.0 | 8.248 | 0.0 |  |
| 98304 | 24.89 | 13.56 | 1.0 | 48.0 | 19.879 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.79 | 94.6 | 74.0 | 95.0 | 192.55 | 99.0 |  |
| 49676288 | 94.6 | 94.65 | 55.0 | 95.0 | 192.365 | 99.0 |  |
| 49709056 | 94.64 | 94.64 | 59.0 | 95.0 | 192.398 | 99.0 |  |
| 49741824 | 94.67 | 94.63 | 62.0 | 95.0 | 192.435 | 99.0 |  |
| 49774592 | 95.0 | 94.63 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 49807360 | 94.23 | 94.66 | 56.0 | 95.0 | 190.951 | 98.0 |  |
| 49840128 | 95.0 | 94.67 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 49872896 | 95.0 | 94.67 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 49905664 | 94.54 | 94.67 | 49.0 | 95.0 | 192.254 | 99.0 |  |
| 49938432 | 95.0 | 94.67 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 49971200 | 95.0 | 94.67 | 95.0 | 95.0 | 193.757 | 100.0 |  |
| 50003968 | 95.0 | 94.68 | 95.0 | 95.0 | 193.759 | 100.0 |  |
