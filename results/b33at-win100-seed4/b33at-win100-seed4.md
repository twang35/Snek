# b33at-win100-seed4

step **50,003,968** · 1526 evals · trailing **94.66** · peak **94.81** @43,122,688 · sef **91.1** · best30 **99.6** @43,122,688

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

![b33at-win100-seed4](b33at-win100-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.64 | 4.64 | 0.0 | 11.0 | 1.568 | 0.0 |  |
| 65536 | 10.59 | 7.62 | 0.0 | 30.0 | 7.538 | 0.0 |  |
| 98304 | 25.33 | 13.52 | 1.0 | 48.0 | 20.409 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 93.64 | 94.64 | 6.0 | 95.0 | 190.421 | 98.0 |  |
| 49676288 | 95.0 | 94.63 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 49709056 | 94.58 | 94.62 | 53.0 | 95.0 | 192.307 | 99.0 |  |
| 49741824 | 95.0 | 94.62 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 49774592 | 93.57 | 94.63 | 20.0 | 95.0 | 190.293 | 98.0 |  |
| 49807360 | 94.64 | 94.62 | 59.0 | 95.0 | 192.402 | 99.0 |  |
| 49840128 | 94.65 | 94.62 | 60.0 | 95.0 | 192.42 | 99.0 |  |
| 49872896 | 93.75 | 94.59 | 14.0 | 95.0 | 190.475 | 98.0 |  |
| 49905664 | 95.0 | 94.65 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 49938432 | 95.0 | 94.66 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 49971200 | 94.8 | 94.66 | 75.0 | 95.0 | 192.564 | 99.0 |  |
| 50003968 | 95.0 | 94.66 | 95.0 | 95.0 | 193.765 | 100.0 |  |
