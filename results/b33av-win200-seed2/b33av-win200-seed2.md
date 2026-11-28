# b33av-win200-seed2

step **50,003,968** · 1526 evals · trailing **94.75** · peak **94.92** @44,859,392 · sef **91.5** · best30 **99.8** @44,990,464

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

![b33av-win200-seed2](b33av-win200-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 11.65 | 11.65 | 0.0 | 27.0 | 7.211 | 0.0 |  |
| 65536 | 36.21 | 23.93 | 10.0 | 59.0 | 31.28 | 0.0 |  |
| 98304 | 34.89 | 27.58 | 1.0 | 64.0 | 29.989 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.25 | 94.75 | 20.0 | 95.0 | 192.022 | 99.0 |  |
| 49676288 | 94.41 | 94.74 | 36.0 | 95.0 | 192.173 | 99.0 |  |
| 49709056 | 94.61 | 94.75 | 56.0 | 95.0 | 192.375 | 99.0 |  |
| 49741824 | 95.0 | 94.76 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 49774592 | 95.0 | 94.76 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 49807360 | 95.0 | 94.76 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 49840128 | 95.0 | 94.76 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 49872896 | 94.77 | 94.75 | 72.0 | 95.0 | 192.492 | 99.0 |  |
| 49905664 | 95.0 | 94.75 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 49938432 | 94.55 | 94.74 | 50.0 | 95.0 | 192.315 | 99.0 |  |
| 49971200 | 95.0 | 94.75 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 50003968 | 95.0 | 94.75 | 95.0 | 95.0 | 193.765 | 100.0 |  |
