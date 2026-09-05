# b17bb-clip01anneal-seed4

step **7,274,496** · 440 evals · trailing **90.11** · peak **93.81** @3,260,416 · sef **41.8** · best30 **90.1** @6,356,992

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 50003968 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.1 |
| ppo_clip_final | 0.02 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b17bb-clip01anneal-seed4](b17bb-clip01anneal-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.04 | 0.04 | 0.0 | 2.0 | -3.049 | 0.0 |  |
| 32768 | 6.24 | 3.14 | 0.0 | 15.0 | 3.242 | 0.0 |  |
| 49152 | 18.7 | 8.33 | 1.0 | 39.0 | 14.19 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 7061504 | 80.07 | 90.26 | 22.0 | 95.0 | 152.842 | 74.0 |  |
| 7077888 | 90.95 | 90.25 | 27.0 | 95.0 | 178.643 | 89.0 |  |
| 7094272 | 92.98 | 90.68 | 28.0 | 95.0 | 184.671 | 93.0 |  |
| 7110656 | 89.39 | 89.69 | 18.0 | 95.0 | 175.089 | 87.0 |  |
| 7127040 | 90.4 | 90.1 | 28.0 | 95.0 | 179.094 | 90.0 |  |
| 7143424 | 91.23 | 89.92 | 8.0 | 95.0 | 183.895 | 94.0 |  |
| 7159808 | 90.13 | 90.49 | 28.0 | 95.0 | 179.818 | 91.0 |  |
| 7176192 | 84.2 | 90.65 | 23.0 | 95.0 | 164.953 | 82.0 |  |
| 7192576 | 89.47 | 90.05 | 27.0 | 95.0 | 176.179 | 88.0 |  |
| 7208960 | 81.58 | 89.9 | 18.0 | 95.0 | 156.343 | 76.0 |  |
| 7225344 | 77.28 | 89.48 | 23.0 | 95.0 | 147.082 | 71.0 |  |
| 7274496 | 70.29 | 90.11 | 20.0 | 95.0 | 131.166 | 62.0 |  |
