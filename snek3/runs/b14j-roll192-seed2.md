# b14j-roll192-seed2

step **33,005,568** · 1337 evals · trailing **94.27** · peak **94.35** @18,014,208 · sef **82.7** · best30 **97.8** @18,186,240

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 24576 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 50003968 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 192 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 24576 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b14j-roll192-seed2](b14j-roll192-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 24576 | 2.39 | 2.39 | 1.0 | 7.0 | -0.63 | 0.0 |  |
| 49152 | 13.56 | 7.98 | 4.0 | 30.0 | 8.605 | 0.0 |  |
| 73728 | 24.84 | 13.6 | 2.0 | 55.0 | 19.885 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 32587776 | 93.34 | 94.23 | 8.0 | 95.0 | 189.355 | 97.0 |  |
| 32612352 | 94.74 | 94.3 | 69.0 | 95.0 | 192.745 | 99.0 |  |
| 32636928 | 94.94 | 94.19 | 92.0 | 95.0 | 191.95 | 98.0 |  |
| 32661504 | 93.26 | 94.23 | 16.0 | 95.0 | 185.25 | 93.0 |  |
| 32686080 | 95.0 | 94.11 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 32710656 | 94.52 | 94.26 | 70.0 | 95.0 | 191.53 | 98.0 |  |
| 32735232 | 94.06 | 94.28 | 60.0 | 95.0 | 189.08 | 96.0 |  |
| 32759808 | 94.81 | 94.31 | 76.0 | 95.0 | 192.815 | 99.0 |  |
| 32833536 | 93.71 | 94.28 | 17.0 | 95.0 | 187.69 | 95.0 |  |
| 32858112 | 94.67 | 94.27 | 66.0 | 95.0 | 191.68 | 98.0 |  |
| 32980992 | 93.52 | 94.23 | 61.0 | 95.0 | 186.55 | 94.0 |  |
| 33005568 | 93.75 | 94.27 | 26.0 | 95.0 | 189.765 | 97.0 |  |
