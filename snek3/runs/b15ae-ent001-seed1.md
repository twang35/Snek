# b15ae-ent001-seed1

step **42,532,864** · 2588 evals · trailing **93.87** · peak **94.38** @22,052,864 · sef **94.6** · best30 **97.9** @22,151,168

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
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.001 |
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
| seed | 1 |
| torch_threads | 1 |

![b15ae-ent001-seed1](b15ae-ent001-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 17.39 | 17.39 | 1.0 | 33.0 | 15.675 | 0.0 |  |
| 32768 | 50.7 | 36.78 | 3.0 | 87.0 | 46.06 | 0.0 |  |
| 49152 | 36.48 | 28.94 | 3.0 | 83.0 | 31.66 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 42221568 | 94.6 | 93.61 | 68.0 | 95.0 | 191.61 | 98.0 |  |
| 42237952 | 94.2 | 93.75 | 59.0 | 95.0 | 189.22 | 96.0 |  |
| 42254336 | 93.54 | 93.84 | 10.0 | 95.0 | 188.515 | 96.0 |  |
| 42369024 | 93.56 | 93.76 | 28.0 | 95.0 | 185.595 | 93.0 |  |
| 42385408 | 94.34 | 93.73 | 72.0 | 95.0 | 188.32 | 95.0 |  |
| 42401792 | 93.05 | 93.7 | 8.0 | 95.0 | 188.025 | 96.0 |  |
| 42418176 | 93.9 | 93.76 | 23.0 | 95.0 | 188.875 | 96.0 |  |
| 42467328 | 94.2 | 93.68 | 59.0 | 95.0 | 186.19 | 93.0 |  |
| 42483712 | 94.6 | 93.77 | 69.0 | 95.0 | 191.61 | 98.0 |  |
| 42500096 | 94.63 | 93.81 | 65.0 | 95.0 | 190.645 | 97.0 |  |
| 42516480 | 94.09 | 93.81 | 39.0 | 95.0 | 190.06 | 97.0 |  |
| 42532864 | 94.33 | 93.87 | 33.0 | 95.0 | 191.295 | 98.0 |  |
