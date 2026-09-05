# b14i-roll192-seed1

step **41,238,528** · 1678 evals · trailing **94.19** · peak **94.67** @27,598,848 · sef **90.0** · best30 **98.3** @27,795,456

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
| seed | 1 |
| torch_threads | 1 |

![b14i-roll192-seed1](b14i-roll192-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 24576 | 23.31 | 23.31 | 6.0 | 42.0 | 18.31 | 0.0 |  |
| 49152 | 40.96 | 32.6 | 2.0 | 86.0 | 36.095 | 0.0 |  |
| 73728 | 32.77 | 28.04 | 8.0 | 63.0 | 27.77 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 40968192 | 93.95 | 94.24 | 24.0 | 95.0 | 190.96 | 98.0 |  |
| 40992768 | 94.86 | 94.28 | 81.0 | 95.0 | 192.865 | 99.0 |  |
| 41017344 | 94.18 | 94.17 | 68.0 | 95.0 | 189.2 | 96.0 |  |
| 41041920 | 91.33 | 94.17 | 16.0 | 95.0 | 183.275 | 93.0 |  |
| 41066496 | 94.34 | 94.17 | 32.0 | 95.0 | 191.26 | 98.0 |  |
| 41091072 | 94.68 | 94.19 | 73.0 | 95.0 | 191.69 | 98.0 |  |
| 41115648 | 93.75 | 94.16 | 26.0 | 95.0 | 187.73 | 95.0 |  |
| 41140224 | 94.63 | 94.18 | 58.0 | 95.0 | 192.635 | 99.0 |  |
| 41164800 | 94.52 | 94.21 | 47.0 | 95.0 | 192.48 | 99.0 |  |
| 41189376 | 94.34 | 94.18 | 68.0 | 95.0 | 190.355 | 97.0 |  |
| 41213952 | 93.67 | 94.17 | 40.0 | 95.0 | 188.69 | 96.0 |  |
| 41238528 | 94.02 | 94.19 | 22.0 | 95.0 | 191.03 | 98.0 |  |
