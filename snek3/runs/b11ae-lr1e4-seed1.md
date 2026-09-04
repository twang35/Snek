# b11ae-lr1e4-seed1

step **32,751,616** · 1993 evals · trailing **94.12** · peak **94.32** @30,654,464 · sef **72.3** · best30 **97.8** @30,982,144

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
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.0001 |
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

![b11ae-lr1e4-seed1](b11ae-lr1e4-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 10.29 | 16.62 | 1.0 | 32.0 | 9.16 | 0.0 |  |
| 32768 | 16.92 | 16.92 | 2.0 | 37.0 | 11.92 | 0.0 |  |
| 49152 | 16.82 | 17.11 | 2.0 | 41.0 | 11.82 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 32473088 | 94.19 | 94.16 | 64.0 | 95.0 | 190.205 | 97.0 |  |
| 32489472 | 93.68 | 94.13 | 53.0 | 95.0 | 188.7 | 96.0 |  |
| 32505856 | 93.32 | 94.12 | 49.0 | 95.0 | 188.34 | 96.0 |  |
| 32522240 | 93.76 | 94.12 | 16.0 | 95.0 | 188.78 | 96.0 |  |
| 32538624 | 93.41 | 94.19 | 53.0 | 95.0 | 187.435 | 95.0 |  |
| 32555008 | 95.0 | 94.17 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 32587776 | 95.0 | 94.14 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 32604160 | 94.2 | 94.15 | 58.0 | 95.0 | 190.215 | 97.0 |  |
| 32620544 | 94.32 | 94.13 | 56.0 | 95.0 | 190.335 | 97.0 |  |
| 32636928 | 93.14 | 94.08 | 53.0 | 95.0 | 187.165 | 95.0 |  |
| 32718848 | 94.36 | 94.07 | 57.0 | 95.0 | 191.37 | 98.0 |  |
| 32751616 | 95.0 | 94.12 | 95.0 | 95.0 | 194.0 | 100.0 |  |
