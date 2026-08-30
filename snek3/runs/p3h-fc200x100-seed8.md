# p3h-fc200x100-seed8

step **57,294,848** · 3490 evals · trailing **94.09** · peak **94.4** @27,803,648 · sef **93.4** · best30 **97.6** @30,638,080

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 6 |
| fc_layers | (200, 100) |
| graph_eval_episodes | 100 |
| max_steps | 400000000 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 8 |
| torch_threads | 1 |

![p3h-fc200x100-seed8](p3h-fc200x100-seed8.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 16.12 | 16.12 | 0.0 | 39.0 | 12.695 | 0.0 |  |
| 32768 | 38.11 | 29.83 | 9.0 | 69.0 | 33.605 | 0.0 |  |
| 49152 | 34.56 | 25.34 | 8.0 | 65.0 | 29.56 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 56999936 | 94.88 | 93.71 | 86.0 | 95.0 | 191.89 | 98.0 |  |
| 57016320 | 94.08 | 93.7 | 18.0 | 95.0 | 191.09 | 98.0 |  |
| 57032704 | 94.4 | 94.0 | 66.0 | 95.0 | 188.38 | 95.0 |  |
| 57049088 | 93.28 | 93.94 | 10.0 | 95.0 | 187.305 | 95.0 |  |
| 57065472 | 94.51 | 93.82 | 67.0 | 95.0 | 189.53 | 96.0 |  |
| 57081856 | 93.65 | 93.98 | 26.0 | 95.0 | 189.665 | 97.0 |  |
| 57098240 | 94.63 | 94.14 | 81.0 | 95.0 | 189.65 | 96.0 |  |
| 57131008 | 94.28 | 93.7 | 70.0 | 95.0 | 189.3 | 96.0 |  |
| 57147392 | 94.74 | 93.76 | 80.0 | 95.0 | 190.71 | 97.0 |  |
| 57163776 | 94.78 | 93.93 | 80.0 | 95.0 | 191.745 | 98.0 |  |
| 57180160 | 94.84 | 94.06 | 79.0 | 95.0 | 192.845 | 99.0 |  |
| 57294848 | 94.88 | 94.09 | 83.0 | 95.0 | 192.885 | 99.0 |  |
