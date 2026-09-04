# b11af-lr1e4-seed2

step **32,784,384** · 1993 evals · trailing **93.88** · peak **94.37** @30,638,080 · sef **80.5** · best30 **97.8** @28,180,480

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
| seed | 2 |
| torch_threads | 1 |

![b11af-lr1e4-seed2](b11af-lr1e4-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.08 | 1.08 | 0.0 | 6.0 | -0.32 | 0.0 |  |
| 32768 | 8.46 | 4.77 | 2.0 | 16.0 | 3.46 | 0.0 |  |
| 49152 | 7.98 | 6.39 | 2.0 | 21.0 | 2.98 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 32473088 | 93.53 | 93.71 | 36.0 | 95.0 | 187.555 | 95.0 |  |
| 32489472 | 94.17 | 93.66 | 12.0 | 95.0 | 192.175 | 99.0 |  |
| 32505856 | 94.25 | 93.73 | 54.0 | 95.0 | 190.265 | 97.0 |  |
| 32522240 | 94.03 | 93.93 | 63.0 | 95.0 | 189.05 | 96.0 |  |
| 32538624 | 93.55 | 93.91 | 51.0 | 95.0 | 187.575 | 95.0 |  |
| 32555008 | 94.37 | 93.9 | 56.0 | 95.0 | 190.385 | 97.0 |  |
| 32636928 | 93.69 | 93.75 | 56.0 | 95.0 | 188.71 | 96.0 |  |
| 32653312 | 94.19 | 93.75 | 53.0 | 95.0 | 191.2 | 98.0 |  |
| 32669696 | 94.33 | 93.84 | 53.0 | 95.0 | 190.345 | 97.0 |  |
| 32686080 | 94.42 | 93.82 | 55.0 | 95.0 | 191.43 | 98.0 |  |
| 32735232 | 93.67 | 93.85 | 56.0 | 95.0 | 188.69 | 96.0 |  |
| 32784384 | 95.0 | 93.88 | 95.0 | 95.0 | 194.0 | 100.0 |  |
