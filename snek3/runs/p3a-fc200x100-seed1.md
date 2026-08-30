# p3a-fc200x100-seed1

step **61,358,080** · 3738 evals · trailing **94.53** · peak **94.65** @11,452,416 · sef **95.2** · best30 **97.8** @49,201,152

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
| seed | 1 |
| torch_threads | 1 |

![p3a-fc200x100-seed1](p3a-fc200x100-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 10.61 | 10.61 | 0.0 | 22.0 | 6.465 | 0.0 |  |
| 32768 | 32.92 | 26.4 | 8.0 | 58.0 | 28.64 | 0.0 |  |
| 49152 | 35.67 | 23.14 | 8.0 | 64.0 | 30.715 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 61063168 | 94.62 | 94.53 | 76.0 | 95.0 | 190.59 | 97.0 |  |
| 61079552 | 94.67 | 94.53 | 82.0 | 95.0 | 190.685 | 97.0 |  |
| 61095936 | 94.59 | 94.54 | 61.0 | 95.0 | 191.6 | 98.0 |  |
| 61112320 | 95.0 | 94.48 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 61128704 | 94.53 | 94.52 | 70.0 | 95.0 | 190.545 | 97.0 |  |
| 61145088 | 94.48 | 94.54 | 67.0 | 95.0 | 190.495 | 97.0 |  |
| 61161472 | 94.69 | 94.54 | 82.0 | 95.0 | 190.705 | 97.0 |  |
| 61194240 | 94.76 | 94.53 | 83.0 | 95.0 | 190.775 | 97.0 |  |
| 61210624 | 94.53 | 94.54 | 76.0 | 95.0 | 188.51 | 95.0 |  |
| 61227008 | 94.92 | 94.54 | 87.0 | 95.0 | 192.88 | 99.0 |  |
| 61341696 | 94.82 | 94.54 | 87.0 | 95.0 | 190.79 | 97.0 |  |
| 61358080 | 94.19 | 94.53 | 68.0 | 95.0 | 188.17 | 95.0 |  |
