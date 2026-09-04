# b11ab-lr4e5-seed2

step **33,079,296** · 2015 evals · trailing **92.21** · peak **94.12** @22,888,448 · sef **68.3** · best30 **96.4** @22,888,448

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
| ppo_learning_rate | 4e-05 |
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

![b11ab-lr4e5-seed2](b11ab-lr4e5-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.69 | 1.69 | 0.0 | 7.0 | 1.145 | 0.0 |  |
| 32768 | 1.49 | 1.59 | 0.0 | 6.0 | 0.99 | 0.0 |  |
| 49152 | 7.41 | 3.53 | 1.0 | 20.0 | 3.04 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 32833536 | 92.18 | 92.18 | 22.0 | 95.0 | 185.21 | 94.0 |  |
| 32866304 | 94.5 | 91.95 | 67.0 | 95.0 | 189.52 | 96.0 |  |
| 32882688 | 93.51 | 92.19 | 52.0 | 95.0 | 188.53 | 96.0 |  |
| 32899072 | 93.17 | 92.19 | 24.0 | 95.0 | 187.195 | 95.0 |  |
| 32915456 | 92.46 | 92.13 | 43.0 | 95.0 | 183.5 | 92.0 |  |
| 32931840 | 94.19 | 92.18 | 60.0 | 95.0 | 188.215 | 95.0 |  |
| 32948224 | 92.3 | 92.14 | 41.0 | 95.0 | 182.345 | 91.0 |  |
| 32964608 | 92.51 | 92.13 | 42.0 | 95.0 | 184.545 | 93.0 |  |
| 32980992 | 94.5 | 92.29 | 58.0 | 95.0 | 191.51 | 98.0 |  |
| 33013760 | 94.04 | 92.13 | 58.0 | 95.0 | 190.055 | 97.0 |  |
| 33062912 | 91.77 | 92.07 | 44.0 | 95.0 | 181.815 | 91.0 |  |
| 33079296 | 93.1 | 92.21 | 53.0 | 95.0 | 187.125 | 95.0 |  |
