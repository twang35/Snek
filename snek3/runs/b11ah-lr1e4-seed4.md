# b11ah-lr1e4-seed4

step **32,030,720** · 1952 evals · trailing **94.09** · peak **94.39** @25,559,040 · sef **74.1** · best30 **97.9** @25,444,352

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
| seed | 4 |
| torch_threads | 1 |

![b11ah-lr1e4-seed4](b11ah-lr1e4-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.27 | 0.27 | 0.0 | 2.0 | -1.13 | 0.0 |  |
| 32768 | 6.01 | 3.14 | 1.0 | 12.0 | 1.505 | 0.0 |  |
| 49152 | 15.25 | 7.18 | 3.0 | 29.0 | 10.25 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 31801344 | 93.14 | 94.0 | 55.0 | 95.0 | 186.17 | 94.0 |  |
| 31817728 | 93.82 | 94.1 | 55.0 | 95.0 | 188.84 | 96.0 |  |
| 31834112 | 94.07 | 94.1 | 60.0 | 95.0 | 190.085 | 97.0 |  |
| 31850496 | 94.25 | 94.05 | 58.0 | 95.0 | 190.22 | 97.0 |  |
| 31866880 | 94.01 | 94.05 | 51.0 | 95.0 | 189.03 | 96.0 |  |
| 31883264 | 94.71 | 94.08 | 66.0 | 95.0 | 192.715 | 99.0 |  |
| 31948800 | 93.04 | 94.03 | 36.0 | 95.0 | 184.035 | 92.0 |  |
| 31965184 | 93.17 | 94.07 | 54.0 | 95.0 | 187.195 | 95.0 |  |
| 31981568 | 93.19 | 94.1 | 55.0 | 95.0 | 185.18 | 93.0 |  |
| 31997952 | 94.95 | 94.1 | 92.0 | 95.0 | 191.96 | 98.0 |  |
| 32014336 | 93.93 | 94.08 | 57.0 | 95.0 | 188.95 | 96.0 |  |
| 32030720 | 94.26 | 94.09 | 60.0 | 95.0 | 190.275 | 97.0 |  |
