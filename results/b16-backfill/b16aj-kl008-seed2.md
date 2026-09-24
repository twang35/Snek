# b16aj-kl008-seed2

step **50,003,968** · 3052 evals · trailing **94.21** · peak **94.49** @33,079,296 · sef **91.0** · best30 **97.4** @24,641,536

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
| ppo_target_kl | 0.008 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b16aj-kl008-seed2](b16aj-kl008-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.83 | 0.83 | 0.0 | 5.0 | -0.076 | 0.0 |  |
| 32768 | 8.03 | 4.43 | 2.0 | 17.0 | 3.019 | 0.0 |  |
| 49152 | 10.31 | 6.39 | 2.0 | 27.0 | 5.304 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.39 | 94.21 | 69.0 | 95.0 | 189.114 | 96.0 |  |
| 49840128 | 94.88 | 94.22 | 83.0 | 95.0 | 192.612 | 99.0 |  |
| 49856512 | 93.56 | 94.26 | 18.0 | 95.0 | 187.291 | 95.0 |  |
| 49872896 | 94.94 | 94.3 | 90.0 | 95.0 | 191.616 | 98.0 |  |
| 49889280 | 94.53 | 94.21 | 64.0 | 95.0 | 189.162 | 96.0 |  |
| 49905664 | 94.36 | 94.25 | 80.0 | 95.0 | 186.03 | 93.0 |  |
| 49922048 | 94.56 | 94.3 | 80.0 | 95.0 | 189.267 | 96.0 |  |
| 49938432 | 93.72 | 94.25 | 62.0 | 95.0 | 185.395 | 93.0 |  |
| 49954816 | 93.26 | 94.25 | 12.0 | 95.0 | 188.89 | 97.0 |  |
| 49971200 | 94.23 | 94.21 | 22.0 | 95.0 | 190.935 | 98.0 |  |
| 49987584 | 94.5 | 94.24 | 82.0 | 95.0 | 188.163 | 95.0 |  |
| 50003968 | 93.66 | 94.21 | 18.0 | 95.0 | 186.379 | 94.0 |  |
