# b13aj-mb128-seed2

step **50,003,968** · 3052 evals · trailing **92.7** · peak **94.51** @16,662,528 · sef **93.0** · best30 **98.1** @16,859,136

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
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 128 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b13aj-mb128-seed2](b13aj-mb128-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.61 | 1.61 | 0.0 | 5.0 | -1.41 | 0.0 |  |
| 32768 | 12.59 | 7.1 | 3.0 | 26.0 | 7.68 | 0.0 |  |
| 49152 | 21.63 | 11.94 | 7.0 | 54.0 | 16.765 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 90.71 | 93.3 | 11.0 | 95.0 | 183.47 | 94.0 |  |
| 49840128 | 93.26 | 93.34 | 18.0 | 95.0 | 189.14 | 97.0 |  |
| 49856512 | 93.53 | 92.86 | 18.0 | 95.0 | 188.46 | 96.0 |  |
| 49872896 | 91.7 | 92.97 | 8.0 | 95.0 | 183.6 | 93.0 |  |
| 49889280 | 93.06 | 93.08 | 18.0 | 95.0 | 187.945 | 96.0 |  |
| 49905664 | 94.07 | 92.74 | 56.0 | 95.0 | 189.045 | 96.0 |  |
| 49922048 | 92.57 | 92.67 | 18.0 | 95.0 | 186.46 | 95.0 |  |
| 49938432 | 94.81 | 92.72 | 78.0 | 95.0 | 191.775 | 98.0 |  |
| 49954816 | 94.5 | 92.73 | 53.0 | 95.0 | 191.51 | 98.0 |  |
| 49971200 | 94.5 | 92.69 | 73.0 | 95.0 | 190.515 | 97.0 |  |
| 49987584 | 93.05 | 92.84 | 16.0 | 95.0 | 187.98 | 96.0 |  |
| 50003968 | 94.53 | 92.7 | 60.0 | 95.0 | 191.54 | 98.0 |  |
