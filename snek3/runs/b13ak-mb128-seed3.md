# b13ak-mb128-seed3

step **50,003,968** · 3052 evals · trailing **94.09** · peak **94.6** @19,021,824 · sef **91.8** · best30 **98.3** @19,087,360

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
| seed | 3 |
| torch_threads | 1 |

![b13ak-mb128-seed3](b13ak-mb128-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.09 | 0.09 | 0.0 | 2.0 | -3.29 | 0.0 |  |
| 32768 | 5.24 | 2.67 | 0.0 | 20.0 | 3.795 | 0.0 |  |
| 49152 | 20.98 | 8.77 | 0.0 | 42.0 | 16.16 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.27 | 94.09 | 4.0 | 95.0 | 186.3 | 94.0 |  |
| 49840128 | 93.88 | 94.09 | 18.0 | 95.0 | 189.895 | 97.0 |  |
| 49856512 | 94.17 | 94.09 | 58.0 | 95.0 | 190.14 | 97.0 |  |
| 49872896 | 94.57 | 94.11 | 52.0 | 95.0 | 192.575 | 99.0 |  |
| 49889280 | 94.41 | 94.17 | 62.0 | 95.0 | 189.43 | 96.0 |  |
| 49905664 | 93.91 | 94.13 | 53.0 | 95.0 | 189.88 | 97.0 |  |
| 49922048 | 94.19 | 94.11 | 22.0 | 95.0 | 191.155 | 98.0 |  |
| 49938432 | 95.0 | 94.14 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49954816 | 94.94 | 94.16 | 89.0 | 95.0 | 192.945 | 99.0 |  |
| 49971200 | 92.52 | 94.12 | 25.0 | 95.0 | 184.465 | 93.0 |  |
| 49987584 | 94.02 | 94.09 | 31.0 | 95.0 | 189.99 | 97.0 |  |
| 50003968 | 94.62 | 94.09 | 57.0 | 95.0 | 192.625 | 99.0 |  |
