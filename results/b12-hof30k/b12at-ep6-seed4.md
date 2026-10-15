# b12at-ep6-seed4

step **50,003,968** · 3052 evals · trailing **94.19** · peak **94.63** @34,701,312 · sef **91.6** · best30 **98.5** @35,717,120

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
| ppo_epochs | 6 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.0003 |
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

![b12at-ep6-seed4](b12at-ep6-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.11 | 1.11 | 0.0 | 4.0 | -2.63 | 0.0 |  |
| 32768 | 31.97 | 23.34 | 6.0 | 58.0 | 26.97 | 0.0 |  |
| 49152 | 30.6 | 15.86 | 12.0 | 53.0 | 25.6 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.61 | 94.18 | 72.0 | 95.0 | 190.625 | 97.0 |  |
| 49840128 | 94.02 | 94.19 | 69.0 | 95.0 | 187.005 | 94.0 |  |
| 49856512 | 93.66 | 94.23 | 49.0 | 95.0 | 185.65 | 93.0 |  |
| 49872896 | 94.69 | 94.21 | 80.0 | 95.0 | 190.705 | 97.0 |  |
| 49889280 | 93.79 | 94.16 | 15.0 | 95.0 | 188.765 | 96.0 |  |
| 49905664 | 94.52 | 94.18 | 72.0 | 95.0 | 189.495 | 96.0 |  |
| 49922048 | 93.64 | 94.19 | 19.0 | 95.0 | 188.57 | 96.0 |  |
| 49938432 | 94.44 | 94.18 | 49.0 | 95.0 | 190.41 | 97.0 |  |
| 49954816 | 94.1 | 94.17 | 57.0 | 95.0 | 189.075 | 96.0 |  |
| 49971200 | 94.13 | 94.15 | 8.0 | 95.0 | 192.135 | 99.0 |  |
| 49987584 | 94.77 | 94.15 | 82.0 | 95.0 | 191.78 | 98.0 |  |
| 50003968 | 93.28 | 94.19 | 11.0 | 95.0 | 189.205 | 97.0 |  |
