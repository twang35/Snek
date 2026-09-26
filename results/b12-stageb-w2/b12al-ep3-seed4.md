# b12al-ep3-seed4

step **50,003,968** · 3052 evals · trailing **94.05** · peak **94.59** @48,168,960 · sef **89.0** · best30 **98.3** @48,431,104

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
| ppo_epochs | 3 |
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

![b12al-ep3-seed4](b12al-ep3-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.6 | 0.6 | 0.0 | 3.0 | -0.035 | 0.0 |  |
| 32768 | 4.47 | 2.53 | 1.0 | 10.0 | 0.19 | 0.0 |  |
| 49152 | 16.46 | 7.18 | 3.0 | 34.0 | 11.46 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.21 | 94.24 | 54.0 | 95.0 | 191.22 | 98.0 |  |
| 49840128 | 93.69 | 94.14 | 14.0 | 95.0 | 188.71 | 96.0 |  |
| 49856512 | 93.91 | 94.27 | 14.0 | 95.0 | 189.925 | 97.0 |  |
| 49872896 | 95.0 | 94.28 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49889280 | 94.37 | 94.28 | 75.0 | 95.0 | 188.395 | 95.0 |  |
| 49905664 | 94.67 | 94.22 | 62.0 | 95.0 | 192.675 | 99.0 |  |
| 49922048 | 94.55 | 94.17 | 52.0 | 95.0 | 191.56 | 98.0 |  |
| 49938432 | 92.52 | 94.06 | 16.0 | 95.0 | 185.55 | 94.0 |  |
| 49954816 | 93.7 | 94.14 | 58.0 | 95.0 | 187.725 | 95.0 |  |
| 49971200 | 93.73 | 94.03 | 50.0 | 95.0 | 188.75 | 96.0 |  |
| 49987584 | 93.7 | 94.04 | 54.0 | 95.0 | 186.73 | 94.0 |  |
| 50003968 | 94.68 | 94.05 | 63.0 | 95.0 | 192.685 | 99.0 |  |
