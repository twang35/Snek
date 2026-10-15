# b12ab-ep1-seed2

step **50,003,968** · 3052 evals · trailing **93.78** · peak **94.22** @45,236,224 · sef **74.9** · best30 **97.6** @35,897,344

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
| ppo_epochs | 1 |
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
| seed | 2 |
| torch_threads | 1 |

![b12ab-ep1-seed2](b12ab-ep1-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.44 | 0.44 | 0.0 | 3.0 | -0.33 | 0.0 |  |
| 32768 | 2.05 | 1.24 | 0.0 | 8.0 | 0.65 | 0.0 |  |
| 49152 | 8.59 | 3.69 | 2.0 | 25.0 | 3.59 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.25 | 93.9 | 67.0 | 95.0 | 189.27 | 96.0 |  |
| 49840128 | 93.75 | 93.86 | 58.0 | 95.0 | 187.775 | 95.0 |  |
| 49856512 | 94.48 | 93.86 | 61.0 | 95.0 | 191.49 | 98.0 |  |
| 49872896 | 93.59 | 93.84 | 54.0 | 95.0 | 187.615 | 95.0 |  |
| 49889280 | 92.98 | 93.8 | 55.0 | 95.0 | 185.015 | 93.0 |  |
| 49905664 | 93.72 | 93.78 | 55.0 | 95.0 | 187.745 | 95.0 |  |
| 49922048 | 93.39 | 93.79 | 57.0 | 95.0 | 186.42 | 94.0 |  |
| 49938432 | 91.6 | 93.71 | 53.0 | 95.0 | 177.665 | 87.0 |  |
| 49954816 | 93.53 | 93.7 | 58.0 | 95.0 | 186.56 | 94.0 |  |
| 49971200 | 94.77 | 93.81 | 75.0 | 95.0 | 191.78 | 98.0 |  |
| 49987584 | 94.19 | 93.81 | 54.0 | 95.0 | 190.205 | 97.0 |  |
| 50003968 | 93.97 | 93.78 | 56.0 | 95.0 | 188.99 | 96.0 |  |
