# b12ac-ep1-seed3

step **50,003,968** · 3052 evals · trailing **93.82** · peak **94.38** @48,824,320 · sef **73.1** · best30 **97.7** @48,939,008

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
| seed | 3 |
| torch_threads | 1 |

![b12ac-ep1-seed3](b12ac-ep1-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.05 | 0.05 | 0.0 | 1.0 | -2.295 | 0.0 |  |
| 32768 | 0.89 | 0.47 | 0.0 | 4.0 | -0.555 | 0.0 |  |
| 49152 | 3.35 | 1.43 | 2.0 | 12.0 | -1.38 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.19 | 93.75 | 55.0 | 95.0 | 190.205 | 97.0 |  |
| 49840128 | 93.07 | 93.86 | 8.0 | 95.0 | 188.09 | 96.0 |  |
| 49856512 | 93.91 | 93.83 | 53.0 | 95.0 | 189.925 | 97.0 |  |
| 49872896 | 94.45 | 93.89 | 65.0 | 95.0 | 191.46 | 98.0 |  |
| 49889280 | 93.81 | 93.75 | 12.0 | 95.0 | 189.78 | 97.0 |  |
| 49905664 | 93.62 | 93.76 | 54.0 | 95.0 | 187.645 | 95.0 |  |
| 49922048 | 94.55 | 93.84 | 61.0 | 95.0 | 191.56 | 98.0 |  |
| 49938432 | 94.54 | 93.87 | 59.0 | 95.0 | 191.55 | 98.0 |  |
| 49954816 | 94.85 | 93.86 | 80.0 | 95.0 | 192.855 | 99.0 |  |
| 49971200 | 94.59 | 93.91 | 71.0 | 95.0 | 191.6 | 98.0 |  |
| 49987584 | 93.81 | 93.82 | 14.0 | 95.0 | 190.82 | 98.0 |  |
| 50003968 | 92.1 | 93.82 | 48.0 | 95.0 | 183.14 | 92.0 |  |
