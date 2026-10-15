# b17av-clipanneal-seed2

step **50,003,968** · 3052 evals · trailing **94.29** · peak **94.58** @32,751,616 · sef **93.4** · best30 **98.2** @46,448,640

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
| ppo_clip_final | 0.02 |
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
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b17av-clipanneal-seed2](b17av-clipanneal-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.85 | 1.85 | 0.0 | 5.0 | -0.979 | 0.0 |  |
| 32768 | 18.12 | 16.96 | 4.0 | 39.0 | 13.076 | 0.0 |  |
| 49152 | 20.26 | 17.78 | 4.0 | 57.0 | 15.226 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.78 | 94.29 | 73.0 | 95.0 | 192.489 | 99.0 |  |
| 49840128 | 95.0 | 94.3 | 95.0 | 95.0 | 193.716 | 100.0 |  |
| 49856512 | 94.31 | 94.29 | 66.0 | 95.0 | 190.035 | 97.0 |  |
| 49872896 | 93.75 | 94.26 | 6.0 | 95.0 | 190.472 | 98.0 |  |
| 49889280 | 94.58 | 94.31 | 79.0 | 95.0 | 189.297 | 96.0 |  |
| 49905664 | 93.71 | 94.31 | 10.0 | 95.0 | 189.431 | 97.0 |  |
| 49922048 | 94.78 | 94.33 | 78.0 | 95.0 | 191.5 | 98.0 |  |
| 49938432 | 94.83 | 94.31 | 85.0 | 95.0 | 190.544 | 97.0 |  |
| 49954816 | 94.06 | 94.32 | 57.0 | 95.0 | 186.78 | 94.0 |  |
| 49971200 | 94.28 | 94.34 | 58.0 | 95.0 | 188.02 | 95.0 |  |
| 49987584 | 94.65 | 94.36 | 72.0 | 95.0 | 190.367 | 97.0 |  |
| 50003968 | 94.93 | 94.29 | 88.0 | 95.0 | 192.624 | 99.0 |  |
