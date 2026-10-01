# b13ar-mb384-seed2

step **50,003,968** · 3052 evals · trailing **94.52** · peak **94.59** @38,141,952 · sef **91.5** · best30 **98.1** @33,931,264

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
| ppo_minibatch | 384 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b13ar-mb384-seed2](b13ar-mb384-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.57 | 0.57 | 0.0 | 3.0 | -0.425 | 0.0 |  |
| 32768 | 7.42 | 4.0 | 2.0 | 17.0 | 2.42 | 0.0 |  |
| 49152 | 11.11 | 6.37 | 3.0 | 24.0 | 6.11 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.01 | 94.48 | 48.0 | 95.0 | 188.985 | 96.0 |  |
| 49840128 | 93.54 | 94.15 | 22.0 | 95.0 | 189.465 | 97.0 |  |
| 49856512 | 94.55 | 94.42 | 58.0 | 95.0 | 190.565 | 97.0 |  |
| 49872896 | 95.0 | 94.29 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49889280 | 95.0 | 94.5 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49905664 | 94.37 | 94.51 | 59.0 | 95.0 | 191.38 | 98.0 |  |
| 49922048 | 93.95 | 94.47 | 22.0 | 95.0 | 189.92 | 97.0 |  |
| 49938432 | 94.67 | 94.5 | 62.0 | 95.0 | 192.675 | 99.0 |  |
| 49954816 | 94.83 | 94.52 | 84.0 | 95.0 | 191.84 | 98.0 |  |
| 49971200 | 92.74 | 94.4 | 6.0 | 95.0 | 187.715 | 96.0 |  |
| 49987584 | 93.34 | 94.47 | 8.0 | 95.0 | 190.305 | 98.0 |  |
| 50003968 | 93.4 | 94.52 | 22.0 | 95.0 | 189.325 | 97.0 |  |
