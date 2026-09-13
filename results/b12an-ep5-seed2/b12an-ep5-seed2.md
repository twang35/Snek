# b12an-ep5-seed2

step **50,003,968** · 3052 evals · trailing **94.19** · peak **94.51** @28,901,376 · sef **92.8** · best30 **97.5** @14,811,136

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
| ppo_epochs | 5 |
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

![b12an-ep5-seed2](b12an-ep5-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.99 | 1.99 | 0.0 | 5.0 | -1.615 | 0.0 |  |
| 32768 | 11.23 | 6.61 | 0.0 | 26.0 | 6.815 | 0.0 |  |
| 49152 | 21.87 | 14.95 | 6.0 | 41.0 | 16.87 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.63 | 94.12 | 76.0 | 95.0 | 190.645 | 97.0 |  |
| 49840128 | 94.0 | 94.17 | 50.0 | 95.0 | 188.975 | 96.0 |  |
| 49856512 | 94.23 | 94.13 | 75.0 | 95.0 | 185.27 | 92.0 |  |
| 49872896 | 93.74 | 94.22 | 62.0 | 95.0 | 185.775 | 93.0 |  |
| 49889280 | 94.0 | 94.26 | 64.0 | 95.0 | 189.02 | 96.0 |  |
| 49905664 | 93.55 | 94.12 | 10.0 | 95.0 | 187.575 | 95.0 |  |
| 49922048 | 94.8 | 94.17 | 80.0 | 95.0 | 190.815 | 97.0 |  |
| 49938432 | 94.97 | 94.15 | 92.0 | 95.0 | 192.975 | 99.0 |  |
| 49954816 | 93.77 | 94.1 | 62.0 | 95.0 | 185.715 | 93.0 |  |
| 49971200 | 94.27 | 94.24 | 69.0 | 95.0 | 188.295 | 95.0 |  |
| 49987584 | 94.81 | 94.22 | 76.0 | 95.0 | 192.815 | 99.0 |  |
| 50003968 | 94.37 | 94.19 | 66.0 | 95.0 | 190.385 | 97.0 |  |
