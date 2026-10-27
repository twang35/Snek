# b15ad-ent0-seed4

step **50,003,968** · 3052 evals · trailing **94.13** · peak **94.58** @22,937,600 · sef **93.4** · best30 **97.9** @33,832,960

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
| ppo_entropy_coef | 0.0 |
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
| seed | 4 |
| torch_threads | 1 |

![b15ad-ent0-seed4](b15ad-ent0-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.22 | 0.22 | 0.0 | 2.0 | -0.595 | 0.0 |  |
| 32768 | 14.53 | 18.88 | 1.0 | 26.0 | 10.43 | 0.0 |  |
| 49152 | 23.51 | 11.87 | 1.0 | 42.0 | 18.51 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.53 | 94.17 | 14.0 | 95.0 | 189.545 | 97.0 |  |
| 49840128 | 94.72 | 94.17 | 73.0 | 95.0 | 191.685 | 98.0 |  |
| 49856512 | 94.95 | 94.23 | 90.0 | 95.0 | 192.955 | 99.0 |  |
| 49872896 | 94.61 | 94.23 | 60.0 | 95.0 | 191.575 | 98.0 |  |
| 49889280 | 94.55 | 94.26 | 66.0 | 95.0 | 190.565 | 97.0 |  |
| 49905664 | 94.16 | 94.28 | 59.0 | 95.0 | 189.18 | 96.0 |  |
| 49922048 | 94.86 | 94.31 | 87.0 | 95.0 | 191.87 | 98.0 |  |
| 49938432 | 94.3 | 94.08 | 35.0 | 95.0 | 191.265 | 98.0 |  |
| 49954816 | 94.17 | 94.2 | 12.0 | 95.0 | 192.175 | 99.0 |  |
| 49971200 | 94.59 | 94.1 | 68.0 | 95.0 | 191.6 | 98.0 |  |
| 49987584 | 94.13 | 94.1 | 32.0 | 95.0 | 190.1 | 97.0 |  |
| 50003968 | 94.17 | 94.13 | 12.0 | 95.0 | 192.175 | 99.0 |  |
