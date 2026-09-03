# b9bz-lam99-seed4

step **50,003,968** · 3052 evals · trailing **94.55** · peak **94.55** @49,954,816 · sef **90.4** · best30 **98.3** @47,841,280

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
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b9bz-lam99-seed4](b9bz-lam99-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.23 | 0.23 | 0.0 | 2.0 | -0.45 | 0.0 |  |
| 32768 | 22.03 | 18.38 | 3.0 | 41.0 | 17.03 | 0.0 |  |
| 49152 | 23.98 | 12.11 | 10.0 | 41.0 | 18.98 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 95.0 | 94.22 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49840128 | 95.0 | 94.38 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49856512 | 95.0 | 94.38 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49872896 | 94.76 | 94.5 | 82.0 | 95.0 | 191.77 | 98.0 |  |
| 49889280 | 95.0 | 94.28 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49905664 | 94.9 | 94.36 | 89.0 | 95.0 | 191.91 | 98.0 |  |
| 49922048 | 94.65 | 94.46 | 70.0 | 95.0 | 191.66 | 98.0 |  |
| 49938432 | 95.0 | 94.41 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49954816 | 94.3 | 94.55 | 63.0 | 95.0 | 189.32 | 96.0 |  |
| 49971200 | 94.67 | 94.52 | 79.0 | 95.0 | 190.685 | 97.0 |  |
| 49987584 | 94.41 | 94.54 | 64.0 | 95.0 | 191.42 | 98.0 |  |
| 50003968 | 95.0 | 94.55 | 95.0 | 95.0 | 194.0 | 100.0 |  |
