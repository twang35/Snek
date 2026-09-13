# b9bf-lam93-seed4

step **50,003,968** · 3052 evals · trailing **94.23** · peak **94.43** @20,283,392 · sef **90.1** · best30 **97.0** @14,696,448

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
| ppo_gae_lambda | 0.93 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 12.6 |
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

![b9bf-lam93-seed4](b9bf-lam93-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 3.14 | 3.14 | 0.0 | 10.0 | 0.75 | 0.0 |  |
| 32768 | 11.2 | 22.9 | 0.0 | 43.0 | 9.44 | 0.0 |  |
| 49152 | 30.93 | 17.04 | 9.0 | 50.0 | 25.975 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.43 | 94.33 | 66.0 | 95.0 | 190.445 | 97.0 |  |
| 49840128 | 94.11 | 94.35 | 42.0 | 95.0 | 190.125 | 97.0 |  |
| 49856512 | 94.49 | 94.31 | 71.0 | 95.0 | 189.51 | 96.0 |  |
| 49872896 | 93.96 | 94.27 | 22.0 | 95.0 | 188.98 | 96.0 |  |
| 49889280 | 94.7 | 94.34 | 81.0 | 95.0 | 190.715 | 97.0 |  |
| 49905664 | 94.0 | 94.32 | 65.0 | 95.0 | 186.985 | 94.0 |  |
| 49922048 | 94.17 | 94.3 | 72.0 | 95.0 | 188.195 | 95.0 |  |
| 49938432 | 94.25 | 94.27 | 66.0 | 95.0 | 187.28 | 94.0 |  |
| 49954816 | 93.99 | 94.25 | 67.0 | 95.0 | 185.03 | 92.0 |  |
| 49971200 | 94.29 | 94.28 | 69.0 | 95.0 | 190.305 | 97.0 |  |
| 49987584 | 93.56 | 94.27 | 22.0 | 95.0 | 186.59 | 94.0 |  |
| 50003968 | 94.31 | 94.23 | 72.0 | 95.0 | 187.34 | 94.0 |  |
