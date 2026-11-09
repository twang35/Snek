# b12ai-ep3-seed1

step **50,003,968** · 3052 evals · trailing **94.12** · peak **94.61** @47,726,592 · sef **89.0** · best30 **98.4** @48,398,336

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
| seed | 1 |
| torch_threads | 1 |

![b12ai-ep3-seed1](b12ai-ep3-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 9.14 | 9.14 | 1.0 | 29.0 | 7.245 | 0.0 |  |
| 32768 | 34.36 | 25.01 | 14.0 | 65.0 | 29.36 | 0.0 |  |
| 49152 | 28.4 | 18.77 | 3.0 | 51.0 | 23.49 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.41 | 94.33 | 66.0 | 95.0 | 188.435 | 95.0 |  |
| 49840128 | 94.49 | 94.27 | 61.0 | 95.0 | 190.505 | 97.0 |  |
| 49856512 | 92.87 | 94.29 | 8.0 | 95.0 | 187.89 | 96.0 |  |
| 49872896 | 93.38 | 94.09 | 12.0 | 95.0 | 186.365 | 94.0 |  |
| 49889280 | 93.75 | 94.08 | 14.0 | 95.0 | 189.765 | 97.0 |  |
| 49905664 | 94.58 | 94.11 | 68.0 | 95.0 | 190.55 | 97.0 |  |
| 49922048 | 93.64 | 94.09 | 22.0 | 95.0 | 189.655 | 97.0 |  |
| 49938432 | 95.0 | 94.09 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49954816 | 93.13 | 94.11 | 30.0 | 95.0 | 183.13 | 91.0 |  |
| 49971200 | 93.62 | 94.26 | 12.0 | 95.0 | 188.64 | 96.0 |  |
| 49987584 | 91.73 | 94.17 | 8.0 | 95.0 | 181.73 | 91.0 |  |
| 50003968 | 94.26 | 94.12 | 78.0 | 95.0 | 187.29 | 94.0 |  |
