# b11az-lr1e3-seed2

step **50,003,968** · 3052 evals · trailing **92.86** · peak **94.16** @32,112,640 · sef **92.0** · best30 **97.1** @25,690,112

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
| ppo_learning_rate | 0.001 |
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

![b11az-lr1e3-seed2](b11az-lr1e3-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.0 | 1.0 | 0.0 | 5.0 | -0.49 | 0.0 |  |
| 32768 | 8.55 | 24.15 | 0.0 | 46.0 | 7.015 | 0.0 |  |
| 49152 | 28.52 | 14.76 | 8.0 | 47.0 | 23.565 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.95 | 93.16 | 3.0 | 95.0 | 173.275 | 82.0 |  |
| 49840128 | 93.06 | 93.29 | 29.0 | 95.0 | 178.585 | 87.0 |  |
| 49856512 | 93.75 | 93.14 | 7.0 | 95.0 | 178.28 | 86.0 |  |
| 49872896 | 93.31 | 93.21 | 42.0 | 95.0 | 175.76 | 84.0 |  |
| 49889280 | 94.14 | 93.09 | 55.0 | 95.0 | 182.74 | 90.0 |  |
| 49905664 | 92.89 | 93.33 | 16.0 | 95.0 | 178.415 | 87.0 |  |
| 49922048 | 90.19 | 93.17 | 3.0 | 95.0 | 164.32 | 76.0 |  |
| 49938432 | 92.21 | 93.35 | 1.0 | 95.0 | 175.7 | 85.0 |  |
| 49954816 | 90.35 | 93.27 | 8.0 | 95.0 | 164.66 | 76.0 |  |
| 49971200 | 89.06 | 92.95 | 24.0 | 95.0 | 152.835 | 66.0 |  |
| 49987584 | 91.28 | 93.06 | 10.0 | 95.0 | 177.845 | 88.0 |  |
| 50003968 | 91.09 | 92.86 | 24.0 | 95.0 | 171.46 | 82.0 |  |
