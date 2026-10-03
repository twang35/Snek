# b11av-lr8e4-seed2

step **50,003,968** · 3052 evals · trailing **92.78** · peak **94.61** @27,836,416 · sef **91.5** · best30 **97.8** @27,885,568

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
| ppo_learning_rate | 0.0008 |
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

![b11av-lr8e4-seed2](b11av-lr8e4-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.95 | 1.95 | 0.0 | 4.0 | -2.195 | 0.0 |  |
| 32768 | 23.26 | 21.03 | 8.0 | 57.0 | 18.755 | 0.0 |  |
| 49152 | 25.21 | 13.58 | 10.0 | 42.0 | 20.21 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.62 | 92.54 | 58.0 | 95.0 | 184.66 | 92.0 |  |
| 49840128 | 92.49 | 92.5 | 21.0 | 95.0 | 186.515 | 95.0 |  |
| 49856512 | 92.91 | 92.64 | 10.0 | 95.0 | 185.895 | 94.0 |  |
| 49872896 | 93.36 | 92.63 | 3.0 | 95.0 | 188.38 | 96.0 |  |
| 49889280 | 93.81 | 92.7 | 5.0 | 95.0 | 190.82 | 98.0 |  |
| 49905664 | 91.55 | 92.55 | 3.0 | 95.0 | 177.615 | 87.0 |  |
| 49922048 | 90.02 | 92.55 | 7.0 | 95.0 | 171.11 | 82.0 |  |
| 49938432 | 93.14 | 92.57 | 7.0 | 95.0 | 184.135 | 92.0 |  |
| 49954816 | 93.06 | 92.55 | 7.0 | 95.0 | 181.115 | 89.0 |  |
| 49971200 | 92.98 | 92.8 | 16.0 | 95.0 | 179.0 | 87.0 |  |
| 49987584 | 93.83 | 92.79 | 57.0 | 95.0 | 186.86 | 94.0 |  |
| 50003968 | 92.07 | 92.78 | 12.0 | 95.0 | 175.105 | 84.0 |  |
