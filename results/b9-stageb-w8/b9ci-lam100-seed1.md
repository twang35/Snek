# b9ci-lam100-seed1

step **50,003,968** · 3052 evals · trailing **94.5** · peak **94.56** @49,807,360 · sef **88.3** · best30 **98.4** @49,790,976

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
| ppo_gae_lambda | 1.0 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 100.0 |
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

![b9ci-lam100-seed1](b9ci-lam100-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 14.71 | 14.71 | 3.0 | 26.0 | 9.71 | 0.0 |  |
| 32768 | 23.49 | 19.1 | 4.0 | 45.0 | 18.49 | 0.0 |  |
| 49152 | 21.36 | 19.85 | 5.0 | 44.0 | 16.36 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.73 | 94.52 | 58.0 | 95.0 | 184.77 | 92.0 |  |
| 49840128 | 94.86 | 94.56 | 81.0 | 95.0 | 192.865 | 99.0 |  |
| 49856512 | 94.65 | 94.56 | 84.0 | 95.0 | 189.67 | 96.0 |  |
| 49872896 | 94.53 | 94.56 | 61.0 | 95.0 | 191.54 | 98.0 |  |
| 49889280 | 94.74 | 94.54 | 78.0 | 95.0 | 191.75 | 98.0 |  |
| 49905664 | 94.59 | 94.55 | 64.0 | 95.0 | 190.605 | 97.0 |  |
| 49922048 | 94.63 | 94.48 | 63.0 | 95.0 | 191.64 | 98.0 |  |
| 49938432 | 94.67 | 94.55 | 77.0 | 95.0 | 191.68 | 98.0 |  |
| 49954816 | 93.55 | 94.48 | 48.0 | 95.0 | 184.59 | 92.0 |  |
| 49971200 | 94.41 | 94.46 | 78.0 | 95.0 | 187.44 | 94.0 |  |
| 49987584 | 94.31 | 94.53 | 60.0 | 95.0 | 191.32 | 98.0 |  |
| 50003968 | 94.93 | 94.5 | 88.0 | 95.0 | 192.935 | 99.0 |  |
