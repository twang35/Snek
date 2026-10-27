# b7ai-fc300x100-seed1

step **50,003,968** · 3052 evals · trailing **94.13** · peak **94.5** @49,414,144 · sef **95.7** · best30 **97.3** @4,358,144

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
| fc_layers | (300, 100) |
| graph_eval_episodes | 100 |
| max_steps | 50003968 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

![b7ai-fc300x100-seed1](b7ai-fc300x100-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 8.04 | 8.04 | 0.0 | 30.0 | 6.28 | 0.0 |  |
| 32768 | 33.07 | 20.55 | 0.0 | 84.0 | 28.295 | 0.0 |  |
| 49152 | 43.99 | 34.76 | 11.0 | 88.0 | 38.99 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.11 | 94.09 | 46.0 | 95.0 | 186.1 | 93.0 |  |
| 49840128 | 93.75 | 94.11 | 60.0 | 95.0 | 183.705 | 91.0 |  |
| 49856512 | 94.36 | 94.27 | 84.0 | 95.0 | 182.325 | 89.0 |  |
| 49872896 | 93.4 | 94.19 | 64.0 | 95.0 | 177.385 | 85.0 |  |
| 49889280 | 94.41 | 94.1 | 81.0 | 95.0 | 186.445 | 93.0 |  |
| 49905664 | 93.0 | 94.29 | 25.0 | 95.0 | 181.96 | 90.0 |  |
| 49922048 | 94.65 | 94.38 | 76.0 | 95.0 | 190.665 | 97.0 |  |
| 49938432 | 93.49 | 94.22 | 24.0 | 95.0 | 188.51 | 96.0 |  |
| 49954816 | 94.61 | 94.1 | 66.0 | 95.0 | 189.54 | 96.0 |  |
| 49971200 | 93.58 | 94.12 | 18.0 | 95.0 | 187.56 | 95.0 |  |
| 49987584 | 94.23 | 94.11 | 32.0 | 95.0 | 189.25 | 96.0 |  |
| 50003968 | 94.75 | 94.13 | 87.0 | 95.0 | 189.725 | 96.0 |  |
