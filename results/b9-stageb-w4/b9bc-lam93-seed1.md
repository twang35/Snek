# b9bc-lam93-seed1

step **50,003,968** · 3052 evals · trailing **93.46** · peak **94.37** @25,821,184 · sef **91.3** · best30 **97.3** @9,568,256

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
| seed | 1 |
| torch_threads | 1 |

![b9bc-lam93-seed1](b9bc-lam93-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 9.45 | 9.45 | 0.0 | 26.0 | 8.95 | 0.0 |  |
| 32768 | 58.85 | 42.58 | 1.0 | 81.0 | 54.795 | 0.0 |  |
| 49152 | 52.85 | 39.02 | 18.0 | 86.0 | 48.66 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.8 | 93.68 | 80.0 | 95.0 | 191.81 | 98.0 |  |
| 49840128 | 93.8 | 93.55 | 20.0 | 95.0 | 186.785 | 94.0 |  |
| 49856512 | 93.77 | 93.62 | 12.0 | 95.0 | 187.795 | 95.0 |  |
| 49872896 | 94.14 | 93.55 | 71.0 | 95.0 | 185.18 | 92.0 |  |
| 49889280 | 94.78 | 93.69 | 81.0 | 95.0 | 191.745 | 98.0 |  |
| 49905664 | 93.53 | 93.55 | 61.0 | 95.0 | 186.56 | 94.0 |  |
| 49922048 | 94.82 | 93.57 | 82.0 | 95.0 | 191.83 | 98.0 |  |
| 49938432 | 93.99 | 93.51 | 26.0 | 95.0 | 190.005 | 97.0 |  |
| 49954816 | 92.17 | 93.46 | 18.0 | 95.0 | 182.17 | 91.0 |  |
| 49971200 | 94.06 | 93.45 | 5.0 | 95.0 | 191.07 | 98.0 |  |
| 49987584 | 94.0 | 93.46 | 8.0 | 95.0 | 189.97 | 97.0 |  |
| 50003968 | 94.73 | 93.46 | 83.0 | 95.0 | 189.705 | 96.0 |  |
