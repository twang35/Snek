# b9bt-lam97-seed2

step **50,003,968** · 3052 evals · trailing **94.3** · peak **94.62** @28,770,304 · sef **91.5** · best30 **98.0** @28,819,456

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
| ppo_gae_lambda | 0.97 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 25.2 |
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

![b9bt-lam97-seed2](b9bt-lam97-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.52 | 2.52 | 0.0 | 6.0 | -0.815 | 0.0 |  |
| 32768 | 17.25 | 17.36 | 7.0 | 34.0 | 12.7 | 0.0 |  |
| 49152 | 23.13 | 12.82 | 5.0 | 44.0 | 18.13 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.1 | 94.33 | 68.0 | 95.0 | 185.14 | 92.0 |  |
| 49840128 | 94.41 | 94.27 | 54.0 | 95.0 | 190.38 | 97.0 |  |
| 49856512 | 93.77 | 94.32 | 59.0 | 95.0 | 186.8 | 94.0 |  |
| 49872896 | 93.38 | 94.22 | 8.0 | 95.0 | 186.365 | 94.0 |  |
| 49889280 | 93.97 | 94.3 | 57.0 | 95.0 | 187.995 | 95.0 |  |
| 49905664 | 93.8 | 94.27 | 32.0 | 95.0 | 186.785 | 94.0 |  |
| 49922048 | 93.91 | 94.33 | 6.0 | 95.0 | 187.935 | 95.0 |  |
| 49938432 | 93.12 | 94.31 | 6.0 | 95.0 | 185.155 | 93.0 |  |
| 49954816 | 93.72 | 94.32 | 6.0 | 95.0 | 189.735 | 97.0 |  |
| 49971200 | 94.26 | 94.34 | 21.0 | 95.0 | 192.22 | 99.0 |  |
| 49987584 | 94.96 | 94.33 | 91.0 | 95.0 | 192.965 | 99.0 |  |
| 50003968 | 94.21 | 94.3 | 59.0 | 95.0 | 190.225 | 97.0 |  |
