# b13al-mb128-seed4

step **50,003,968** · 3052 evals · trailing **93.45** · peak **94.56** @29,294,592 · sef **94.5** · best30 **98.3** @29,261,824

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
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 128 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b13al-mb128-seed4](b13al-mb128-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.29 | 0.29 | 0.0 | 2.0 | -0.39 | 0.0 |  |
| 32768 | 26.62 | 13.46 | 1.0 | 51.0 | 21.89 | 0.0 |  |
| 49152 | 30.58 | 19.16 | 8.0 | 52.0 | 25.58 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 94.05 | 92.93 | 50.0 | 95.0 | 189.975 | 97.0 |  |
| 49840128 | 94.93 | 92.88 | 90.0 | 95.0 | 191.94 | 98.0 |  |
| 49856512 | 94.67 | 92.88 | 70.0 | 95.0 | 191.68 | 98.0 |  |
| 49872896 | 94.93 | 92.84 | 88.0 | 95.0 | 192.935 | 99.0 |  |
| 49889280 | 92.8 | 92.8 | 36.0 | 95.0 | 185.695 | 94.0 |  |
| 49905664 | 93.79 | 92.92 | 14.0 | 95.0 | 189.76 | 97.0 |  |
| 49922048 | 94.45 | 93.11 | 74.0 | 95.0 | 190.465 | 97.0 |  |
| 49938432 | 95.0 | 93.05 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49954816 | 93.84 | 93.15 | 6.0 | 95.0 | 190.805 | 98.0 |  |
| 49971200 | 93.46 | 93.2 | 20.0 | 95.0 | 189.385 | 97.0 |  |
| 49987584 | 94.85 | 93.3 | 80.0 | 95.0 | 192.855 | 99.0 |  |
| 50003968 | 93.76 | 93.45 | 18.0 | 95.0 | 188.735 | 96.0 |  |
