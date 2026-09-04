# b10bj-g95-seed4

step **50,003,968** · 3052 evals · trailing **93.66** · peak **94.26** @16,416,768 · sef **53.0** · best30 **93.5** @33,931,264

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.95 |
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
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 14.5 |
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

![b10bj-g95-seed4](b10bj-g95-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.74 | 2.74 | 0.0 | 11.0 | 0.71 | 0.0 |  |
| 32768 | 4.91 | 13.13 | 0.0 | 29.0 | 4.275 | 0.0 |  |
| 49152 | 23.05 | 15.61 | 1.0 | 46.0 | 18.905 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.77 | 93.65 | 26.0 | 95.0 | 183.815 | 91.0 |  |
| 49840128 | 94.38 | 93.66 | 82.0 | 95.0 | 180.445 | 87.0 |  |
| 49856512 | 92.91 | 93.63 | 22.0 | 95.0 | 175.99 | 84.0 |  |
| 49872896 | 93.19 | 93.69 | 43.0 | 95.0 | 174.28 | 82.0 |  |
| 49889280 | 92.58 | 93.55 | 31.0 | 95.0 | 178.645 | 87.0 |  |
| 49905664 | 91.46 | 93.41 | 12.0 | 95.0 | 174.54 | 84.0 |  |
| 49922048 | 92.1 | 93.38 | 6.0 | 95.0 | 173.19 | 82.0 |  |
| 49938432 | 93.06 | 93.68 | 9.0 | 95.0 | 181.115 | 89.0 |  |
| 49954816 | 93.31 | 93.7 | 5.0 | 95.0 | 183.355 | 91.0 |  |
| 49971200 | 94.65 | 93.72 | 72.0 | 95.0 | 190.665 | 97.0 |  |
| 49987584 | 93.68 | 93.69 | 27.0 | 95.0 | 180.74 | 88.0 |  |
| 50003968 | 92.31 | 93.66 | 1.0 | 95.0 | 175.39 | 84.0 |  |
