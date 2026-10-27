# b12ae-ep2-seed1

step **50,003,968** · 3052 evals · trailing **93.66** · peak **94.45** @27,099,136 · sef **87.0** · best30 **98.1** @27,033,600

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
| ppo_epochs | 2 |
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

![b12ae-ep2-seed1](b12ae-ep2-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 10.1 | 17.15 | 1.0 | 23.0 | 9.33 | 0.0 |  |
| 32768 | 15.05 | 15.05 | 3.0 | 30.0 | 10.05 | 0.0 |  |
| 49152 | 16.32 | 15.69 | 4.0 | 35.0 | 11.32 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.89 | 93.73 | 53.0 | 95.0 | 189.905 | 97.0 |  |
| 49840128 | 93.93 | 93.66 | 26.0 | 95.0 | 189.945 | 97.0 |  |
| 49856512 | 94.16 | 93.68 | 62.0 | 95.0 | 188.185 | 95.0 |  |
| 49872896 | 94.11 | 93.66 | 55.0 | 95.0 | 190.125 | 97.0 |  |
| 49889280 | 93.91 | 93.69 | 20.0 | 95.0 | 190.92 | 98.0 |  |
| 49905664 | 93.19 | 93.65 | 16.0 | 95.0 | 186.22 | 94.0 |  |
| 49922048 | 93.62 | 93.66 | 65.0 | 95.0 | 183.665 | 91.0 |  |
| 49938432 | 93.91 | 93.71 | 57.0 | 95.0 | 184.95 | 92.0 |  |
| 49954816 | 94.11 | 93.74 | 78.0 | 95.0 | 186.145 | 93.0 |  |
| 49971200 | 93.0 | 93.74 | 20.0 | 95.0 | 185.035 | 93.0 |  |
| 49987584 | 92.98 | 93.68 | 12.0 | 95.0 | 185.015 | 93.0 |  |
| 50003968 | 94.31 | 93.66 | 65.0 | 95.0 | 189.33 | 96.0 |  |
