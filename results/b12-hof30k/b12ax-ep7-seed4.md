# b12ax-ep7-seed4

step **50,003,968** · 3052 evals · trailing **93.06** · peak **94.54** @35,127,296 · sef **90.8** · best30 **98.1** @35,241,984

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
| ppo_epochs | 7 |
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
| seed | 4 |
| torch_threads | 1 |

![b12ax-ep7-seed4](b12ax-ep7-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 4.57 | 4.57 | 1.0 | 12.0 | 0.245 | 0.0 |  |
| 32768 | 37.48 | 30.46 | 1.0 | 81.0 | 32.75 | 0.0 |  |
| 49152 | 38.51 | 27.94 | 3.0 | 82.0 | 33.555 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.75 | 93.29 | 56.0 | 95.0 | 187.64 | 95.0 |  |
| 49840128 | 93.19 | 93.29 | 20.0 | 95.0 | 184.185 | 92.0 |  |
| 49856512 | 92.69 | 93.26 | 20.0 | 95.0 | 185.585 | 94.0 |  |
| 49872896 | 94.47 | 93.03 | 54.0 | 95.0 | 191.435 | 98.0 |  |
| 49889280 | 91.76 | 93.29 | 22.0 | 95.0 | 183.57 | 93.0 |  |
| 49905664 | 94.77 | 93.33 | 72.0 | 95.0 | 192.775 | 99.0 |  |
| 49922048 | 93.3 | 93.05 | 12.0 | 95.0 | 185.29 | 93.0 |  |
| 49938432 | 93.15 | 93.21 | 19.0 | 95.0 | 181.07 | 89.0 |  |
| 49954816 | 89.65 | 93.04 | 13.0 | 95.0 | 174.54 | 86.0 |  |
| 49971200 | 92.52 | 93.08 | 13.0 | 95.0 | 184.375 | 93.0 |  |
| 49987584 | 94.23 | 93.11 | 36.0 | 95.0 | 191.195 | 98.0 |  |
| 50003968 | 94.16 | 93.06 | 14.0 | 95.0 | 191.125 | 98.0 |  |
