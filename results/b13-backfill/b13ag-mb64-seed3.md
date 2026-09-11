# b13ag-mb64-seed3

step **50,003,968** · 3052 evals · trailing **92.78** · peak **94.24** @20,660,224 · sef **94.3** · best30 **97.3** @20,512,768

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
| ppo_minibatch | 64 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b13ag-mb64-seed3](b13ag-mb64-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.07 | 0.07 | 0.0 | 2.0 | -2.095 | 0.0 |  |
| 32768 | 1.04 | 0.56 | 0.0 | 6.0 | 0.54 | 0.0 |  |
| 49152 | 17.47 | 6.19 | 0.0 | 35.0 | 13.1 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.45 | 92.67 | 28.0 | 95.0 | 185.39 | 94.0 |  |
| 49840128 | 94.33 | 92.77 | 35.0 | 95.0 | 191.295 | 98.0 |  |
| 49856512 | 93.03 | 92.76 | 29.0 | 95.0 | 187.96 | 96.0 |  |
| 49872896 | 92.68 | 92.8 | 17.0 | 95.0 | 185.575 | 94.0 |  |
| 49889280 | 91.74 | 92.76 | 27.0 | 95.0 | 183.505 | 93.0 |  |
| 49905664 | 91.06 | 92.68 | 22.0 | 95.0 | 180.745 | 91.0 |  |
| 49922048 | 92.82 | 92.78 | 53.0 | 95.0 | 181.69 | 90.0 |  |
| 49938432 | 93.63 | 92.78 | 15.0 | 95.0 | 186.57 | 94.0 |  |
| 49954816 | 94.39 | 92.77 | 39.0 | 95.0 | 191.355 | 98.0 |  |
| 49971200 | 93.09 | 92.72 | 41.0 | 95.0 | 185.035 | 93.0 |  |
| 49987584 | 94.82 | 92.78 | 77.0 | 95.0 | 192.825 | 99.0 |  |
| 50003968 | 94.78 | 92.78 | 86.0 | 95.0 | 190.795 | 97.0 |  |
