# b11aq-lr5e4-seed1

step **50,003,968** · 3052 evals · trailing **93.1** · peak **94.6** @21,889,024 · sef **91.8** · best30 **98.0** @39,895,040

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
| ppo_learning_rate | 0.0005 |
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

![b11aq-lr5e4-seed1](b11aq-lr5e4-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 20.24 | 20.24 | 0.0 | 45.0 | 15.87 | 0.0 |  |
| 32768 | 37.52 | 32.41 | 0.0 | 82.0 | 32.745 | 0.0 |  |
| 49152 | 32.24 | 26.24 | 15.0 | 81.0 | 27.24 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 89.85 | 93.11 | 8.0 | 95.0 | 150.635 | 62.0 |  |
| 49840128 | 93.06 | 94.06 | 75.0 | 95.0 | 170.08 | 78.0 |  |
| 49856512 | 90.9 | 93.93 | 1.0 | 95.0 | 172.94 | 83.0 |  |
| 49872896 | 92.6 | 93.61 | 67.0 | 95.0 | 170.57 | 79.0 |  |
| 49889280 | 90.65 | 93.68 | 8.0 | 95.0 | 165.5 | 76.0 |  |
| 49905664 | 92.61 | 93.53 | 75.0 | 95.0 | 170.715 | 79.0 |  |
| 49922048 | 90.84 | 93.27 | 12.0 | 95.0 | 162.75 | 73.0 |  |
| 49938432 | 94.01 | 93.09 | 75.0 | 95.0 | 184.01 | 91.0 |  |
| 49954816 | 94.49 | 93.6 | 77.0 | 95.0 | 187.43 | 94.0 |  |
| 49971200 | 94.45 | 93.52 | 77.0 | 95.0 | 189.425 | 96.0 |  |
| 49987584 | 92.81 | 93.05 | 11.0 | 95.0 | 179.645 | 88.0 |  |
| 50003968 | 94.38 | 93.1 | 76.0 | 95.0 | 186.37 | 93.0 |  |
