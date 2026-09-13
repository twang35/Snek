# b7ar-fc160x160-seed2

step **50,003,968** · 3052 evals · trailing **93.6** · peak **94.48** @49,037,312 · sef **94.2** · best30 **97.4** @49,201,152

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
| fc_layers | (160, 160) |
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
| seed | 2 |
| torch_threads | 1 |

![b7ar-fc160x160-seed2](b7ar-fc160x160-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 6.63 | 6.63 | 0.0 | 25.0 | 3.565 | 0.0 |  |
| 32768 | 35.83 | 23.69 | 9.0 | 65.0 | 30.83 | 0.0 |  |
| 49152 | 26.75 | 25.59 | 8.0 | 48.0 | 21.795 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.56 | 93.76 | 22.0 | 95.0 | 189.485 | 97.0 |  |
| 49840128 | 94.05 | 93.57 | 21.0 | 95.0 | 190.065 | 97.0 |  |
| 49856512 | 93.34 | 93.79 | 26.0 | 95.0 | 186.325 | 94.0 |  |
| 49872896 | 93.81 | 93.58 | 64.0 | 95.0 | 185.845 | 93.0 |  |
| 49889280 | 94.65 | 93.6 | 81.0 | 95.0 | 188.585 | 95.0 |  |
| 49905664 | 94.4 | 93.6 | 66.0 | 95.0 | 187.43 | 94.0 |  |
| 49922048 | 94.27 | 93.58 | 76.0 | 95.0 | 188.295 | 95.0 |  |
| 49938432 | 94.04 | 93.58 | 68.0 | 95.0 | 187.07 | 94.0 |  |
| 49954816 | 92.89 | 93.56 | 22.0 | 95.0 | 180.855 | 89.0 |  |
| 49971200 | 93.72 | 93.57 | 71.0 | 95.0 | 181.73 | 89.0 |  |
| 49987584 | 93.76 | 93.56 | 18.0 | 95.0 | 187.74 | 95.0 |  |
| 50003968 | 93.85 | 93.6 | 60.0 | 95.0 | 184.8 | 92.0 |  |
