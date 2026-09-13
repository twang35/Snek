# b10cj-g100-seed2

step **50,003,968** · 3052 evals · trailing **91.17** · peak **94.62** @36,585,472 · sef **44.2** · best30 **98.7** @42,532,864

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 1.0 |
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
| ppo_horizon | 50.0 |
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

![b10cj-g100-seed2](b10cj-g100-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.63 | 1.63 | 0.0 | 8.0 | -0.805 | 0.0 |  |
| 32768 | 8.13 | 4.88 | 2.0 | 21.0 | 3.13 | 0.0 |  |
| 49152 | 7.99 | 5.92 | 2.0 | 21.0 | 2.99 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.02 | 91.6 | 32.0 | 95.0 | 184.055 | 93.0 |  |
| 49840128 | 87.27 | 91.22 | 26.0 | 95.0 | 168.36 | 82.0 |  |
| 49856512 | 87.2 | 91.35 | 29.0 | 95.0 | 171.275 | 85.0 |  |
| 49872896 | 89.72 | 91.65 | 8.0 | 95.0 | 178.77 | 90.0 |  |
| 49889280 | 90.62 | 91.59 | 24.0 | 95.0 | 181.66 | 92.0 |  |
| 49905664 | 85.82 | 91.51 | 30.0 | 95.0 | 167.905 | 83.0 |  |
| 49922048 | 92.63 | 91.74 | 39.0 | 95.0 | 185.615 | 94.0 |  |
| 49938432 | 92.98 | 91.53 | 30.0 | 95.0 | 187.005 | 95.0 |  |
| 49954816 | 92.09 | 91.67 | 30.0 | 95.0 | 184.125 | 93.0 |  |
| 49971200 | 91.59 | 90.7 | 47.0 | 95.0 | 181.635 | 91.0 |  |
| 49987584 | 93.92 | 89.93 | 53.0 | 95.0 | 188.94 | 96.0 |  |
| 50003968 | 92.96 | 91.17 | 49.0 | 95.0 | 185.99 | 94.0 |  |
