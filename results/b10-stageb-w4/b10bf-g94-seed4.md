# b10bf-g94-seed4

step **50,003,968** · 3052 evals · trailing **93.37** · peak **94.2** @32,768,000 · sef **26.3** · best30 **90.4** @33,062,912

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.94 |
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
| ppo_horizon | 12.7 |
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

![b10bf-g94-seed4](b10bf-g94-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.82 | 2.82 | 0.0 | 8.0 | 0.79 | 0.0 |  |
| 32768 | 2.73 | 2.77 | 0.0 | 18.0 | 2.23 | 0.0 |  |
| 49152 | 27.45 | 11.0 | 0.0 | 89.0 | 23.845 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 93.28 | 93.41 | 68.0 | 95.0 | 175.32 | 83.0 |  |
| 49840128 | 93.87 | 93.16 | 40.0 | 95.0 | 181.925 | 89.0 |  |
| 49856512 | 94.2 | 93.51 | 84.0 | 95.0 | 176.24 | 83.0 |  |
| 49872896 | 94.21 | 93.41 | 69.0 | 95.0 | 180.275 | 87.0 |  |
| 49889280 | 94.5 | 93.25 | 86.0 | 95.0 | 181.56 | 88.0 |  |
| 49905664 | 94.18 | 93.56 | 75.0 | 95.0 | 181.24 | 88.0 |  |
| 49922048 | 91.35 | 93.46 | 19.0 | 95.0 | 168.28 | 78.0 |  |
| 49938432 | 92.4 | 93.41 | 61.0 | 95.0 | 154.45 | 63.0 |  |
| 49954816 | 93.77 | 93.47 | 74.0 | 95.0 | 171.875 | 79.0 |  |
| 49971200 | 92.75 | 93.43 | 66.0 | 95.0 | 165.835 | 74.0 |  |
| 49987584 | 91.69 | 93.3 | 14.0 | 95.0 | 162.695 | 72.0 |  |
| 50003968 | 93.15 | 93.37 | 70.0 | 95.0 | 168.18 | 76.0 |  |
