# b19o-adameps1e8-seed3

step **6,225,920** · 374 evals · trailing **91.49** · peak **93.81** @3,801,088 · sef **45.5** · best30 **90.6** @6,045,696

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
| ppo_adam_epsilon | 1e-08 |
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b19o-adameps1e8-seed3](b19o-adameps1e8-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.0 | 0.0 | 0.0 | 0.0 | -4.112 | 0.0 |  |
| 32768 | 2.19 | 1.09 | 1.0 | 10.0 | 1.575 | 0.0 |  |
| 49152 | 12.47 | 11.96 | 0.0 | 42.0 | 9.461 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 5947392 | 91.35 | 91.58 | 53.0 | 95.0 | 178.034 | 88.0 |  |
| 5963776 | 92.31 | 90.75 | 53.0 | 95.0 | 181.991 | 91.0 |  |
| 5980160 | 92.43 | 91.56 | 52.0 | 95.0 | 184.107 | 93.0 |  |
| 5996544 | 91.13 | 91.61 | 51.0 | 95.0 | 178.808 | 89.0 |  |
| 6012928 | 90.63 | 91.58 | 28.0 | 95.0 | 178.336 | 89.0 |  |
| 6029312 | 94.58 | 91.77 | 73.0 | 95.0 | 190.228 | 97.0 |  |
| 6045696 | 94.07 | 91.84 | 57.0 | 95.0 | 188.721 | 96.0 |  |
| 6062080 | 85.86 | 91.32 | 6.0 | 95.0 | 162.597 | 78.0 |  |
| 6078464 | 87.72 | 91.73 | 27.0 | 95.0 | 171.448 | 85.0 |  |
| 6160384 | 81.42 | 91.42 | 22.0 | 95.0 | 156.198 | 76.0 |  |
| 6193152 | 89.17 | 91.74 | 12.0 | 95.0 | 171.876 | 84.0 |  |
| 6225920 | 92.78 | 91.49 | 10.0 | 95.0 | 187.42 | 96.0 |  |
