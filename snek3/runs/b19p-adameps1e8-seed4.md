# b19p-adameps1e8-seed4

step **6,225,920** · 373 evals · trailing **86.43** · peak **93.49** @2,768,896 · sef **53.1** · best30 **91.2** @4,603,904

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
| seed | 4 |
| torch_threads | 1 |

![b19p-adameps1e8-seed4](b19p-adameps1e8-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.33 | 0.33 | 0.0 | 2.0 | -0.623 | 0.0 |  |
| 32768 | 16.49 | 13.94 | 1.0 | 34.0 | 11.954 | 0.0 |  |
| 49152 | 25.01 | 12.67 | 4.0 | 45.0 | 19.978 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 5931008 | 91.7 | 87.95 | 8.0 | 95.0 | 181.388 | 91.0 |  |
| 5947392 | 92.16 | 88.06 | 44.0 | 95.0 | 182.798 | 92.0 |  |
| 5963776 | 91.78 | 87.25 | 52.0 | 95.0 | 179.475 | 89.0 |  |
| 5980160 | 92.91 | 87.7 | 54.0 | 95.0 | 183.593 | 92.0 |  |
| 5996544 | 91.38 | 87.23 | 55.0 | 95.0 | 179.083 | 89.0 |  |
| 6012928 | 89.31 | 87.22 | 43.0 | 95.0 | 171.03 | 83.0 |  |
| 6029312 | 85.74 | 87.13 | 28.0 | 95.0 | 163.466 | 79.0 |  |
| 6045696 | 87.83 | 86.51 | 10.0 | 95.0 | 169.558 | 83.0 |  |
| 6078464 | 90.89 | 87.02 | 49.0 | 95.0 | 177.592 | 88.0 |  |
| 6176768 | 85.21 | 86.73 | 27.0 | 95.0 | 163.939 | 80.0 |  |
| 6193152 | 87.19 | 86.28 | 27.0 | 95.0 | 170.913 | 85.0 |  |
| 6225920 | 84.72 | 86.43 | 12.0 | 95.0 | 163.46 | 80.0 |  |
