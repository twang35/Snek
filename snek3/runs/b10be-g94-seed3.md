# b10be-g94-seed3

step **20,463,616** · 1247 evals · trailing **92.12** · peak **93.92** @6,553,600 · sef **26.5** · best30 **87.7** @17,612,800

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
| seed | 3 |
| torch_threads | 1 |

![b10be-g94-seed3](b10be-g94-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.05 | 0.05 | 0.0 | 1.0 | -0.585 | 0.0 |  |
| 32768 | 0.32 | 0.18 | 0.0 | 2.0 | -0.18 | 0.0 |  |
| 49152 | 9.76 | 3.38 | 1.0 | 26.0 | 7.01 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 20250624 | 93.31 | 92.12 | 28.0 | 95.0 | 168.43 | 76.0 |  |
| 20267008 | 91.32 | 92.02 | 12.0 | 95.0 | 163.41 | 73.0 |  |
| 20283392 | 90.94 | 92.03 | 28.0 | 95.0 | 147.155 | 57.0 |  |
| 20299776 | 91.81 | 91.97 | 36.0 | 95.0 | 155.985 | 65.0 |  |
| 20316160 | 91.13 | 92.08 | 18.0 | 95.0 | 134.365 | 44.0 |  |
| 20332544 | 88.65 | 91.95 | 1.0 | 95.0 | 123.97 | 36.0 |  |
| 20348928 | 91.86 | 92.1 | 36.0 | 95.0 | 145.09 | 54.0 |  |
| 20365312 | 90.14 | 91.8 | 3.0 | 95.0 | 141.335 | 52.0 |  |
| 20381696 | 93.78 | 91.8 | 76.0 | 95.0 | 164.92 | 72.0 |  |
| 20398080 | 93.27 | 91.87 | 54.0 | 95.0 | 167.395 | 75.0 |  |
| 20430848 | 93.91 | 91.97 | 42.0 | 95.0 | 176.99 | 84.0 |  |
| 20463616 | 94.45 | 92.12 | 88.0 | 95.0 | 176.535 | 83.0 |  |
