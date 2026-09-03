# b10bc-g94-seed1

step **21,184,512** · 1286 evals · trailing **93.26** · peak **94.03** @12,058,624 · sef **25.6** · best30 **86.3** @20,529,152

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
| seed | 1 |
| torch_threads | 1 |

![b10bc-g94-seed1](b10bc-g94-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 9.76 | 9.76 | 0.0 | 26.0 | 9.215 | 0.0 |  |
| 32768 | 14.79 | 30.16 | 0.0 | 82.0 | 13.75 | 0.0 |  |
| 49152 | 56.31 | 35.39 | 16.0 | 95.0 | 54.015 | 1.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 20889600 | 93.09 | 93.29 | 69.0 | 95.0 | 166.13 | 74.0 |  |
| 20905984 | 92.96 | 93.24 | 6.0 | 95.0 | 171.065 | 79.0 |  |
| 20922368 | 92.31 | 93.25 | 14.0 | 95.0 | 175.39 | 84.0 |  |
| 20938752 | 92.2 | 93.16 | 12.0 | 95.0 | 173.29 | 82.0 |  |
| 20955136 | 92.57 | 93.2 | 43.0 | 95.0 | 174.655 | 83.0 |  |
| 20971520 | 93.97 | 93.21 | 50.0 | 95.0 | 184.015 | 91.0 |  |
| 20987904 | 94.7 | 93.18 | 78.0 | 95.0 | 189.72 | 96.0 |  |
| 21020672 | 93.24 | 93.19 | 8.0 | 95.0 | 177.27 | 85.0 |  |
| 21053440 | 93.91 | 93.2 | 69.0 | 95.0 | 172.965 | 80.0 |  |
| 21069824 | 92.68 | 93.19 | 28.0 | 95.0 | 161.785 | 70.0 |  |
| 21168128 | 93.89 | 93.18 | 49.0 | 95.0 | 181.9 | 89.0 |  |
| 21184512 | 94.31 | 93.26 | 74.0 | 95.0 | 183.315 | 90.0 |  |
