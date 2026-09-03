# b10ba-g93-seed3

step **20,938,752** · 1270 evals · trailing **93.26** · peak **93.85** @12,419,072 · sef **10.9** · best30 **83.8** @20,938,752

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.93 |
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
| ppo_horizon | 11.3 |
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

![b10ba-g93-seed3](b10ba-g93-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.05 | 0.05 | 0.0 | 1.0 | -0.45 | 0.0 |  |
| 32768 | 0.16 | 0.11 | 0.0 | 2.0 | -0.34 | 0.0 |  |
| 49152 | 14.41 | 4.87 | 0.0 | 41.0 | 12.02 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 20627456 | 91.72 | 92.75 | 21.0 | 95.0 | 173.805 | 83.0 |  |
| 20643840 | 93.58 | 92.82 | 72.0 | 95.0 | 178.65 | 86.0 |  |
| 20660224 | 94.28 | 92.73 | 82.0 | 95.0 | 179.35 | 86.0 |  |
| 20676608 | 93.45 | 92.97 | 66.0 | 95.0 | 173.545 | 81.0 |  |
| 20692992 | 93.27 | 92.98 | 16.0 | 95.0 | 176.35 | 84.0 |  |
| 20709376 | 93.62 | 93.05 | 73.0 | 95.0 | 176.7 | 84.0 |  |
| 20725760 | 94.21 | 93.13 | 80.0 | 95.0 | 183.26 | 90.0 |  |
| 20742144 | 92.83 | 93.37 | 42.0 | 95.0 | 179.89 | 88.0 |  |
| 20791296 | 94.19 | 93.29 | 57.0 | 95.0 | 184.19 | 91.0 |  |
| 20807680 | 93.62 | 93.21 | 44.0 | 95.0 | 174.71 | 82.0 |  |
| 20840448 | 93.01 | 93.16 | 37.0 | 95.0 | 170.075 | 78.0 |  |
| 20938752 | 94.18 | 93.26 | 72.0 | 95.0 | 177.26 | 84.0 |  |
