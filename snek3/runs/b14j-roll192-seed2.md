# b14j-roll192-seed2

step **40,747,008** · 1658 evals · trailing **93.94** · peak **94.35** @18,014,208 · sef **86.1** · best30 **98.1** @35,831,808

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 24576 |
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
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 192 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 24576 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 2 |
| torch_threads | 1 |

![b14j-roll192-seed2](b14j-roll192-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 24576 | 2.39 | 2.39 | 1.0 | 7.0 | -0.63 | 0.0 |  |
| 49152 | 13.56 | 7.98 | 4.0 | 30.0 | 8.605 | 0.0 |  |
| 73728 | 24.84 | 13.6 | 2.0 | 55.0 | 19.885 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 40476672 | 93.79 | 94.02 | 10.0 | 95.0 | 190.8 | 98.0 |  |
| 40501248 | 93.98 | 93.99 | 63.0 | 95.0 | 188.005 | 95.0 |  |
| 40525824 | 93.96 | 94.02 | 36.0 | 95.0 | 189.93 | 97.0 |  |
| 40550400 | 94.15 | 93.92 | 67.0 | 95.0 | 189.17 | 96.0 |  |
| 40574976 | 93.04 | 93.94 | 24.0 | 95.0 | 187.065 | 95.0 |  |
| 40599552 | 93.62 | 93.91 | 18.0 | 95.0 | 186.65 | 94.0 |  |
| 40624128 | 94.18 | 93.94 | 55.0 | 95.0 | 190.195 | 97.0 |  |
| 40648704 | 94.34 | 93.92 | 64.0 | 95.0 | 188.365 | 95.0 |  |
| 40673280 | 93.12 | 93.92 | 43.0 | 95.0 | 186.105 | 94.0 |  |
| 40697856 | 93.44 | 93.87 | 10.0 | 95.0 | 188.46 | 96.0 |  |
| 40722432 | 93.13 | 93.89 | 14.0 | 95.0 | 186.115 | 94.0 |  |
| 40747008 | 94.45 | 93.94 | 80.0 | 95.0 | 187.48 | 94.0 |  |
