# b10bd-g94-seed2

step **20,316,160** · 1237 evals · trailing **92.47** · peak **94.01** @15,220,736 · sef **18.5** · best30 **88.7** @15,040,512

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
| seed | 2 |
| torch_threads | 1 |

![b10bd-g94-seed2](b10bd-g94-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.1 | 2.1 | 0.0 | 6.0 | -1.01 | 0.0 |  |
| 32768 | 11.26 | 6.68 | 0.0 | 22.0 | 6.44 | 0.0 |  |
| 49152 | 21.54 | 14.57 | 0.0 | 49.0 | 16.72 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 20086784 | 92.92 | 93.01 | 26.0 | 95.0 | 165.055 | 73.0 |  |
| 20103168 | 93.41 | 93.07 | 78.0 | 95.0 | 163.51 | 71.0 |  |
| 20119552 | 93.23 | 93.07 | 80.0 | 95.0 | 154.375 | 62.0 |  |
| 20135936 | 91.18 | 92.9 | 10.0 | 95.0 | 141.425 | 51.0 |  |
| 20152320 | 90.06 | 92.83 | 8.0 | 95.0 | 125.38 | 36.0 |  |
| 20168704 | 91.37 | 92.74 | 72.0 | 95.0 | 130.625 | 40.0 |  |
| 20185088 | 90.86 | 92.64 | 59.0 | 95.0 | 122.155 | 32.0 |  |
| 20201472 | 91.02 | 92.55 | 10.0 | 95.0 | 136.245 | 46.0 |  |
| 20217856 | 90.66 | 92.47 | 10.0 | 95.0 | 134.935 | 45.0 |  |
| 20234240 | 91.94 | 92.46 | 75.0 | 95.0 | 134.225 | 43.0 |  |
| 20299776 | 92.72 | 92.48 | 74.0 | 95.0 | 137.99 | 46.0 |  |
| 20316160 | 92.86 | 92.47 | 78.0 | 95.0 | 138.13 | 46.0 |  |
