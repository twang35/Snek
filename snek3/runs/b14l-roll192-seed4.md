# b14l-roll192-seed4

step **39,149,568** · 1593 evals · trailing **94.27** · peak **94.43** @38,141,952 · sef **86.0** · best30 **98.1** @18,186,240

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
| seed | 4 |
| torch_threads | 1 |

![b14l-roll192-seed4](b14l-roll192-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 24576 | 0.91 | 0.91 | 0.0 | 4.0 | -4.0 | 0.0 |  |
| 49152 | 22.3 | 11.61 | 5.0 | 42.0 | 17.39 | 0.0 |  |
| 73728 | 24.75 | 15.99 | 5.0 | 46.0 | 19.75 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 38879232 | 94.59 | 94.26 | 59.0 | 95.0 | 191.6 | 98.0 |  |
| 38903808 | 94.65 | 94.26 | 60.0 | 95.0 | 192.655 | 99.0 |  |
| 38928384 | 94.42 | 94.33 | 64.0 | 95.0 | 191.43 | 98.0 |  |
| 38952960 | 94.25 | 94.24 | 68.0 | 95.0 | 190.265 | 97.0 |  |
| 38977536 | 93.62 | 94.27 | 49.0 | 95.0 | 188.64 | 96.0 |  |
| 39002112 | 94.62 | 94.33 | 57.0 | 95.0 | 192.625 | 99.0 |  |
| 39026688 | 94.4 | 94.26 | 56.0 | 95.0 | 191.41 | 98.0 |  |
| 39051264 | 94.13 | 94.32 | 8.0 | 95.0 | 192.135 | 99.0 |  |
| 39075840 | 94.78 | 94.31 | 73.0 | 95.0 | 192.785 | 99.0 |  |
| 39100416 | 92.97 | 94.24 | 33.0 | 95.0 | 184.96 | 93.0 |  |
| 39124992 | 94.29 | 94.26 | 24.0 | 95.0 | 192.295 | 99.0 |  |
| 39149568 | 94.25 | 94.27 | 55.0 | 95.0 | 191.26 | 98.0 |  |
