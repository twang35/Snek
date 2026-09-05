# b14o-roll256-seed3

step **40,730,624** · 1239 evals · trailing **93.99** · peak **94.44** @27,721,728 · sef **88.0** · best30 **98.1** @27,623,424

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 32768 |
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
| ppo_rollout | 256 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 32768 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b14o-roll256-seed3](b14o-roll256-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 0.1 | 0.1 | 0.0 | 1.0 | -2.875 | 0.0 |  |
| 65536 | 1.74 | 0.92 | 1.0 | 7.0 | 1.195 | 0.0 |  |
| 98304 | 18.05 | 6.63 | 3.0 | 40.0 | 13.455 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 40239104 | 94.01 | 94.01 | 66.0 | 95.0 | 185.05 | 92.0 |  |
| 40271872 | 93.43 | 93.98 | 68.0 | 95.0 | 183.475 | 91.0 |  |
| 40304640 | 93.13 | 94.14 | 70.0 | 95.0 | 179.195 | 87.0 |  |
| 40337408 | 91.88 | 94.03 | 14.0 | 95.0 | 176.95 | 86.0 |  |
| 40370176 | 93.65 | 94.11 | 73.0 | 95.0 | 182.7 | 90.0 |  |
| 40402944 | 94.34 | 94.02 | 67.0 | 95.0 | 189.36 | 96.0 |  |
| 40435712 | 93.29 | 94.09 | 63.0 | 95.0 | 183.335 | 91.0 |  |
| 40468480 | 93.87 | 94.05 | 71.0 | 95.0 | 183.915 | 91.0 |  |
| 40501248 | 93.5 | 94.03 | 20.0 | 95.0 | 186.485 | 94.0 |  |
| 40599552 | 93.63 | 93.97 | 63.0 | 95.0 | 182.68 | 90.0 |  |
| 40632320 | 93.85 | 93.98 | 34.0 | 95.0 | 186.835 | 94.0 |  |
| 40730624 | 94.13 | 93.99 | 8.0 | 95.0 | 192.135 | 99.0 |  |
