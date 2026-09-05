# b14n-roll256-seed2

step **41,451,520** · 1262 evals · trailing **94.4** · peak **94.55** @15,106,048 · sef **88.0** · best30 **98.2** @41,451,520

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
| seed | 2 |
| torch_threads | 1 |

![b14n-roll256-seed2](b14n-roll256-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 3.77 | 3.77 | 1.0 | 10.0 | -0.915 | 0.0 |  |
| 65536 | 17.45 | 15.4 | 4.0 | 38.0 | 12.495 | 0.0 |  |
| 98304 | 24.98 | 14.38 | 7.0 | 43.0 | 19.98 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 40992768 | 94.77 | 94.2 | 84.0 | 95.0 | 190.785 | 97.0 |  |
| 41025536 | 94.63 | 94.25 | 58.0 | 95.0 | 192.635 | 99.0 |  |
| 41058304 | 94.23 | 94.35 | 56.0 | 95.0 | 191.24 | 98.0 |  |
| 41091072 | 92.97 | 94.15 | 10.0 | 95.0 | 188.985 | 97.0 |  |
| 41123840 | 95.0 | 94.23 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 41156608 | 94.71 | 94.23 | 76.0 | 95.0 | 191.72 | 98.0 |  |
| 41189376 | 94.75 | 94.23 | 70.0 | 95.0 | 192.755 | 99.0 |  |
| 41222144 | 94.93 | 94.35 | 88.0 | 95.0 | 192.935 | 99.0 |  |
| 41254912 | 95.0 | 94.37 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 41320448 | 92.73 | 94.37 | 12.0 | 95.0 | 188.745 | 97.0 |  |
| 41418752 | 93.95 | 94.39 | 28.0 | 95.0 | 190.96 | 98.0 |  |
| 41451520 | 95.0 | 94.4 | 95.0 | 95.0 | 194.0 | 100.0 |  |
