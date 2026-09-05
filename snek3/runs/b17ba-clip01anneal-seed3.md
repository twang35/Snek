# b17ba-clip01anneal-seed3

step **7,274,496** · 440 evals · trailing **88.73** · peak **93.89** @3,522,560 · sef **42.5** · best30 **91.9** @6,455,296

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
| ppo_adam_epsilon | 1e-07 |
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.1 |
| ppo_clip_final | 0.02 |
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

![b17ba-clip01anneal-seed3](b17ba-clip01anneal-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.0 | 0.0 | 0.0 | 0.0 | -5.001 | 0.0 |  |
| 32768 | 0.07 | 0.04 | 0.0 | 1.0 | -0.481 | 0.0 |  |
| 49152 | 0.18 | 0.08 | 0.0 | 2.0 | -0.372 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 7061504 | 93.37 | 86.3 | 56.0 | 95.0 | 186.034 | 94.0 |  |
| 7077888 | 92.8 | 86.52 | 43.0 | 95.0 | 184.456 | 93.0 |  |
| 7094272 | 90.69 | 86.29 | 24.0 | 95.0 | 178.394 | 89.0 |  |
| 7110656 | 92.08 | 86.59 | 49.0 | 95.0 | 181.772 | 91.0 |  |
| 7127040 | 91.29 | 87.19 | 16.0 | 95.0 | 176.969 | 87.0 |  |
| 7143424 | 92.96 | 87.42 | 55.0 | 95.0 | 185.615 | 94.0 |  |
| 7159808 | 94.06 | 87.9 | 44.0 | 95.0 | 189.683 | 97.0 |  |
| 7176192 | 91.3 | 88.21 | 10.0 | 95.0 | 181.969 | 92.0 |  |
| 7192576 | 93.26 | 89.1 | 37.0 | 95.0 | 187.899 | 96.0 |  |
| 7241728 | 92.03 | 88.5 | 8.0 | 95.0 | 182.686 | 92.0 |  |
| 7258112 | 93.69 | 88.46 | 52.0 | 95.0 | 188.356 | 96.0 |  |
| 7274496 | 94.22 | 88.73 | 67.0 | 95.0 | 188.879 | 96.0 |  |
