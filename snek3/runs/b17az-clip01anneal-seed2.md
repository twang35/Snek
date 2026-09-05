# b17az-clip01anneal-seed2

step **7,847,936** · 474 evals · trailing **78.66** · peak **94.05** @4,538,368 · sef **35.9** · best30 **88.8** @6,062,080

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
| seed | 2 |
| torch_threads | 1 |

![b17az-clip01anneal-seed2](b17az-clip01anneal-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.12 | 0.12 | 0.0 | 2.0 | -4.259 | 0.0 |  |
| 32768 | 1.78 | 0.95 | 0.0 | 6.0 | -0.471 | 0.0 |  |
| 49152 | 6.82 | 8.05 | 0.0 | 21.0 | 3.209 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 7634944 | 66.38 | 82.58 | 21.0 | 95.0 | 119.28 | 54.0 |  |
| 7651328 | 79.5 | 82.52 | 28.0 | 95.0 | 147.278 | 69.0 |  |
| 7667712 | 82.48 | 82.18 | 28.0 | 95.0 | 156.231 | 75.0 |  |
| 7684096 | 85.3 | 82.42 | 32.0 | 95.0 | 162.038 | 78.0 |  |
| 7700480 | 83.33 | 82.43 | 28.0 | 95.0 | 161.078 | 79.0 |  |
| 7716864 | 69.99 | 81.3 | 26.0 | 95.0 | 125.868 | 57.0 |  |
| 7733248 | 71.37 | 80.6 | 25.0 | 95.0 | 130.221 | 60.0 |  |
| 7749632 | 64.27 | 79.85 | 6.0 | 95.0 | 112.202 | 49.0 |  |
| 7766016 | 82.16 | 78.61 | 29.0 | 95.0 | 153.913 | 73.0 |  |
| 7815168 | 84.31 | 79.59 | 12.0 | 95.0 | 159.062 | 76.0 |  |
| 7831552 | 88.0 | 78.73 | 41.0 | 95.0 | 167.71 | 81.0 |  |
| 7847936 | 90.73 | 78.66 | 32.0 | 95.0 | 179.42 | 90.0 |  |
