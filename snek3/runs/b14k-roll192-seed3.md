# b14k-roll192-seed3

step **32,489,472** · 1312 evals · trailing **94.4** · peak **94.51** @30,793,728 · sef **88.1** · best30 **97.9** @31,703,040

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
| seed | 3 |
| torch_threads | 1 |

![b14k-roll192-seed3](b14k-roll192-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 24576 | 0.05 | 0.05 | 0.0 | 1.0 | -2.025 | 0.0 |  |
| 49152 | 6.39 | 8.26 | 1.0 | 18.0 | 5.125 | 0.0 |  |
| 73728 | 18.35 | 9.2 | 0.0 | 33.0 | 13.755 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 31973376 | 94.55 | 94.38 | 62.0 | 95.0 | 191.56 | 98.0 |  |
| 31997952 | 94.64 | 94.38 | 77.0 | 95.0 | 191.65 | 98.0 |  |
| 32022528 | 93.74 | 94.37 | 18.0 | 95.0 | 188.76 | 96.0 |  |
| 32047104 | 94.09 | 94.34 | 62.0 | 95.0 | 188.115 | 95.0 |  |
| 32071680 | 94.62 | 94.38 | 76.0 | 95.0 | 190.635 | 97.0 |  |
| 32096256 | 94.87 | 94.34 | 88.0 | 95.0 | 191.88 | 98.0 |  |
| 32120832 | 94.34 | 94.37 | 52.0 | 95.0 | 190.31 | 97.0 |  |
| 32145408 | 95.0 | 94.39 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 32169984 | 93.5 | 94.36 | 14.0 | 95.0 | 186.485 | 94.0 |  |
| 32194560 | 94.63 | 94.38 | 84.0 | 95.0 | 189.65 | 96.0 |  |
| 32268288 | 94.58 | 94.38 | 56.0 | 95.0 | 191.59 | 98.0 |  |
| 32489472 | 94.21 | 94.4 | 56.0 | 95.0 | 190.225 | 97.0 |  |
