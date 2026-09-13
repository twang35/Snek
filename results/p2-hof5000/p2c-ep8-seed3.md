# p2c-ep8-seed3

step **265,207,808** · 16180 evals · trailing **93.16** · peak **94.52** @199,491,584 · sef **96.3** · best30 **97.8** @72,351,744

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 6 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 400000000 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 8 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![p2c-ep8-seed3](p2c-ep8-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.07 | 0.07 | 0.0 | 1.0 | -4.03 | 0.0 |  |
| 32768 | 1.38 | 0.72 | 0.0 | 8.0 | 0.88 | 0.0 |  |
| 49152 | 25.1 | 21.67 | 6.0 | 45.0 | 20.415 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 264912896 | 93.74 | 92.78 | 32.0 | 95.0 | 188.58 | 96.0 |  |
| 264929280 | 94.8 | 92.86 | 87.0 | 95.0 | 190.397 | 97.0 |  |
| 264945664 | 93.68 | 92.72 | 7.0 | 95.0 | 188.264 | 96.0 |  |
| 264962048 | 94.71 | 92.78 | 82.0 | 95.0 | 189.242 | 96.0 |  |
| 264978432 | 94.55 | 93.04 | 61.0 | 95.0 | 191.158 | 98.0 |  |
| 265027584 | 94.12 | 92.86 | 67.0 | 95.0 | 185.84 | 93.0 |  |
| 265043968 | 94.83 | 93.15 | 78.0 | 95.0 | 192.79 | 99.0 |  |
| 265060352 | 93.47 | 92.75 | 43.0 | 95.0 | 183.851 | 92.0 |  |
| 265076736 | 94.76 | 92.85 | 80.0 | 95.0 | 190.64 | 97.0 |  |
| 265093120 | 92.71 | 92.83 | 17.0 | 95.0 | 182.35 | 91.0 |  |
| 265109504 | 94.3 | 92.89 | 68.0 | 95.0 | 187.105 | 94.0 |  |
| 265207808 | 94.28 | 93.16 | 71.0 | 95.0 | 185.696 | 93.0 |  |
