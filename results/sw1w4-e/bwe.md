# bwe

step **256,409,600** · 73 evals · trailing **93.4** · peak **93.58** @255,918,080 · sef **100.0** · best30 **94.4** @255,918,080

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 4 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 256409600 |
| min_checkpoint_score | 0.0 |
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
| seed | 8 |
| torch_threads | 1 |

## Resumes

Resumed at 255,213,568

![bwe](bwe.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 255229952 | 93.15 | 92.41 | 60.0 | 95.0 | 179.985 | 88.0 |  |
| 255246336 | 92.0 | 91.99 | 22.0 | 95.0 | 180.735 | 90.0 |  |
| 255262720 | 91.44 | 92.03 | 24.0 | 95.0 | 176.285 | 86.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 256229376 | 94.47 | 93.38 | 57.0 | 95.0 | 191.435 | 98.0 |  |
| 256245760 | 93.84 | 93.34 | 28.0 | 95.0 | 189.72 | 97.0 |  |
| 256262144 | 92.18 | 93.44 | 22.0 | 95.0 | 183.99 | 93.0 |  |
| 256278528 | 93.07 | 93.42 | 36.0 | 95.0 | 184.925 | 93.0 |  |
| 256294912 | 91.0 | 93.28 | 18.0 | 95.0 | 179.69 | 90.0 |  |
| 256311296 | 94.34 | 93.34 | 59.0 | 95.0 | 187.145 | 94.0 |  |
| 256327680 | 94.54 | 93.37 | 58.0 | 95.0 | 190.465 | 97.0 |  |
| 256344064 | 94.04 | 93.4 | 18.0 | 95.0 | 188.97 | 96.0 |  |
| 256360448 | 93.64 | 93.33 | 28.0 | 95.0 | 185.45 | 93.0 |  |
| 256376832 | 94.45 | 93.49 | 54.0 | 95.0 | 190.375 | 97.0 |  |
| 256393216 | 93.35 | 93.51 | 20.0 | 95.0 | 188.235 | 96.0 |  |
| 256409600 | 93.77 | 93.4 | 35.0 | 95.0 | 183.59 | 91.0 |  |
