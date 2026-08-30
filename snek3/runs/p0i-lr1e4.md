# p0i-lr1e4

step **10,010,624** · 611 evals · trailing **92.28** · peak **94.14** @6,717,440 · sef **40.8** · best30 **95.0** @9,814,016

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
| max_steps | 10000000 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_clip | 0.2 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0001 |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

## Resumes

Resumed at 3,014,656

![p0i-lr1e4](p0i-lr1e4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 11.42 | 17.48 | 1.0 | 39.0 | 10.155 | 0.0 |  |
| 32768 | 16.77 | 16.77 | 1.0 | 32.0 | 11.905 | 0.0 |  |
| 49152 | 17.63 | 17.2 | 4.0 | 37.0 | 12.63 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 9830400 | 87.79 | 92.47 | 27.0 | 95.0 | 173.855 | 87.0 |  |
| 9846784 | 90.75 | 92.17 | 29.0 | 95.0 | 178.805 | 89.0 |  |
| 9863168 | 86.18 | 89.94 | 27.0 | 95.0 | 170.255 | 85.0 |  |
| 9879552 | 83.97 | 89.32 | 26.0 | 95.0 | 165.06 | 82.0 |  |
| 9895936 | 84.18 | 90.4 | 26.0 | 95.0 | 164.275 | 81.0 |  |
| 9912320 | 92.95 | 92.59 | 29.0 | 95.0 | 184.985 | 93.0 |  |
| 9928704 | 88.36 | 92.43 | 14.0 | 95.0 | 170.445 | 83.0 |  |
| 9945088 | 92.88 | 92.31 | 29.0 | 95.0 | 183.92 | 92.0 |  |
| 9961472 | 87.01 | 92.37 | 27.0 | 95.0 | 170.09 | 84.0 |  |
| 9977856 | 93.32 | 92.58 | 59.0 | 95.0 | 184.36 | 92.0 |  |
| 9994240 | 92.3 | 92.16 | 46.0 | 95.0 | 181.35 | 90.0 |  |
| 10010624 | 92.67 | 92.28 | 58.0 | 95.0 | 183.71 | 92.0 |  |
