# b10ak-g85-seed3

step **50,003,968** · 3052 evals · trailing **82.63** · peak **93.0** @16,138,240 · sef **0.0** · best30 **66.6** @34,357,248

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.85 |
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
| ppo_horizon | 6.0 |
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

![b10ak-g85-seed3](b10ak-g85-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.03 | 0.03 | 0.0 | 1.0 | -0.47 | 0.0 |  |
| 32768 | 1.39 | 0.71 | 0.0 | 7.0 | 0.89 | 0.0 |  |
| 49152 | 20.22 | 23.82 | 2.0 | 63.0 | 17.785 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 69.83 | 82.05 | 15.0 | 95.0 | 91.725 | 24.0 |  |
| 49840128 | 80.51 | 82.13 | 20.0 | 95.0 | 110.225 | 31.0 |  |
| 49856512 | 68.25 | 82.47 | 8.0 | 95.0 | 81.145 | 15.0 |  |
| 49872896 | 81.06 | 82.84 | 14.0 | 95.0 | 106.84 | 27.0 |  |
| 49889280 | 81.24 | 82.32 | 16.0 | 95.0 | 113.94 | 34.0 |  |
| 49905664 | 78.08 | 82.62 | 19.0 | 95.0 | 108.565 | 32.0 |  |
| 49922048 | 86.42 | 82.51 | 15.0 | 95.0 | 117.535 | 32.0 |  |
| 49938432 | 85.33 | 82.64 | 10.0 | 95.0 | 113.37 | 29.0 |  |
| 49954816 | 82.24 | 82.61 | 16.0 | 95.0 | 99.155 | 18.0 |  |
| 49971200 | 78.09 | 82.79 | 14.0 | 95.0 | 99.755 | 23.0 |  |
| 49987584 | 83.59 | 82.48 | 14.0 | 95.0 | 96.75 | 14.0 |  |
| 50003968 | 79.88 | 82.63 | 14.0 | 95.0 | 97.79 | 19.0 |  |
