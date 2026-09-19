# b9ap-lam85-seed4

step **50,003,968** · 3052 evals · trailing **93.34** · peak **94.24** @40,878,080 · sef **89.4** · best30 **95.9** @38,731,776

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
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.85 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 6.3 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b9ap-lam85-seed4](b9ap-lam85-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 3.37 | 3.37 | 0.0 | 13.0 | 1.52 | 0.0 |  |
| 32768 | 5.09 | 4.23 | 0.0 | 19.0 | 4.41 | 0.0 |  |
| 49152 | 30.88 | 25.54 | 2.0 | 60.0 | 26.51 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 90.4 | 93.52 | 34.0 | 95.0 | 166.515 | 77.0 |  |
| 49840128 | 93.86 | 93.62 | 66.0 | 95.0 | 183.86 | 91.0 |  |
| 49856512 | 94.22 | 93.6 | 72.0 | 95.0 | 189.24 | 96.0 |  |
| 49872896 | 94.11 | 93.57 | 73.0 | 95.0 | 185.15 | 92.0 |  |
| 49889280 | 94.79 | 93.64 | 86.0 | 95.0 | 190.805 | 97.0 |  |
| 49905664 | 94.13 | 93.67 | 72.0 | 95.0 | 185.125 | 92.0 |  |
| 49922048 | 93.25 | 93.62 | 70.0 | 95.0 | 179.315 | 87.0 |  |
| 49938432 | 91.67 | 93.3 | 22.0 | 95.0 | 170.77 | 80.0 |  |
| 49954816 | 91.4 | 93.42 | 22.0 | 95.0 | 165.48 | 75.0 |  |
| 49971200 | 89.55 | 93.32 | 41.0 | 95.0 | 153.68 | 65.0 |  |
| 49987584 | 94.17 | 93.34 | 79.0 | 95.0 | 184.125 | 91.0 |  |
| 50003968 | 93.0 | 93.34 | 63.0 | 95.0 | 179.065 | 87.0 |  |
