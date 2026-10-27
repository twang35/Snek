# b10an-g90-seed2

step **50,003,968** · 3052 evals · trailing **91.32** · peak **93.65** @12,058,624 · sef **2.4** · best30 **75.8** @49,561,600

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.9 |
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
| ppo_horizon | 8.5 |
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

![b10an-g90-seed2](b10an-g90-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.54 | 2.54 | 1.0 | 8.0 | -0.795 | 0.0 |  |
| 32768 | 11.28 | 6.91 | 1.0 | 24.0 | 6.46 | 0.0 |  |
| 49152 | 20.76 | 11.53 | 0.0 | 45.0 | 16.03 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 91.71 | 91.95 | 18.0 | 95.0 | 152.9 | 62.0 |  |
| 49840128 | 91.96 | 92.39 | 55.0 | 95.0 | 161.11 | 70.0 |  |
| 49856512 | 92.67 | 92.26 | 55.0 | 95.0 | 161.82 | 70.0 |  |
| 49872896 | 92.13 | 91.85 | 30.0 | 95.0 | 173.175 | 82.0 |  |
| 49889280 | 92.56 | 91.72 | 45.0 | 95.0 | 177.63 | 86.0 |  |
| 49905664 | 91.65 | 91.77 | 47.0 | 95.0 | 156.775 | 66.0 |  |
| 49922048 | 92.45 | 91.9 | 43.0 | 95.0 | 175.485 | 84.0 |  |
| 49938432 | 89.71 | 91.6 | 32.0 | 95.0 | 156.78 | 68.0 |  |
| 49954816 | 91.84 | 91.45 | 31.0 | 95.0 | 154.93 | 64.0 |  |
| 49971200 | 90.29 | 91.5 | 8.0 | 95.0 | 159.44 | 70.0 |  |
| 49987584 | 90.75 | 91.42 | 41.0 | 95.0 | 152.89 | 63.0 |  |
| 50003968 | 90.22 | 91.32 | 34.0 | 95.0 | 151.41 | 62.0 |  |
