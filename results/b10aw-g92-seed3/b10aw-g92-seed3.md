# b10aw-g92-seed3

step **50,003,968** · 3052 evals · trailing **93.1** · peak **93.89** @19,939,328 · sef **9.4** · best30 **84.2** @38,928,384

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.92 |
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
| ppo_horizon | 10.2 |
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

![b10aw-g92-seed3](b10aw-g92-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.06 | 0.06 | 0.0 | 1.0 | -0.53 | 0.0 |  |
| 32768 | 2.71 | 1.39 | 0.0 | 10.0 | 1.49 | 0.0 |  |
| 49152 | 13.5 | 5.42 | 0.0 | 31.0 | 10.885 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 91.33 | 93.32 | 54.0 | 95.0 | 147.5 | 57.0 |  |
| 49840128 | 92.32 | 93.28 | 49.0 | 95.0 | 148.49 | 57.0 |  |
| 49856512 | 93.35 | 93.23 | 69.0 | 95.0 | 154.54 | 62.0 |  |
| 49872896 | 93.45 | 93.21 | 65.0 | 95.0 | 162.51 | 70.0 |  |
| 49889280 | 93.05 | 93.19 | 63.0 | 95.0 | 160.12 | 68.0 |  |
| 49905664 | 92.23 | 93.18 | 58.0 | 95.0 | 151.34 | 60.0 |  |
| 49922048 | 92.72 | 93.19 | 57.0 | 95.0 | 149.885 | 58.0 |  |
| 49938432 | 92.76 | 93.15 | 69.0 | 95.0 | 152.865 | 61.0 |  |
| 49954816 | 93.19 | 93.1 | 35.0 | 95.0 | 161.345 | 69.0 |  |
| 49971200 | 92.52 | 93.11 | 49.0 | 95.0 | 151.675 | 60.0 |  |
| 49987584 | 92.67 | 93.06 | 18.0 | 95.0 | 155.85 | 64.0 |  |
| 50003968 | 94.03 | 93.1 | 84.0 | 95.0 | 166.165 | 73.0 |  |
