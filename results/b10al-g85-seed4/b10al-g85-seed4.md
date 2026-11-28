# b10al-g85-seed4

step **50,003,968** · 3052 evals · trailing **91.2** · peak **93.24** @38,191,104 · sef **0.1** · best30 **68.0** @38,273,024

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
| seed | 4 |
| torch_threads | 1 |

![b10al-g85-seed4](b10al-g85-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.53 | 2.53 | 0.0 | 11.0 | 1.04 | 0.0 |  |
| 32768 | 6.02 | 4.27 | 0.0 | 25.0 | 4.98 | 0.0 |  |
| 49152 | 33.72 | 23.54 | 0.0 | 76.0 | 30.16 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.39 | 91.85 | 36.0 | 95.0 | 162.49 | 71.0 |  |
| 49840128 | 93.0 | 91.82 | 25.0 | 95.0 | 167.08 | 75.0 |  |
| 49856512 | 91.37 | 91.82 | 33.0 | 95.0 | 158.485 | 68.0 |  |
| 49872896 | 93.31 | 91.88 | 63.0 | 95.0 | 166.44 | 74.0 |  |
| 49889280 | 91.59 | 91.72 | 21.0 | 95.0 | 164.585 | 74.0 |  |
| 49905664 | 92.54 | 91.79 | 32.0 | 95.0 | 159.61 | 68.0 |  |
| 49922048 | 91.82 | 91.67 | 35.0 | 95.0 | 154.91 | 64.0 |  |
| 49938432 | 88.19 | 91.48 | 18.0 | 95.0 | 135.18 | 48.0 |  |
| 49954816 | 91.8 | 91.3 | 15.0 | 95.0 | 143.99 | 53.0 |  |
| 49971200 | 90.51 | 91.22 | 8.0 | 95.0 | 136.73 | 47.0 |  |
| 49987584 | 92.55 | 91.26 | 66.0 | 95.0 | 146.775 | 55.0 |  |
| 50003968 | 91.07 | 91.2 | 24.0 | 95.0 | 141.27 | 51.0 |  |
