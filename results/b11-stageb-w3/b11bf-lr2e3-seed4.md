# b11bf-lr2e3-seed4

step **50,003,968** · 3052 evals · trailing **92.05** · peak **94.18** @15,548,416 · sef **91.9** · best30 **97.2** @43,466,752

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
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 0.002 |
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

![b11bf-lr2e3-seed4](b11bf-lr2e3-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.39 | 1.39 | 0.0 | 7.0 | -0.19 | 0.0 |  |
| 32768 | 22.73 | 17.04 | 3.0 | 45.0 | 18.18 | 0.0 |  |
| 49152 | 26.99 | 14.19 | 6.0 | 50.0 | 21.99 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49823744 | 92.6 | 91.96 | 8.0 | 95.0 | 185.54 | 94.0 |  |
| 49840128 | 93.07 | 91.93 | 18.0 | 95.0 | 187.095 | 95.0 |  |
| 49856512 | 90.62 | 91.94 | 13.0 | 95.0 | 176.415 | 87.0 |  |
| 49872896 | 91.83 | 91.95 | 30.0 | 95.0 | 183.55 | 93.0 |  |
| 49889280 | 91.76 | 92.02 | 15.0 | 95.0 | 183.57 | 93.0 |  |
| 49905664 | 93.1 | 91.97 | 48.0 | 95.0 | 180.885 | 89.0 |  |
| 49922048 | 91.1 | 91.91 | 1.0 | 95.0 | 181.055 | 91.0 |  |
| 49938432 | 91.32 | 91.97 | 7.0 | 95.0 | 178.2 | 88.0 |  |
| 49954816 | 93.88 | 91.98 | 54.0 | 95.0 | 188.855 | 96.0 |  |
| 49971200 | 93.87 | 92.01 | 17.0 | 95.0 | 189.885 | 97.0 |  |
| 49987584 | 91.7 | 92.05 | 1.0 | 95.0 | 184.73 | 94.0 |  |
| 50003968 | 92.81 | 92.05 | 6.0 | 95.0 | 182.72 | 91.0 |  |
