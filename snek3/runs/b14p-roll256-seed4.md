# b14p-roll256-seed4

step **40,108,032** · 1220 evals · trailing **94.39** · peak **94.47** @35,782,656 · sef **86.4** · best30 **97.9** @36,700,160

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 32768 |
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
| ppo_rollout | 256 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 32768 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b14p-roll256-seed4](b14p-roll256-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 7.2 | 7.2 | 1.0 | 18.0 | 2.56 | 0.0 |  |
| 65536 | 22.79 | 14.99 | 2.0 | 44.0 | 18.33 | 0.0 |  |
| 98304 | 26.65 | 18.88 | 2.0 | 49.0 | 21.65 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 39616512 | 94.28 | 94.37 | 36.0 | 95.0 | 191.245 | 98.0 |  |
| 39649280 | 94.63 | 94.24 | 80.0 | 95.0 | 190.645 | 97.0 |  |
| 39682048 | 93.95 | 94.31 | 28.0 | 95.0 | 190.915 | 98.0 |  |
| 39714816 | 95.0 | 94.36 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 39747584 | 93.96 | 94.28 | 63.0 | 95.0 | 187.985 | 95.0 |  |
| 39780352 | 94.38 | 94.29 | 58.0 | 95.0 | 191.39 | 98.0 |  |
| 39813120 | 93.87 | 94.31 | 58.0 | 95.0 | 187.895 | 95.0 |  |
| 39845888 | 93.59 | 94.35 | 2.0 | 95.0 | 188.61 | 96.0 |  |
| 39878656 | 93.95 | 94.36 | 63.0 | 95.0 | 187.975 | 95.0 |  |
| 39976960 | 94.9 | 94.38 | 88.0 | 95.0 | 191.91 | 98.0 |  |
| 40075264 | 94.42 | 94.38 | 57.0 | 95.0 | 191.43 | 98.0 |  |
| 40108032 | 94.63 | 94.39 | 58.0 | 95.0 | 192.635 | 99.0 |  |
