# b10az-g93-seed2

step **21,102,592** · 1285 evals · trailing **92.11** · peak **93.5** @15,187,968 · sef **6.8** · best30 **81.2** @17,022,976

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.93 |
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
| ppo_horizon | 11.3 |
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

![b10az-g93-seed2](b10az-g93-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.37 | 2.37 | 0.0 | 6.0 | -1.145 | 0.0 |  |
| 32768 | 9.47 | 5.92 | 0.0 | 21.0 | 4.83 | 0.0 |  |
| 49152 | 20.68 | 10.84 | 0.0 | 43.0 | 15.815 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 20873216 | 94.06 | 91.58 | 83.0 | 95.0 | 175.15 | 82.0 |  |
| 20889600 | 94.46 | 91.57 | 73.0 | 95.0 | 183.51 | 90.0 |  |
| 20905984 | 94.56 | 91.54 | 82.0 | 95.0 | 186.595 | 93.0 |  |
| 20922368 | 93.79 | 91.98 | 61.0 | 95.0 | 178.86 | 86.0 |  |
| 20938752 | 93.39 | 91.53 | 10.0 | 95.0 | 179.455 | 87.0 |  |
| 20955136 | 93.09 | 91.54 | 67.0 | 95.0 | 175.175 | 83.0 |  |
| 20971520 | 94.59 | 91.75 | 88.0 | 95.0 | 183.64 | 90.0 |  |
| 20987904 | 94.38 | 91.63 | 78.0 | 95.0 | 183.43 | 90.0 |  |
| 21004288 | 94.38 | 91.59 | 74.0 | 95.0 | 183.43 | 90.0 |  |
| 21069824 | 94.72 | 92.3 | 90.0 | 95.0 | 184.765 | 91.0 |  |
| 21086208 | 94.24 | 91.87 | 80.0 | 95.0 | 180.305 | 87.0 |  |
| 21102592 | 94.0 | 92.11 | 82.0 | 95.0 | 173.1 | 80.0 |  |
