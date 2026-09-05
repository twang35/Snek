# b19m-adameps1e8-seed1

step **6,406,144** · 386 evals · trailing **89.49** · peak **92.33** @1,638,400 · sef **54.4** · best30 **89.0** @3,981,312

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
| ppo_adam_epsilon | 1e-08 |
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

![b19m-adameps1e8-seed1](b19m-adameps1e8-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 17.24 | 26.66 | 3.0 | 40.0 | 15.522 | 0.0 |  |
| 32768 | 47.92 | 34.13 | 12.0 | 91.0 | 42.908 | 0.0 |  |
| 49152 | 33.76 | 31.37 | 6.0 | 57.0 | 28.735 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 6144000 | 92.86 | 89.72 | 47.0 | 95.0 | 183.539 | 92.0 |  |
| 6160384 | 90.65 | 89.36 | 12.0 | 95.0 | 178.36 | 89.0 |  |
| 6176768 | 92.9 | 89.48 | 52.0 | 95.0 | 185.572 | 94.0 |  |
| 6193152 | 92.89 | 89.53 | 48.0 | 95.0 | 184.555 | 93.0 |  |
| 6209536 | 91.37 | 89.54 | 35.0 | 95.0 | 178.052 | 88.0 |  |
| 6225920 | 87.67 | 89.66 | 37.0 | 95.0 | 169.365 | 83.0 |  |
| 6242304 | 89.02 | 89.4 | 37.0 | 95.0 | 171.691 | 84.0 |  |
| 6258688 | 88.84 | 89.45 | 41.0 | 95.0 | 170.553 | 83.0 |  |
| 6340608 | 89.02 | 89.41 | 41.0 | 95.0 | 171.733 | 84.0 |  |
| 6373376 | 91.92 | 89.42 | 52.0 | 95.0 | 181.618 | 91.0 |  |
| 6389760 | 92.0 | 89.52 | 57.0 | 95.0 | 181.684 | 91.0 |  |
| 6406144 | 89.7 | 89.49 | 34.0 | 95.0 | 173.417 | 85.0 |  |
