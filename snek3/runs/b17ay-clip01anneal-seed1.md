# b17ay-clip01anneal-seed1

step **7,634,944** · 461 evals · trailing **89.0** · peak **94.22** @3,883,008 · sef **37.7** · best30 **92.2** @6,193,152

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
| ppo_clip | 0.1 |
| ppo_clip_final | 0.02 |
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

![b17ay-clip01anneal-seed1](b17ay-clip01anneal-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.15 | 1.15 | 0.0 | 3.0 | -3.815 | 0.0 |  |
| 32768 | 29.37 | 15.26 | 10.0 | 62.0 | 26.286 | 0.0 |  |
| 49152 | 36.81 | 22.44 | 13.0 | 61.0 | 31.718 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 7421952 | 90.37 | 89.04 | 36.0 | 95.0 | 177.038 | 88.0 |  |
| 7438336 | 90.1 | 89.2 | 14.0 | 95.0 | 175.775 | 87.0 |  |
| 7454720 | 92.72 | 89.48 | 44.0 | 95.0 | 183.381 | 92.0 |  |
| 7471104 | 93.02 | 89.25 | 43.0 | 95.0 | 186.689 | 95.0 |  |
| 7487488 | 90.69 | 89.3 | 44.0 | 95.0 | 177.399 | 88.0 |  |
| 7503872 | 92.13 | 89.25 | 45.0 | 95.0 | 182.798 | 92.0 |  |
| 7520256 | 92.18 | 89.21 | 20.0 | 95.0 | 183.854 | 93.0 |  |
| 7536640 | 93.43 | 89.41 | 55.0 | 95.0 | 187.09 | 95.0 |  |
| 7553024 | 91.63 | 89.01 | 54.0 | 95.0 | 177.307 | 87.0 |  |
| 7602176 | 91.16 | 89.13 | 44.0 | 95.0 | 179.848 | 90.0 |  |
| 7618560 | 90.93 | 89.18 | 42.0 | 95.0 | 178.628 | 89.0 |  |
| 7634944 | 88.12 | 89.0 | 29.0 | 95.0 | 169.834 | 83.0 |  |
