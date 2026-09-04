# b11aa-lr4e5-seed1

step **32,571,392** · 1988 evals · trailing **92.46** · peak **93.65** @5,832,704 · sef **47.1** · best30 **93.4** @28,180,480

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
| ppo_gae_lambda | 0.99 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 50.3 |
| ppo_learning_rate | 4e-05 |
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

![b11aa-lr4e5-seed1](b11aa-lr4e5-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.78 | 0.78 | 0.0 | 4.0 | 0.28 | 0.0 |  |
| 32768 | 6.39 | 3.58 | 1.0 | 27.0 | 4.09 | 0.0 |  |
| 49152 | 13.77 | 8.58 | 2.0 | 32.0 | 8.77 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 32391168 | 93.85 | 91.74 | 58.0 | 95.0 | 188.87 | 96.0 |  |
| 32407552 | 93.9 | 92.09 | 44.0 | 95.0 | 189.915 | 97.0 |  |
| 32423936 | 93.06 | 89.7 | 56.0 | 95.0 | 185.095 | 93.0 |  |
| 32440320 | 94.17 | 90.93 | 56.0 | 95.0 | 189.19 | 96.0 |  |
| 32456704 | 93.01 | 92.36 | 55.0 | 95.0 | 185.045 | 93.0 |  |
| 32473088 | 92.57 | 89.92 | 44.0 | 95.0 | 184.605 | 93.0 |  |
| 32489472 | 91.78 | 90.26 | 43.0 | 95.0 | 181.825 | 91.0 |  |
| 32505856 | 93.93 | 89.54 | 54.0 | 95.0 | 189.945 | 97.0 |  |
| 32522240 | 91.66 | 90.64 | 46.0 | 95.0 | 177.725 | 87.0 |  |
| 32538624 | 92.8 | 89.55 | 56.0 | 95.0 | 184.835 | 93.0 |  |
| 32555008 | 92.31 | 92.03 | 45.0 | 95.0 | 184.345 | 93.0 |  |
| 32571392 | 92.7 | 92.46 | 57.0 | 95.0 | 183.74 | 92.0 |  |
