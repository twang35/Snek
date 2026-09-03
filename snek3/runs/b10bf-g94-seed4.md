# b10bf-g94-seed4

step **20,824,064** · 1260 evals · trailing **93.27** · peak **94.07** @16,269,312 · sef **15.3** · best30 **85.5** @16,236,544

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.94 |
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
| ppo_horizon | 12.7 |
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

![b10bf-g94-seed4](b10bf-g94-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 2.82 | 2.82 | 0.0 | 8.0 | 0.79 | 0.0 |  |
| 32768 | 2.73 | 2.77 | 0.0 | 18.0 | 2.23 | 0.0 |  |
| 49152 | 27.45 | 11.0 | 0.0 | 89.0 | 23.845 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 20463616 | 93.53 | 92.84 | 45.0 | 95.0 | 166.66 | 74.0 |  |
| 20480000 | 93.47 | 92.88 | 34.0 | 95.0 | 174.56 | 82.0 |  |
| 20496384 | 93.64 | 92.89 | 14.0 | 95.0 | 180.7 | 88.0 |  |
| 20512768 | 92.93 | 92.9 | 28.0 | 95.0 | 178.0 | 86.0 |  |
| 20529152 | 93.2 | 92.91 | 6.0 | 95.0 | 173.295 | 81.0 |  |
| 20545536 | 94.2 | 93.02 | 61.0 | 95.0 | 179.27 | 86.0 |  |
| 20561920 | 93.31 | 93.25 | 28.0 | 95.0 | 182.36 | 90.0 |  |
| 20578304 | 93.74 | 93.28 | 72.0 | 95.0 | 176.82 | 84.0 |  |
| 20594688 | 94.65 | 93.31 | 90.0 | 95.0 | 179.72 | 86.0 |  |
| 20660224 | 92.8 | 93.3 | 14.0 | 95.0 | 152.995 | 61.0 |  |
| 20725760 | 92.07 | 93.27 | 40.0 | 95.0 | 144.305 | 53.0 |  |
| 20824064 | 92.06 | 93.27 | 32.0 | 95.0 | 155.24 | 64.0 |  |
