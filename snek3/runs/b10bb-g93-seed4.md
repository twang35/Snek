# b10bb-g93-seed4

step **20,283,392** · 1229 evals · trailing **90.33** · peak **94.06** @15,876,096 · sef **4.3** · best30 **81.2** @16,056,320

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
| seed | 4 |
| torch_threads | 1 |

![b10bb-g93-seed4](b10bb-g93-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 3.06 | 3.06 | 0.0 | 10.0 | 0.58 | 0.0 |  |
| 32768 | 0.95 | 2.0 | 0.0 | 7.0 | 0.45 | 0.0 |  |
| 49152 | 1.56 | 1.86 | 0.0 | 27.0 | 1.06 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 19955712 | 86.95 | 90.13 | 3.0 | 95.0 | 110.33 | 24.0 |  |
| 19972096 | 88.03 | 90.06 | 14.0 | 95.0 | 113.355 | 26.0 |  |
| 19988480 | 84.25 | 90.05 | 15.0 | 95.0 | 109.62 | 26.0 |  |
| 20004864 | 90.11 | 90.35 | 8.0 | 95.0 | 118.465 | 29.0 |  |
| 20021248 | 89.41 | 90.39 | 5.0 | 95.0 | 132.645 | 44.0 |  |
| 20037632 | 88.61 | 90.32 | 13.0 | 95.0 | 119.95 | 32.0 |  |
| 20054016 | 88.68 | 90.28 | 28.0 | 95.0 | 110.025 | 22.0 |  |
| 20119552 | 90.31 | 90.28 | 42.0 | 95.0 | 130.56 | 41.0 |  |
| 20135936 | 89.73 | 90.27 | 12.0 | 95.0 | 143.955 | 55.0 |  |
| 20152320 | 92.12 | 90.34 | 18.0 | 95.0 | 154.305 | 63.0 |  |
| 20168704 | 92.5 | 90.35 | 34.0 | 95.0 | 155.68 | 64.0 |  |
| 20283392 | 92.92 | 90.33 | 70.0 | 95.0 | 147.145 | 55.0 |  |
