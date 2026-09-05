# b14p-roll256-seed4

step **49,577,984** · 1511 evals · trailing **94.55** · peak **94.73** @46,039,040 · sef **89.0** · best30 **98.7** @45,547,520

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
| 49152000 | 95.0 | 94.6 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49184768 | 94.15 | 94.51 | 58.0 | 95.0 | 190.165 | 97.0 |  |
| 49217536 | 94.54 | 94.54 | 56.0 | 95.0 | 191.55 | 98.0 |  |
| 49250304 | 94.66 | 94.56 | 70.0 | 95.0 | 191.67 | 98.0 |  |
| 49283072 | 94.93 | 94.56 | 88.0 | 95.0 | 192.935 | 99.0 |  |
| 49315840 | 94.24 | 94.53 | 59.0 | 95.0 | 189.26 | 96.0 |  |
| 49348608 | 94.67 | 94.56 | 62.0 | 95.0 | 192.675 | 99.0 |  |
| 49381376 | 94.76 | 94.52 | 78.0 | 95.0 | 190.775 | 97.0 |  |
| 49414144 | 95.0 | 94.56 | 95.0 | 95.0 | 194.0 | 100.0 |  |
| 49512448 | 94.65 | 94.55 | 77.0 | 95.0 | 190.665 | 97.0 |  |
| 49545216 | 94.35 | 94.54 | 55.0 | 95.0 | 191.36 | 98.0 |  |
| 49577984 | 94.55 | 94.55 | 71.0 | 95.0 | 191.56 | 98.0 |  |
