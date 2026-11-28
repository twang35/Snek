# b33aq-win100-seed1

step **50,003,968** · 1526 evals · trailing **94.52** · peak **94.92** @41,058,304 · sef **92.1** · best30 **99.8** @41,189,376

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
| init_from | None |
| max_steps | 50003968 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_anneal_fraction | 0.5 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_discount_final | 0.999 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | 0.001 |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.95 |
| ppo_gae_lambda_final | 0.999 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 16.8 |
| ppo_horizon_final | 500.3 |
| ppo_learning_rate | 0.00025 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 512 |
| ppo_normalize_adv | True |
| ppo_rollout | 256 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 32768 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 1 |
| torch_threads | 1 |

![b33aq-win100-seed1](b33aq-win100-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 7.19 | 7.19 | 0.0 | 16.0 | 2.266 | 0.0 |  |
| 65536 | 27.03 | 20.93 | 0.0 | 54.0 | 22.483 | 0.0 |  |
| 98304 | 28.58 | 17.88 | 1.0 | 53.0 | 23.573 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 49643520 | 94.21 | 94.54 | 16.0 | 95.0 | 191.985 | 99.0 |  |
| 49676288 | 94.68 | 94.55 | 63.0 | 95.0 | 192.447 | 99.0 |  |
| 49709056 | 94.53 | 94.55 | 66.0 | 95.0 | 191.293 | 98.0 |  |
| 49741824 | 95.0 | 94.56 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 49774592 | 94.23 | 94.56 | 18.0 | 95.0 | 191.995 | 99.0 |  |
| 49807360 | 94.91 | 94.57 | 86.0 | 95.0 | 192.674 | 99.0 |  |
| 49840128 | 95.0 | 94.57 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 49872896 | 95.0 | 94.59 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 49905664 | 94.23 | 94.53 | 18.0 | 95.0 | 191.995 | 99.0 |  |
| 49938432 | 93.62 | 94.52 | 22.0 | 95.0 | 189.395 | 97.0 |  |
| 49971200 | 95.0 | 94.57 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 50003968 | 94.51 | 94.52 | 46.0 | 95.0 | 192.271 | 99.0 |  |
