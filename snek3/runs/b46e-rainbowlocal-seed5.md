# b46e-rainbowlocal-seed5

step **29,000** · 29 evals · trailing **56.24** · peak **56.24** @29,000 · sef **0.0** · best30 **0.0** @29,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | rainbow |
| batch_size | 128 |
| beta_anneal_steps | 300000 |
| btr_blocks | 3 |
| btr_layer_norm | False |
| btr_residual | False |
| btr_spectral_norm | True |
| collect_envs | 1 |
| discount | 0.99 |
| dist_atoms | 51 |
| dist_embedding | 64 |
| dist_kappa | 1.0 |
| dist_policy_samples | 8 |
| dist_quantiles | 32 |
| dist_tau_prime_samples | 8 |
| dist_tau_samples | 8 |
| dist_v_max | 110.0 |
| dist_v_min | -10.0 |
| epsilon_anneal_steps | 1 |
| epsilon_schedule | eval |
| eval_interval | 1000 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| fork_branches | 4 |
| fork_max_steps | 60 |
| fork_min_length | 85 |
| fork_prob | 0.5 |
| gradient_clipping | 0.0 |
| graph_eval_episodes | 100 |
| guided_fraction | 0.8 |
| init_from | None |
| initial_collect_steps | 2000 |
| initial_epsilon | 0.4 |
| is_beta | 0.4 |
| is_beta_final | 1.0 |
| is_normalization | mean |
| is_weights | True |
| learning_rate | 1e-05 |
| max_steps | 3000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.002 |
| munchausen_alpha | 0.0 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 3 |
| priority_exponent | 0.6 |
| rainbow_double | True |
| rainbow_dueling | True |
| rainbow_epsilon_decay | linear |
| rainbow_epsilon_zero_at | 0.0 |
| rainbow_head | c51 |
| rainbow_munchausen_logpi | target |
| rainbow_noisy | True |
| rainbow_noisy_sigma | 0.5 |
| rainbow_prefill_epsilon | random |
| rainbow_stream_width | 512 |
| replay_buffer_max_length | 100000 |
| replay_ratio | 1.0 |
| reset_alpha | 0.5 |
| reset_anneal_gamma |  |
| reset_anneal_n_step |  |
| reset_anneal_steps | 10000 |
| reset_interval | 0 |
| reset_stop_after | 0 |
| seed | 5 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b46e-rainbowlocal-seed5](b46e-rainbowlocal-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 13.32 | 13.32 | 1.0 | 37.0 | 8.731 | 0.0 | 0.4 |
| 2000 | 20.15 | 16.73 | 1.0 | 38.0 | 15.302 | 0.0 | 0.4 |
| 3000 | 20.45 | 17.97 | 2.0 | 33.0 | 15.472 | 0.0 | 0.1 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 18000 | 74.25 | 42.14 | 25.0 | 90.0 | 72.017 | 0.0 | 0.0125 |
| 19000 | 76.53 | 43.95 | 46.0 | 90.0 | 74.756 | 0.0 | 0.01248 |
| 20000 | 76.92 | 45.6 | 2.0 | 91.0 | 75.062 | 0.0 | 0.01248 |
| 21000 | 77.79 | 47.13 | 12.0 | 90.0 | 76.119 | 0.0 | 0.01248 |
| 22000 | 78.21 | 48.54 | 60.0 | 89.0 | 76.695 | 0.0 | 0.01249 |
| 23000 | 79.14 | 49.87 | 27.0 | 92.0 | 77.507 | 0.0 | 0.01249 |
| 24000 | 77.56 | 51.03 | 7.0 | 92.0 | 75.829 | 0.0 | 0.01249 |
| 25000 | 80.7 | 52.21 | 2.0 | 95.0 | 80.143 | 1.0 | 0.01249 |
| 26000 | 80.42 | 53.3 | 46.0 | 92.0 | 78.89 | 0.0 | 0.01249 |
| 27000 | 80.69 | 54.31 | 50.0 | 95.0 | 80.113 | 1.0 | 0.01248 |
| 28000 | 81.53 | 55.28 | 58.0 | 92.0 | 79.953 | 0.0 | 0.01248 |
| 29000 | 83.06 | 56.24 | 68.0 | 95.0 | 82.61 | 1.0 | 0.01247 |
